(function () {
  'use strict';
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  const STANDARD_FONT_DATA_URL = 'https://unpkg.com/pdfjs-dist@3.11.174/standard_fonts/';

  const $ = id => document.getElementById(id);
  // Some option controls have been removed from the page; fall back to the
  // original defaults instead of throwing when one is absent.
  const checked = (id, dflt) => { const el = $(id); return el ? el.checked : dflt; };
  const textOf = (id, dflt) => { const el = $(id); return el ? el.value : dflt; };
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const normName = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  const alnum = s => String(s).toUpperCase().replace(/[^A-Z0-9]/g, '');

  // docs: { name, short, numPages, pages:[{text, norm, hasText}], fieldList:[{name,value,page,label,section}],
  //         fields:{name:Set}, text, error }
  const state = { docs: [], refIdx: -1, refAuto: true, mode: 'jobbook', rowsB: [], jb: null, jbRows: [], reviews: new Map(), reviewNotes: new Map(), reviewsFor: '', reviewPosition: null, reviewScope: 'all', ignoredDocTypes: [], baseline: null, skip: { dates: true, initials: false, names: false, company: false, version: false, yesno: false, blanks: false } };

  // Keep the original PDF blobs, as well as their extracted text, in IndexedDB.
  // localStorage is deliberately not used here: a normal job book can be far too
  // large for it.  The blob is restored before the viewer is opened, so revisiting
  // this page does not leave it with page-number placeholders and no PDF to draw.
  const STORE_DB = 'snc-persisted-documents';
  const STORE_DOCS = 'documents';
  const STORE_SESSION = 'session';
  const SESSION_KEY = 'current';
  let dbPromise = null;

  function openStore() {
    if (!('indexedDB' in window)) return Promise.reject(new Error('IndexedDB is not available'));
    if (!dbPromise) dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(STORE_DB, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_DOCS)) db.createObjectStore(STORE_DOCS, { keyPath: 'id' });
        if (!db.objectStoreNames.contains(STORE_SESSION)) db.createObjectStore(STORE_SESSION);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Could not open document storage'));
    });
    return dbPromise;
  }

  function storedDoc(d) {
    return {
      id: d.id, name: d.name, blob: d.blob, numPages: d.numPages, pages: d.pages,
      fieldList: d.fieldList, blankFields: d.blankFields, formV: d.formV, fields: Object.fromEntries(Object.entries(d.fields).map(([k, v]) => [k, [...v]])),
      text: d.text, error: d.error,
    };
  }
  function restoredDoc(d) {
    return { ...d, fields: Object.fromEntries(Object.entries(d.fields || {}).map(([k, v]) => [k, new Set(v)])) };
  }
  function transaction(db, store, mode, work) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(store, mode);
      let result;
      try { result = work(tx.objectStore(store)); } catch (e) { reject(e); return; }
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error || new Error('Document storage failed'));
      tx.onabort = () => reject(tx.error || new Error('Document storage was aborted'));
    });
  }
  async function saveDocument(d) {
    const db = await openStore();
    await transaction(db, STORE_DOCS, 'readwrite', store => store.put(storedDoc(d)));
  }
  async function saveSession() {
    const db = await openStore();
    const ref = state.docs[state.refIdx];
    await transaction(db, STORE_SESSION, 'readwrite', store => store.put({
      ids: state.docs.map(d => d.id), refId: ref && ref.id, refAuto: state.refAuto,
      reviewPosition: state.reviewPosition, reviewScope: state.reviewScope, ignoredDocTypes: state.ignoredDocTypes, skip: state.skip,
    }, SESSION_KEY));
  }
  async function removeStoredDocument(id) {
    const db = await openStore();
    await transaction(db, STORE_DOCS, 'readwrite', store => store.delete(id));
  }
  async function clearStoredDocuments() {
    const db = await openStore();
    await transaction(db, STORE_DOCS, 'readwrite', store => store.clear());
    await transaction(db, STORE_SESSION, 'readwrite', store => store.delete(SESSION_KEY));
  }
  async function restoreDocuments() {
    try {
      const db = await openStore();
      const session = await new Promise((resolve, reject) => {
        const req = db.transaction(STORE_SESSION).objectStore(STORE_SESSION).get(SESSION_KEY);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
      if (!session || !session.ids || !session.ids.length) return;
      const docs = await Promise.all(session.ids.map(id => new Promise((resolve, reject) => {
        const req = db.transaction(STORE_DOCS).objectStore(STORE_DOCS).get(id);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      })));
      state.docs = docs.filter(d => d && d.blob).map(restoredDoc);
      // Older saved sessions predate form-field rectangles. Re-extract only those
      // PDFs once, so future reviews can point to the actual source field instead
      // of guessing from a repeated label or value.
      const missingLocations = state.docs.filter(d => d.fieldList && (d.formV !== 2 || d.blankFields.some(f => !Array.isArray(f.rect)) || d.fieldList.some(f => !Array.isArray(f.rect))));
      if (missingLocations.length) {
        await Promise.all(missingLocations.map(async oldDoc => {
          try {
            const refreshed = await extract(oldDoc.blob, oldDoc.name);
            refreshed.id = oldDoc.id;
            const index = state.docs.indexOf(oldDoc);
            if (index >= 0) state.docs[index] = refreshed;
            await saveDocument(refreshed);
          } catch (e) {
            console.warn('Could not refresh saved field locations:', e);
          }
        }));
      }
      state.refAuto = session.refAuto !== false;
      state.refIdx = state.docs.findIndex(d => d.id === session.refId);
      state.reviewPosition = session.reviewPosition || null;
      state.reviewScope = ['all', 'romc', 'sif', 'other'].includes(session.reviewScope) ? session.reviewScope : 'all';
      state.ignoredDocTypes = Array.isArray(session.ignoredDocTypes) ? session.ignoredDocTypes.filter(k => typeof k === 'string') : [];
      if (session.skip && typeof session.skip === 'object') Object.keys(state.skip).forEach(k => { if (typeof session.skip[k] === 'boolean') state.skip[k] = session.skip[k]; });
      if (state.refIdx < 0) state.refAuto = true;
      renderDocs();
      run();
      if (state.docs.length) showProgress(`Restored ${state.docs.length} saved PDF${state.docs.length === 1 ? '' : 's'}.`);
    } catch (e) {
      // Private browsing, quota limits, and disabled browser storage should not
      // prevent normal one-visit use of the tool.
      console.warn('Could not restore saved PDFs:', e);
    }
  }
  const newDocumentId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);

  // Reviewer marks are kept per source field and required review target. A field
  // that must be checked in two places deliberately has two independent marks.
  const reviewKey = e => [e.sourceId || '', e.name || e.label, e.value, e.targetKey || 'legacy'].join('\u0000');
  const fieldKey = e => [e.sourceId || '', e.name || e.label, e.value, e.page || ''].join('\u0000');
  const legacyReviewKey = e => [e.label, e.n].join('\u0000');
  function loadReviews(ref) {
    if (state.reviewsFor === ref.name) return;
    state.reviewsFor = ref.name;
    state.reviews = new Map();
    state.reviewNotes = new Map();
    try {
      const saved = JSON.parse(localStorage.getItem('snc-reviews:' + ref.name) || '{}');
      const marks = saved.marks || saved; // keep pre-note review marks working
      Object.entries(marks).forEach(([k, v]) => state.reviews.set(k, v));
      Object.entries(saved.notes || {}).forEach(([k, v]) => state.reviewNotes.set(k, v));
    } catch (e) { /* storage unavailable */ }
  }
  function saveReviews() {
    try { localStorage.setItem('snc-reviews:' + state.reviewsFor, JSON.stringify({ marks: Object.fromEntries(state.reviews), notes: Object.fromEntries(state.reviewNotes) })); } catch (e) { /* storage unavailable */ }
  }
  // Previous releases stored a review mark per ROMC label/value, before a field
  // could have more than one required target. Carry it forward when there is
  // exactly one new target. Ambiguous old marks remain stored, rather than being
  // incorrectly applied to every new target.
  function migrateLegacyReviews(entries) {
    const byLegacyKey = new Map();
    entries.filter(e => e.sourceKind === 'romc').forEach(e => {
      const key = legacyReviewKey(e);
      if (!byLegacyKey.has(key)) byLegacyKey.set(key, []);
      byLegacyKey.get(key).push(e);
    });
    let changed = false, unresolved = 0;
    for (const [key, mark] of state.reviews) {
      const parts = key.split('\u0000');
      if (parts.length !== 2) continue;
      const choices = byLegacyKey.get(key) || [];
      if (choices.length === 1) {
        const targetKey = reviewKey(choices[0]);
        if (!state.reviews.has(targetKey)) { state.reviews.set(targetKey, mark); changed = true; }
      } else if (choices.length > 1) unresolved++;
    }
    if (changed) saveReviews();
    return unresolved;
  }

  // Link that opens a document page in the viewer, highlighting a value.
  const openLink = (i, page, hl, text) =>
    `<a href="#" class="open" data-doc="${i}" data-page="${page}" data-hl="${esc(hl || '')}">${text}</a>`;

  // =====================================================================
  // PDF extraction
  // =====================================================================

  // Join positioned text items left to right. pdf.js often splits a word into several
  // runs ("Se" + "rvice"), so only insert a space where there is a visible gap.
  function joinItems(items) {
    let out = '', prevEnd = null;
    for (const it of items) {
      const gap = prevEnd == null ? 0 : it.x - prevEnd;
      if (out && (gap > it.h * 0.18 || /\s$/.test(out) || /^\s/.test(it.str))) out += ' ';
      out += it.str;
      prevEnd = it.x + it.w;
    }
    return out.replace(/\s+/g, ' ').trim();
  }

  // Group positioned text items into visual lines (top to bottom, left to right).
  function groupLines(items) {
    const sorted = [...items].sort((a, b) => b.y - a.y || a.x - b.x);
    const lines = [];
    for (const it of sorted) {
      const ln = lines[lines.length - 1];
      if (ln && Math.abs(ln.y - it.y) < 3) ln.items.push(it);
      else lines.push({ y: it.y, items: [it] });
    }
    for (const ln of lines) {
      ln.items.sort((a, b) => a.x - b.x);
      ln.text = joinItems(ln.items);
    }
    return lines;
  }

  // The text a person would read as this field's label: words to its left on the
  // same row, or failing that, the nearest words directly above it.
  function rowLabel(items, rect) {
    const [x1, y1, x2, y2] = rect;
    const left = items.filter(it => it.x + it.w <= x1 + 3 && it.y < y2 && it.y + it.h > y1)
      .sort((a, b) => a.x - b.x);
    let s = joinItems(left);
    if (!s.trim()) {
      const above = items.filter(it => it.y >= y2 - 1 && it.y < y2 + 28 && it.x < x2 && it.x + it.w > x1)
        .sort((a, b) => a.y - b.y || a.x - b.x);
      if (above.length) s = joinItems(above.filter(i => Math.abs(i.y - above[0].y) < 3).sort((a, b) => a.x - b.x));
    }
    s = s.replace(/\s+/g, ' ').trim();
    // In tables like "Lift wire | Ferrule number", the row says which wire and the
    // column header says what the value is, so add the header.
    if (s && !/number|serial|vui|batch|type|manufacturer|year|date|version|weight|frequency|\bmk\b|name/i.test(s)) {
      // Header text is often split into fragments ("Ferrule", "n", "umber"), so rebuild
      // each line above the field (within its column) before reading it.
      const col = items.filter(it => it.y > y2 && it.y < y2 + 160 && it.x < x2 && it.x + it.w > x1 - 4);
      const head = groupLines(col).reverse().map(l => l.text).find(t => /number|serial|vui/i.test(t));
      if (head) s += ' – ' + head;
    }
    return s.length > 90 ? s.slice(0, 87) + '…' : s;
  }

  async function extract(file, displayName) {
    const doc = { id: newDocumentId(), name: displayName || file.name, blob: file, numPages: 0, pages: [], fieldList: [], blankFields: [], formV: 2, fields: {}, text: '', error: null };
    let pdf;
    try {
      const data = new Uint8Array(await file.arrayBuffer());
      pdf = await pdfjsLib.getDocument({ data, standardFontDataUrl: STANDARD_FONT_DATA_URL }).promise;
      doc.numPages = pdf.numPages;
      const headings = [];
      const widgets = [];
      const texts = [];

      for (let p = 1; p <= pdf.numPages; p++) {
        const page = await pdf.getPage(p);
        const tc = await page.getTextContent();
        const items = [];
        const parts = [];
        for (const it of tc.items) {
          parts.push(it.str, it.hasEOL ? '\n' : ' ');
          if (it.str && it.str.trim()) items.push({
            str: it.str, x: it.transform[4], y: it.transform[5], w: it.width || 0,
            h: it.height || Math.abs(it.transform[3]) || 8,
          });
        }
        let pageText = parts.join('');

        // Numbered section headings such as "4.6 Rescue/emergency descent equipment".
        // Table-of-contents lines (dot leaders) are ignored.
        for (const ln of groupLines(items)) {
          const near = [ln.items[0]];
          for (let k = 1; k < ln.items.length; k++) {
            const prev = near[near.length - 1];
            if (ln.items[k].x - (prev.x + prev.w) > 90) break;
            near.push(ln.items[k]);
          }
          const m = joinItems(near).match(/^(\d{1,2}(?:\.\d{1,2})*)\s+([A-Za-z][^\n]{1,90})$/);
          if (m && ln.items[0].x < 140 && !/\.{4,}|…{2,}/.test(ln.text) && !/^\d+\s+of\s+\d+/i.test(ln.text)) {
            headings.push({ page: p, y: ln.y, num: m[1], title: m[2].trim() });
          }
        }

        let annots = [];
        try { annots = await page.getAnnotations({ intent: 'display' }); } catch (e) { /* no annotations */ }
        const pageFieldVals = [];
        const notes = [];
        for (const a of annots) {
          if (a.subtype !== 'Widget') {
            // Text typed onto a scan (FreeText boxes, comments) often holds the serials.
            const c = (a.contentsObj && a.contentsObj.str) || a.contents || '';
            if (c && c.trim()) notes.push(c.replace(/\r/g, '\n').trim());
            continue;
          }
          if (!a.fieldName || a.radioButton || a.pushButton) continue;
          if (a.checkBox) {
            // Ticked boxes are answers ("Checked"); unticked ones are recorded as blanks.
            const on = a.fieldValue != null && a.fieldValue !== '' && a.fieldValue !== 'Off';
            widgets.push({ name: a.fieldName, value: on ? 'Checked' : '', page: p, rect: a.rect, row: rowLabel(items, a.rect), blank: !on, checkbox: true });
            continue;
          }
          let v = a.fieldValue;
          if (v == null || v === '' || v === 'Off') { widgets.push({ name: a.fieldName, value: '', page: p, rect: a.rect, row: rowLabel(items, a.rect), blank: true }); continue; }
          if (Array.isArray(v)) v = v.join(', ');
          v = String(v).trim();
          if (!v) { widgets.push({ name: a.fieldName, value: '', page: p, rect: a.rect, row: rowLabel(items, a.rect), blank: true }); continue; }
          pageFieldVals.push(v);
          widgets.push({ name: a.fieldName, value: v, page: p, rect: a.rect, row: rowLabel(items, a.rect) });
        }

        if (notes.length) pageText += '\n[Annotations]\n' + notes.join('\n') + '\n';
        texts.push(pageText);
        const chars = items.reduce((n, it) => n + it.str.trim().length, 0);
        doc.pages.push({ text: pageText, norm: alnum(pageText + ' ' + pageFieldVals.join(' ')), hasText: chars >= 40 || pageFieldVals.length > 0 });
        page.cleanup();
      }

      // Give every field a readable label: "<section> – <row label>".
      headings.sort((a, b) => a.page - b.page || b.y - a.y);
      for (const w of widgets) {
        let sec = null;
        for (const h of headings) {
          if (h.page < w.page || (h.page === w.page && h.y > w.rect[3] - 2)) sec = h; else break;
        }
        const secLabel = sec ? `${sec.num} ${sec.title}` : '';
        const label = [secLabel, w.row].filter(Boolean).join(' – ') || w.name;
          if (w.blank) { doc.blankFields.push({ name: w.name, page: w.page, label, section: sec ? sec.num : '', rect: w.rect, checkbox: !!w.checkbox }); continue; }
          doc.fieldList.push({ name: w.name, value: w.value, page: w.page, label, section: sec ? sec.num : '', rect: w.rect, checkbox: !!w.checkbox });
        (doc.fields[w.name] ||= new Set()).add(w.value);
      }
      doc.text = texts.join('\n');
    } catch (e) {
      doc.error = e.message || String(e);
    } finally {
      if (pdf) try { await pdf.destroy(); } catch (e) { /* ignore */ }
    }
    return doc;
  }

  // =====================================================================
  // Options & parsing
  // =====================================================================
  function opts() {
    return {
      useFields: checked('snc-usefields', true),
      useText: checked('snc-usetext', true),
      minTwo: checked('snc-mintwo', true),
      ignoreCase: checked('snc-ignorecase', true),
      ignoreSep: checked('snc-ignoresep', true),
      fieldFilter: textOf('snc-fieldfilter', 'serial|s\/?n\b|sn$|_sn|sn_').trim(),
    };
  }

  function parseAliases() {
    const map = new Map();
    for (const line of textOf('snc-aliases', '').split('\n')) {
      const i = line.indexOf('=');
      if (i < 0) continue;
      const canon = line.slice(0, i).trim();
      if (!canon) continue;
      map.set(normName(canon), canon);
      for (const a of line.slice(i + 1).split(',')) if (a.trim()) map.set(normName(a), canon);
    }
    return map;
  }

  // "Label | regex" lines (pattern mode) or "label regex | filename regex" lines (job book rules).
  function parsePipeLines(text, errEl, makeLeft) {
    const out = [], errs = [];
    text.split('\n').forEach((line, n) => {
      if (!line.trim() || line.trim().startsWith('#')) return;
      const i = line.indexOf(' | ') >= 0 ? line.indexOf(' | ') : line.indexOf('|');
      if (i < 0) { errs.push(`Line ${n + 1}: missing "|"`); return; }
      const left = line.slice(0, i).trim(), right = line.slice(i + (line[i] === ' ' ? 3 : 1)).trim();
      try { out.push(makeLeft(left, right)); } catch (e) { errs.push(`Line ${n + 1}: ${e.message}`); }
    });
    errEl.textContent = errs.join(' · ');
    return out;
  }
  const parseRules = () => parsePipeLines($('snc-rules').value, $('snc-ruleerr'),
    (label, src) => ({ label, re: new RegExp(src, 'gi') }));
  // A value worth cross-checking: has a digit, at least 5 letters/digits, and isn't
  // a date, a year, a weight or N/A.
  function serialLike(v) {
    const n = alnum(v);
    return n.length >= 5 && /\d/.test(n) && !/^N\/?A$/i.test(v.trim()) && !/\bkg\b/i.test(v) &&
      !/^\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}$/.test(v.trim()) && !/^(19|20)\d\d$/.test(v.trim());
  }

  // Serial-looking values next to a Serial / S/N / VUI / Ferrule / Batch label in page text.
  const LABELED_RE = /(?:serial\s*(?:no\.?|number|#)?|s\/n|\bsn\b|\bvui\b|ferrule(?:\s*(?:no\.?|number))?|batch\s*(?:no\.?|number)?)[\s:#.\-]*(?:\[[^\]\n]*\])?[^\n\d]{0,20}?((?=[A-Z0-9\-\/.]*\d)[A-Z0-9][A-Z0-9\-\/.]{4,})/gi;
  const LABEL_HINT = /serial|s\/n|\bsn\b|vui|ferrule|batch/i;

  function labeledSerials(doc) {
    const out = [];
    doc.pages.forEach((pg, i) => {
      for (const m of pg.text.matchAll(LABELED_RE)) {
        const v = m[1].replace(/[.\-\/]+$/, '');
        if (serialLike(v)) out.push({ value: v, page: i + 1, context: m[0].replace(/\s+/g, ' ').trim() });
      }
    });
    for (const f of doc.fieldList) {
      if (LABEL_HINT.test(f.label) && serialLike(f.value)) out.push({ value: f.value, page: f.page, context: f.label });
    }
    return out;
  }

  function levenshtein(a, b, max) {
    if (Math.abs(a.length - b.length) > max) return max + 1;
    let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      let rowMin = i;
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        if (cur[j] < rowMin) rowMin = cur[j];
      }
      if (rowMin > max) return max + 1;
      prev = cur;
    }
    return prev[b.length];
  }

  // Closest-looking token in a document, to make typos (one wrong digit) easy to spot.
  function nearMiss(doc, target) {
    const t = alnum(target);
    if (t.length < 5) return null;
    let best = null;
    doc.pages.forEach((pg, i) => {
      const tokens = (pg.text + ' ' + doc.fieldList.filter(f => f.page === i + 1).map(f => f.value).join(' '))
        .split(/[\s,;:()]+/);
      for (const tok of tokens) {
        const n = alnum(tok);
        if (n.length < t.length - 2 || n.length > t.length + 2 || !/\d/.test(n)) continue;
        const d = levenshtein(n, t, 2);
        const allowed = t.length >= 9 ? 2 : t.length >= 6 ? 1 : 0;
        if (d > 0 && d <= allowed && (!best || d < best.d)) best = { value: tok.trim(), page: i + 1, d };
      }
    });
    return best;
  }

  // Strip the shared "Pad_T-044_WTG_255190-" style prefix so columns stay readable.
  function setShortNames() {
    const names = state.docs.map(d => d.name.split('/').pop().replace(/\.pdf$/i, ''));
    let prefix = names.length > 1 ? names.reduce((p, n) => { let i = 0; while (i < p.length && p[i] === n[i]) i++; return p.slice(0, i); }) : '';
    // Don't cut a word in half: back up to the last separator if the prefix ends mid-word.
    if (names.some(n => /[A-Za-z0-9]/.test(n[prefix.length] || ''))) prefix = prefix.replace(/[^\-_ ]*$/, '');
    state.docs.forEach((d, i) => {
      d.short = names[i].slice(prefix.length)
        .replace(/^[\s\-_#]+/, '').replace(/^U2013[_\s-]*/, '')   // "#U2013" is an encoded en dash
        .replace(/_/g, ' ').trim() || names[i];
    });
  }

  function autoPickReference() {
    if (!state.refAuto && state.refIdx >= 0 && state.refIdx < state.docs.length) return;
    let idx = state.docs.findIndex(d => !d.error && /romc|recording.?of.?main.?components/i.test(d.name));
    if (idx < 0) {
      let best = 0;
      state.docs.forEach((d, i) => { if (!d.error && d.fieldList.length > best) { best = d.fieldList.length; idx = i; } });
    }
    state.refIdx = idx;
  }

  // =====================================================================
  // Mode A: full field review plan. Every populated, non-N/A ROMC and SIF
  // field is a review item. Document, IH10, VDD and project-tracker checks
  // are separate targets, so one successful check never hides another.
  // =====================================================================
  const isNA = value => /^(?:n\/?a|not applicable)$/i.test(String(value).trim());
  // Dates, initials, names, companies and Yes/No answers can each be skipped from
  // the review queue with the toggles on the Start Job Book Review card
  // (state.skip). Dates are identified by their form-field name/label as well as
  // common date-only values.
  const isDateField = (field, value) => {
    const label = `${field.name || ''} ${field.label || ''}`;
    const v = String(value || '').trim();
    return /\bdate\b|date[_\s-]?(?:of|signed|completed|issued|approved)|\bdated\b|timestamp/i.test(label) ||
      /^\d{1,2}[\/.\-]\d{1,2}[\/.\-]\d{2,4}$/.test(v) ||
      /^\d{4}[\/.\-]\d{1,2}[\/.\-]\d{1,2}$/.test(v) ||
      /^\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{2,4}$/i.test(v);
  };
  const isInitialsField = field => /initial/i.test(`${field.name || ''} ${field.label || ''}`);
  // Software / firmware / document version numbers, identified by their label.
  const isVersionField = field => /\bversion\b|\bver\.|firmware|software|\bsw\b|\brevision\b|\brev\b/i.test(`${field.name || ''} ${field.label || ''}`);
  const isCompanyField = field => /company|contractor|manufacturer|supplier|vendor|customer|client|employer|organi[sz]ation/i.test(`${field.name || ''} ${field.label || ''}`);
  // A person's name: signatory/technician style labels, not site, turbine or component names.
  const isPersonNameField = field => {
    const label = `${field.name || ''} ${field.label || ''}`;
    return /\b(?:name|technician|inspector|engineer|supervisor|witness|signed.?by|performed.?by|checked.?by|approved.?by|completed.?by|prepared.?by|reviewed.?by)\b/i.test(label) &&
      !/manufacturer|company|site|farm|project|turbine|component|part|model|tower|file|document|wtg|pad/i.test(label);
  };
  const isYesNoValue = value => /^(?:yes|no|y|n|true|false|yes\s*\/\s*no|checked|unchecked|ticked|unticked)$/i.test(String(value || '').trim());
  const shouldSkipField = (field, value) =>
    (state.skip.dates && isDateField(field, value)) ||
    (state.skip.initials && isInitialsField(field)) ||
    (state.skip.names && isPersonNameField(field)) ||
    (state.skip.company && isCompanyField(field)) ||
    (state.skip.version && isVersionField(field)) ||
    (state.skip.yesno && isYesNoValue(value));
  const docTarget = (key, label, re, photo) => ({ key, label, re, photo: !!photo });
  const manualTarget = (key, label) => ({ key, label, manual: true });
  // Forms whose own fields are reviewed (ROMC and SIF have document cross-checks; the others are reviewed on their own).
  const KIND_LABEL = { romc: 'ROMC', sif: 'SIF', genalign: 'Generator Alignment', qcfound: 'QC Foundation Earthing', qcturb: 'QC Earthing Between Turbines', document: 'Document' };
  const kindLabel = kind => KIND_LABEL[kind] || String(kind).toUpperCase();
  function sourceKind(doc) {
    if (/recording.?of.?main.?components|\bromc\b/i.test(doc.name)) return 'romc';
    if (/service.?inspection.?form|\bsif\b/i.test(doc.name)) return 'sif';
    if (/generator.?alignment/i.test(doc.name)) return 'genalign';
    if (/earthing.?between/i.test(doc.name)) return 'qcturb';
    if (/foundation.?earthing/i.test(doc.name)) return 'qcfound';
    return null;
  }
  // Every document type in the Review Guide is required by default. A site that
  // does not use one can tick "Ignore for this site" (state.ignoredDocTypes),
  // which greys the row out instead of removing it.
  const documentTypes = [
    { key: 'romc', label: 'Recording of Main Components (ROMC)', re: /recording.?of.?main.?components|\bromc\b/i, suggested: 'ROMC_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'sif', label: 'Service Inspection Form (SIF)', re: /service.?inspection.?form|\bsif\b/i, suggested: 'SIF_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'rescue', label: 'Rescue Kit Inspection', re: /rescue.?kit/i, suggested: 'Rescue_Kit_Inspection_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'hv', label: 'High Voltage Cable Test Report', re: /high.?voltage.?cable|\bhv.?cable/i, suggested: 'High_Voltage_Cable_Test_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'safety', label: 'Safety Cable Inspection', re: /safety.?cable|fall.?arrest|wire.?rope/i, suggested: 'Safety_Cable_Inspection_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'mechanical', label: 'Mechanical Completion Checklist', re: /mechanical.?completion/i, suggested: 'Mechanical_Completion_Checklist_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'bolts', label: 'Bolt Certificates', re: /bolt.?cert/i, suggested: 'Bolt_Certificates_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'baseLevel', label: 'Base Level Report', re: /base.?level/i, required: () => true },
    { key: 'qcFoundation', label: 'Quality Control Foundation Earthing', re: /foundation.?earthing/i, required: () => true },
    { key: 'qcTurbines', label: 'Quality Control of Earthing Between Turbines', re: /earthing.?between/i, required: () => true },
    { key: 'hardware', label: 'Hardware Lubrication Checklist', re: /hardware.?lubrication/i, required: () => true },
    { key: 'flange', label: 'Flange Reports', re: /flange/i, required: () => true },
    { key: 'genAlign', label: 'Generator Alignment Form', re: /generator.?alignment/i, required: () => true },
    { key: 'tensioning', label: 'Foundation Tensioning, Concrete and Grout Documents', re: /foundation.?tensioning|tensioning/i, required: () => true },
    { key: 'tenPercent', label: '10 Percent Checklist', re: /10.?percent/i, required: () => true },
    { key: 'serviceLift', label: 'Service Lift Installation Checklist', re: /service.?lift/i, suggested: 'Service_Lift_Installation_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
    { key: 'aviation', label: 'Aviation Light Manual', re: /aviation|faa|obstruction/i, suggested: 'Aviation_Light_Manual_WTG-XXXXXX_Pad-XX.pdf', required: () => true },
  ];
  const isDocTypeIgnored = type => state.ignoredDocTypes.includes(type.key);
  function requiredDocumentTypes() { return documentTypes.filter(type => type.required(state.docs) && !isDocTypeIgnored(type)); }
  function targetsFor(source, field) {
    const sec = field.section || '';
    if (source !== 'romc' && source !== 'sif') return [manualTarget('form-review', 'Form accuracy and completeness review')];
    if (source === 'sif') {
      const targets = [manualTarget('sif-completeness', 'SIF completeness, dates, signatures and N/A review')];
      if (field.page <= 4) targets.push(docTarget('romc-header', 'ROMC turbine / pad / WTG header', /recording.?of.?main.?components|\bromc\b/i));
      return targets;
    }
    if (!sec && field.page <= 2) return [docTarget('sif-header', 'SIF turbine / pad / WTG header', /service.?inspection.?form|\bsif\b/i)];
    if (/^4\.6\b/.test(sec)) return [
      manualTarget('ih10-rescue', 'IH10 component record'),
      docTarget('rescue-kit', 'Rescue Kit Inspection', /rescue.?kit/i, true),
    ];
    if (/^9\.2\b/.test(sec)) return [
      manualTarget('ih10-vdd-hv', 'IH10 and VDD component record'),
      docTarget('hv-cable', 'High Voltage Cable Test Report', /high.?voltage.?cable|\bhv.?cable/i, true),
    ];
    if (/^10\b/.test(sec)) return [
      manualTarget('ih10-vdd-lift', 'IH10 and VDD component record'),
      docTarget('service-lift', 'Service Lift Inspection', /service.?lift/i, true),
    ];
    if (/^11\b/.test(sec)) return [
      manualTarget('ih10-bolts', 'IH10 item / VUI record'),
      docTarget('bolt-certs', 'Bolt Certificates', /bolt.?cert/i, true),
    ];
    if (/^13\.[23]\b/.test(sec)) return [
      docTarget('aviation-lights', 'Aviation Light Manual', /aviation|faa|obstruction/i, true),
      manualTarget('tracker-aviation', 'Project tracker duplicate-serial check'),
    ];
    if (/^(?:14|15)\b/.test(sec)) return [manualTarget('tracker-special', 'Project tracker duplicate-serial check')];
    if (/^(?:2|3|4\.[1-5]|4\.[7-9]|5|6|7|8|9\.[13]|12)\b/.test(sec)) return [manualTarget('ih10-vdd', 'IH10 and VDD component record')];
    return [manualTarget('romc-review', 'ROMC field accuracy review')];
  }
  function groupReviewEntries(entries) {
    const byField = new Map();
    for (const entry of entries) {
      const key = fieldKey(entry);
      if (!byField.has(key)) byField.set(key, { key, entry, targets: [] });
      byField.get(key).targets.push(entry);
    }
    return [...byField.values()].sort((a, b) => reviewTableOrder(a.entry, b.entry));
  }
  // PDF coordinates start at the bottom-left. The largest Y coordinate is the
  // top of a field, so sorting it descending follows the natural reading order.
  function reviewLocationOrder(a, b) {
    const top = entry => Array.isArray(entry.rect) ? Math.max(entry.rect[1], entry.rect[3]) : -1;
    const left = entry => Array.isArray(entry.rect) ? Math.min(entry.rect[0], entry.rect[2]) : 0;
    return (a.page - b.page) || (top(b) - top(a)) || (left(a) - left(b)) || a.label.localeCompare(b.label);
  }
  function reviewTableOrder(a, b) {
    const sourceOrder = { romc: 0, sif: 1, genalign: 2, qcfound: 3, qcturb: 4 };
    return ((sourceOrder[a.sourceKind] ?? 9) - (sourceOrder[b.sourceKind] ?? 9)) || ((a.sourceIdx || 0) - (b.sourceIdx || 0)) || reviewLocationOrder(a, b) || a.targetLabel.localeCompare(b.targetLabel);
  }
  function noteDetails(note) {
    if (!note) return { text: '', document: null, images: [] };
    return typeof note === 'string' ? { text: note, document: null, images: [] } : { text: note.text || '', document: note.document || null, images: Array.isArray(note.images) ? note.images : [] };
  }
  function noteLabel(note) {
    const details = noteDetails(note);
    return details.document && details.document.name ? ` (${details.document.name})` : '';
  }
  function reviewTip(entry) {
    const target = entry.targetLabel;
    const common = `Compare the highlighted ${kindLabel(entry.sourceKind)} field with ${target}. Mark Verified only when the value is present and matches.`;
    switch (entry.targetKey) {
      case 'romc-header': return `${common} This is a turbine / pad / WTG header check.`;
      case 'sif-header': return `${common} This is a turbine / pad / WTG header check.`;
      case 'rescue-kit': return `${common} Use the Rescue Kit Inspection record, including the applicable serial or VUI.`;
      case 'hv-cable': return `${common} Use the High Voltage Cable Test Report and confirm the matching cable/component identifier.`;
      case 'service-lift': return `${common} Use the Service Lift Inspection record and confirm the matching lift identifier.`;
      case 'bolt-certs': return `${common} Use the Bolt Certificates and confirm the relevant batch or component identifier.`;
      case 'aviation-lights': return `${common} Use the Aviation Light Manual and confirm the applicable light identifier.`;
      case 'ih10-rescue': return `${common} Find the matching component entry in IH10.`;
      case 'ih10-vdd-hv': return `${common} Find the matching component entry in IH10 and VDD.`;
      case 'ih10-vdd-lift': return `${common} Find the matching component entry in IH10 and VDD.`;
      case 'ih10-bolts': return `${common} Confirm the IH10 item and VUI record.`;
      case 'ih10-vdd': return `${common} Find the matching component entry in IH10 and VDD.`;
      case 'tracker-aviation': return `${common} Check the project tracker for a duplicate serial.`;
      case 'tracker-special': return `${common} Check the project tracker for a duplicate serial.`;
      case 'form-review': return `Check that this ${kindLabel(entry.sourceKind)} entry is complete, legible and correct. Mark Issue if it needs follow-up.`;
      case 'page-review': return 'Read the whole page. Mark Verified when everything on it is complete, legible and correct; mark Issue if anything needs follow-up.';
      case 'blank-field': return 'This field was left blank. Confirm it is allowed to be empty (or should be N/A). Mark Issue if it must be completed.';
      case 'unticked-box': return 'This box is not ticked. Confirm that is correct. Mark Issue if it should be ticked.';
      case 'sif-completeness': return `Confirm this SIF field is complete and accurate. Mark Issue if its required supporting information is missing or needs follow-up.`;
      default: return common;
    }
  }
  function viewerDocuments() { return state.docs.map((doc, index) => ({ id: doc.id, name: doc.name, index })); }
  function compareJobBook() {
    const sources = state.docs.map((d, i) => ({ d, i, kind: ['romc', 'sif'].includes(sourceKind(d)) ? sourceKind(d) : 'document' })).filter(x => !x.d.error);
    if (!sources.length) { state.jb = null; return; }
    const entries = [];
    for (const { d, i, kind } of sources) {
      for (const f of (kind === 'document' ? [] : d.fieldList)) {
        const value = String(f.value || '').trim();
        if (!value || isNA(value) || shouldSkipField(f, value)) continue;
        // A comma belongs to this one form field; it never creates another review.
        for (const target of targetsFor(kind, f)) {
          const entry = { sourceId: d.id || d.name, sourceIdx: i, sourceKind: kind, name: f.name, label: f.label, value, n: alnum(value), page: f.page, rect: f.rect, section: f.section, targetKey: target.key, targetLabel: target.label, expected: [], found: [], scanPages: [] };
          if (target.manual) entry.status = 'manual';
          else {
            entry.expected = state.docs.map((candidate, index) => ({ candidate, index }))
              .filter(x => x.index !== i && !x.candidate.error && target.re.test(x.candidate.name)).map(x => x.index);
            if (!entry.expected.length) entry.status = 'notloaded';
            else if (!entry.n) entry.status = 'visual';
            else {
              for (const index of entry.expected) {
                const pages = state.docs[index].pages.map((pg, pageIndex) => pg.norm.includes(entry.n) ? pageIndex + 1 : 0).filter(Boolean);
                if (pages.length) entry.found.push({ i: index, pages });
                else {
                  const scanned = state.docs[index].pages.map((pg, pageIndex) => pg.hasText ? 0 : pageIndex + 1).filter(Boolean);
                  if (scanned.length) entry.scanPages.push({ i: index, pages: scanned });
                }
              }
              entry.status = entry.found.length ? 'confirmed' : (entry.scanPages.length || target.photo) ? 'visual' : 'notfound';
            }
          }
          entries.push(entry);
        }
      }
      // Blank fields and unticked boxes are flagged for review unless the Skip toggle is on.
      if (kind !== 'document' && !state.skip.blanks) for (const f of d.blankFields || []) {
        if (shouldSkipField(f, '')) continue;
        const target = f.checkbox ? manualTarget('unticked-box', 'Unticked box: confirm it should be unticked') : manualTarget('blank-field', 'Blank field: confirm it may be empty');
        entries.push({ sourceId: d.id || d.name, sourceIdx: i, sourceKind: kind, name: f.name, label: f.label, value: f.checkbox ? '(unticked)' : '(blank)', n: '', page: f.page, rect: f.rect, section: f.section, targetKey: target.key, targetLabel: target.label, expected: [], found: [], scanPages: [], status: 'manual' });
      }
      // Everything in every document is reviewed: any page with no reviewed field gets a whole-page review.
      const covered = new Set(entries.filter(e => e.sourceIdx === i).map(e => e.page));
      for (let p = 1; p <= (d.numPages || 0); p++) {
        if (covered.has(p)) continue;
        entries.push({ sourceId: d.id || d.name, sourceIdx: i, sourceKind: kind, name: `page-${p}`, label: `${d.short || d.name} – page ${p} of ${d.numPages}`, value: `Page ${p}`, n: '', page: p, rect: null, section: '', targetKey: 'page-review', targetLabel: 'Whole-page review: confirm everything on the page is complete and correct', expected: [], found: [], scanPages: [], status: 'manual', pageReview: true });
      }
    }
    entries.sort(reviewTableOrder);
    state.jb = { entries, groups: groupReviewEntries(entries), extra: [], refIdx: state.refIdx };
  }

  const JB = {
    manual: { t: 'MANUAL REVIEW', c: 'warn', o: 0 },
    mismatch: { t: 'MISMATCH', c: 'bad', o: 0 },
    duplicate: { t: 'DUPLICATE', c: 'bad', o: 1 },
    notfound: { t: 'NOT FOUND', c: 'bad', o: 2 },
    visual: { t: 'CHECK VISUALLY', c: 'warn', o: 2.5 },
    notloaded: { t: 'DOC MISSING', c: 'warn', o: 3 },
    confirmed: { t: 'CONFIRMED', c: 'ok', o: 4 },
    alsofound: { t: 'ALSO IN DOCS', c: 'ok', o: 5 },
    refonly: { t: 'ROMC ONLY', c: 'neutral', o: 6 },
  };

  function jbDetail(e) {
    const docName = i => esc(state.docs[i].short);
    switch (e.status) {
      case 'manual': return `Review against <strong>${esc(e.targetLabel)}</strong>`;
      case 'duplicate': return `Same value also entered for: ${e.dupWith.map(esc).join('; ')}`;
      case 'mismatch': return `${openLink(e.near.doc, e.near.page, e.near.value, `${docName(e.near.doc)} p.${e.near.page}`)} has <span class="chip bad">${esc(e.near.value)}</span> — ${e.near.d} character${e.near.d > 1 ? 's' : ''} different`;
      case 'notfound': return `Not in ${e.expected.map(docName).join(', ')}`;
      case 'visual': {
        const where = e.scanPages.map(s => openLink(s.i, s.pages[0], e.value, `${docName(s.i)} p.${rangeList(s.pages)}`));
        return `Not in the text of ${e.expected.map(docName).join(', ')}. ` +
          (where.length ? `Check the scanned page(s): ${where.join('; ')}` : 'Check the photo of the tag');
      }
      case 'notloaded': return `Expected document is not loaded: <strong>${esc(e.targetLabel)}</strong>`;
      default: return '';
    }
  }

  // [1,2,3,7,8] -> "1–3, 7–8"
  function rangeList(nums) {
    const out = [];
    for (let k = 0; k < nums.length; k++) {
      let j = k;
      while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
      out.push(j > k ? `${nums[k]}–${nums[j]}` : `${nums[k]}`);
      k = j;
    }
    return out.join(', ');
  }

  // keepRows: re-draw the same rows in the same order (used while stepping through in the viewer).
  function renderJobBook(keepRows) {
    const launcher = $('snc-review-launcher'), sum = $('snc-jb-summary'), tbl = $('snc-jb-table'), ex = $('snc-jb-extra');
    const jb = state.jb;
    if (!jb) {
      launcher.innerHTML = state.docs.length ? '<div class="review-launcher-empty">Choose or load a ROMC or SIF document to begin a Job Book Review.</div>' : '';
      sum.innerHTML = state.docs.length ? '<p class="none">Pick a reference document (the ROMC) in the list above.</p>' : '';
      tbl.innerHTML = ''; ex.innerHTML = '';
      return;
    }
    const ref = state.docs[jb.refIdx] || state.docs.find(d => sourceKind(d) === 'romc') || state.docs[0];
    const issuesOnly = checked('snc-issuesonly', false);
    const counts = {};
    jb.entries.forEach(e => { counts[e.status] = (counts[e.status] || 0) + 1; });
    sum.innerHTML = `<p>Review plan: <strong>${jb.entries.length}</strong> populated ROMC/SIF field targets</p><p>` +
      Object.keys(JB).filter(k => counts[k]).map(k => `<span class="st-${JB[k].c}">${counts[k]} ${JB[k].t.toLowerCase()}</span>`).join('') + '</p>';

    loadReviews(ref);
    const unresolvedLegacy = migrateLegacyReviews(jb.entries);
    const reviewed = Object.values(Object.fromEntries(state.reviews));
    const nVer = reviewed.filter(v => v === 'verified').length, nIss = reviewed.filter(v => v === 'issue').length;
    const progress = scope => {
      const groups = reviewGroups(scope);
      const total = groups.reduce((n, group) => n + group.targets.length, 0);
      const complete = groups.reduce((n, group) => n + group.targets.filter(e => state.reviews.get(reviewKey(e)) === 'verified').length, 0);
      return { total, complete };
    };
    const allProgress = progress('all'), romcProgress = progress('romc'), sifProgress = progress('sif'), otherProgress = progress('other');
    launcher.innerHTML = `<section class="review-launcher" aria-label="Start Job Book Review"><div class="review-launcher-copy"><strong>Start Job Book Review</strong><span>Review every required field one at a time, with its source and matching job-book record side by side.</span></div><label class="review-scope-label">Review<select id="snc-review-scope" aria-label="Review scope"><option value="all"${state.reviewScope === 'all' ? ' selected' : ''}>All job book fields</option><option value="romc"${state.reviewScope === 'romc' ? ' selected' : ''}>ROMC fields only</option><option value="sif"${state.reviewScope === 'sif' ? ' selected' : ''}>SIF fields only</option><option value="other"${state.reviewScope === 'other' ? ' selected' : ''}>Other documents (page review)</option></select></label><button type="button" class="btn btn-review-start" data-start-review${allProgress.total ? '' : ' disabled'}>Start / resume review</button><div class="review-skip" role="group" aria-label="Fields to skip"><span>Skip:</span>${[['dates', 'Dates'], ['initials', 'Initials'], ['names', 'Names'], ['company', 'Company'], ['version', 'Version numbers'], ['yesno', 'Yes / No answers'], ['blanks', 'Blank / unticked fields']].map(([k, t]) => `<label><input type="checkbox" data-skip="${k}"${state.skip[k] ? ' checked' : ''}> ${t}</label>`).join('')}</div><div class="review-launcher-progress">All <strong>${allProgress.complete}/${allProgress.total}</strong> &middot; ROMC <strong>${romcProgress.complete}/${romcProgress.total}</strong> &middot; SIF <strong>${sifProgress.complete}/${sifProgress.total}</strong>${otherProgress.total ? ` &middot; Other documents <strong>${otherProgress.complete}/${otherProgress.total}</strong>` : ''}</div></section>`;
    if (nVer || nIss) sum.innerHTML += `<p class="src">Your review: ${nVer} verified · ${nIss} marked as issue</p>`;
    if (unresolvedLegacy) sum.innerHTML += `<p class="src">${unresolvedLegacy} earlier review mark${unresolvedLegacy === 1 ? '' : 's'} kept for reference because this field now has multiple separate targets.</p>`;

    const rows = keepRows ? state.jbRows : jb.entries
      .filter(e => !issuesOnly || (['manual', 'mismatch', 'duplicate', 'notfound', 'visual', 'notloaded'].includes(e.status) && state.reviews.get(reviewKey(e)) !== 'verified') || state.reviews.get(reviewKey(e)) === 'issue')
      .sort(reviewTableOrder);
    state.jbRows = rows;
    let h = '<thead><tr><th>Status</th><th>Source field</th><th>Value</th><th>Review target</th><th>Found in</th><th>Notes</th><th></th></tr></thead><tbody>';
    if (!rows.length) h += `<tr><td colspan="7" class="none">${issuesOnly ? 'No open reviews. Untick “Issues only” to see every field.' : 'No populated ROMC or SIF fields found.'}</td></tr>`;
    rows.forEach((e, k) => {
      const found = e.found.map(f => `<div class="${e.expected.includes(f.i) ? 'exp' : ''}">${esc(state.docs[f.i].short)} <span class="src">${f.pages.slice(0, 4).map(pn => openLink(f.i, pn, e.value, 'p.' + pn)).join(', ')}${f.pages.length > 4 ? '…' : ''}</span></div>`).join('') || '<span class="none">—</span>';
      const mark = state.reviews.get(reviewKey(e));
      const note = state.reviewNotes.get(reviewKey(e));
      const issue = noteDetails(note);
      const badge = mark ? `<div><span class="mark mark-${mark}">${mark === 'verified' ? '✓ Verified' : '⚠ Issue'}</span>${issue.text ? `<div class="src">${esc(issue.text)}${esc(noteLabel(note))}</div>` : issue.document ? `<div class="src">Attached to ${esc(issue.document.name)}</div>` : ''}</div>` : '';
      h += `<tr${mark ? ` class="row-${mark}"` : ''}><td><span class="st st-${JB[e.status].c}">${JB[e.status].t}</span>${badge}</td>` +
        `<td>${esc(kindLabel(e.sourceKind))}: ${esc(e.label)}<div class="src">${openLink(e.sourceIdx, e.page, e.value, `${kindLabel(e.sourceKind)} p.` + e.page)}</div></td>` +
        `<td><span class="chip ${JB[e.status].c}">${esc(e.value)}</span></td><td>${esc(e.targetLabel)}</td><td>${found}</td><td class="notes">${jbDetail(e)}</td>` +
        `<td><button type="button" class="btn btn-small" data-review="${k}">Review</button></td></tr>`;
    });
    tbl.innerHTML = h + '</tbody>';

    if (jb.extra.length) {
      let x = `<details><summary><strong>${jb.extra.length} serial-like value(s) in other documents that aren't in the ROMC</strong> <span class="src">— review for typos or missing ROMC entries (tool and equipment serials will also appear here)</span></summary>` +
        '<div class="tablewrap"><table><thead><tr><th>Document</th><th>Value</th><th>Context</th></tr></thead><tbody>';
      for (const s of jb.extra.sort((a, b) => a.doc - b.doc || a.page - b.page)) {
        x += `<tr><td>${esc(state.docs[s.doc].short)} <span class="src">${openLink(s.doc, s.page, s.value, 'p.' + s.page)}</span></td><td><span class="chip warn">${esc(s.value)}</span></td><td class="src">${esc(s.context.slice(0, 90))}</td></tr>`;
      }
      ex.innerHTML = x + '</tbody></table></div></details>';
    } else ex.innerHTML = '';
  }

  // =====================================================================
  // Mode B: compare fields / text patterns across documents
  // =====================================================================
  function compareFields() {
    const o = opts();
    const norm = v => {
      let s = String(v).trim().replace(/\s+/g, ' ');
      if (o.ignoreCase) s = s.toUpperCase();
      if (o.ignoreSep) s = s.replace(/[\s\-_\/.]/g, '');
      return s;
    };
    const docs = state.docs.filter(d => !d.error);
    const rows = new Map();
    const row = (label, src) => {
      const key = normName(label);
      if (!rows.has(key)) rows.set(key, { label, srcs: new Set(), cells: docs.map(() => []) });
      const r = rows.get(key);
      r.srcs.add(src);
      return r;
    };
    const add = (r, di, v) => { if (v && !r.cells[di].includes(v)) r.cells[di].push(v); };

    if (o.useFields) {
      let filt = null;
      try { filt = o.fieldFilter ? new RegExp(o.fieldFilter, 'i') : null; } catch (e) { filt = null; }
      const aliases = parseAliases();
      docs.forEach((d, di) => {
        for (const [name, vals] of Object.entries(d.fields)) {
          const canon = aliases.get(normName(name));
          if (!canon && filt && !filt.test(name)) continue;
          const r = row(canon || name, 'Form field');
          for (const v of vals) add(r, di, v);
        }
      });
    }
    if (o.useText) {
      for (const rule of parseRules()) {
        const r = row(rule.label, 'Text pattern');
        docs.forEach((d, di) => {
          for (const m of d.text.matchAll(rule.re)) add(r, di, (m[1] ?? m[0]).trim());
        });
      }
    }

    const sig = set => [...set].sort().join('\u0000');
    const out = [];
    for (const r of rows.values()) {
      r.normSets = r.cells.map(c => new Set(c.map(norm)));
      const have = r.normSets.filter(s => s.size);
      r.src = [...r.srcs].sort().join(' + ');
      if (o.minTwo && !r.srcs.has('Text pattern') && have.length < 2) continue;
      if (have.length === 0) r.status = 'missing';
      else if (have.length === 1) r.status = docs.length > 1 ? 'single' : 'match';
      else if (!have.every(s => sig(s) === sig(have[0]))) r.status = 'mismatch';
      else r.status = have.length < docs.length ? 'missing' : 'match';
      r.norm = norm;
      out.push(r);
    }
    const order = { mismatch: 0, missing: 1, single: 2, match: 3 };
    out.sort((a, b) => order[a.status] - order[b.status] || a.label.localeCompare(b.label));
    state.rowsB = out;
    state.cmpDocs = docs;
  }

  const LABEL_B = { match: 'MATCH', mismatch: 'MISMATCH', missing: 'MISSING', single: 'ONLY 1 DOC' };
  const CLASS_B = { match: 'ok', mismatch: 'bad', missing: 'warn', single: 'warn' };

  function renderFields() {
    const docs = state.cmpDocs || [];
    const t = $('snc-table');
    if (!docs.length) { t.innerHTML = ''; $('snc-summary').innerHTML = ''; return; }
    const counts = { mismatch: 0, missing: 0, single: 0, match: 0 };
    state.rowsB.forEach(r => counts[r.status]++);
    $('snc-summary').innerHTML =
      `<p><span class="st-bad">${counts.mismatch} mismatch</span>` +
      `<span class="st-warn">${counts.missing + counts.single} missing / one doc only</span>` +
      `<span class="st-ok">${counts.match} match</span></p>`;
    let h = '<thead><tr><th>Status</th><th>Item</th>' + docs.map(d => `<th>${esc(d.short)}</th>`).join('') + '</tr></thead><tbody>';
    if (!state.rowsB.length) h += `<tr><td colspan="${docs.length + 2}" class="none">No serial numbers found. Check the field filter or text patterns.</td></tr>`;
    for (const r of state.rowsB) {
      h += `<tr><td><span class="st st-${CLASS_B[r.status]}">${LABEL_B[r.status]}</span></td><td>${esc(r.label)}<div class="src">${r.src}</div></td>`;
      const have = r.normSets.filter(s => s.size);
      r.cells.forEach(vals => {
        if (!vals.length) { h += '<td class="none">—</td>'; return; }
        h += '<td>' + vals.map(v => {
          const count = have.filter(s => s.has(r.norm(v))).length;
          const cls = count === have.length ? 'ok' : count > have.length / 2 ? 'warn' : 'bad';
          return `<span class="chip ${cls}" title="Found in ${count} of ${have.length} documents">${esc(v)}</span>`;
        }).join('') + '</td>';
      });
      h += '</tr>';
    }
    t.innerHTML = h + '</tbody>';
  }

  // =====================================================================
  // Rendering, loading, events
  // =====================================================================
  function run() {
    setShortNames();
    autoPickReference();
    compareJobBook();
    renderJobBook();
    $('snc-issues-csv').disabled = !state.jb || !state.jb.entries.some(e => state.reviews.get(reviewKey(e)) === 'issue');
    $('snc-issues-pdf').disabled = $('snc-issues-csv').disabled;
    $('snc-snapshot').disabled = !state.docs.some(d => !d.error);
    renderCompare();
  }

  function renderDocs() {
    setShortNames();
    autoPickReference();
    const loaded = state.docs.map((d, i) => {
      if (d.error) return `<li><strong>${esc(d.name)}</strong> <span class="err">Could not read: ${esc(d.error)}</span> <button class="btn btn-small" data-rm="${i}">Remove</button></li>`;
      const scanned = d.pages.filter(p => !p.hasText).length;
      const textInfo = scanned === 0 ? 'text' : scanned === d.numPages ? '<span class="warn-text">scanned – no text</span>' : `<span class="warn-text">${scanned} of ${d.numPages} pages scanned</span>`;
      return `<li class="doc-row"><label class="ref" title="Use as reference (master) document"><input type="radio" name="snc-ref" value="${i}" ${i === state.refIdx ? 'checked' : ''}> Reference</label>
        <span class="docname" title="${esc(d.name)}">${openLink(i, 1, '', esc(d.short))}</span>
        <span class="meta">${d.numPages} p · ${d.fieldList.length} fields · ${textInfo}</span>
        <details data-i="${i}"><summary class="meta">Text &amp; fields</summary></details>
        <button class="btn btn-small" data-rm="${i}">Remove</button></li>`;
    });
    const missing = documentTypes.filter(type => type.required(state.docs) && !state.docs.some(doc => !doc.error && type.re.test(doc.name))).map(type => {
      const ignored = isDocTypeIgnored(type);
      const toggle = `<label class="doc-ignore"><input type="checkbox" data-ignore-type="${esc(type.key)}"${ignored ? ' checked' : ''}> Ignore for this site</label>`;
      return ignored
        ? `<li class="doc-row doc-ignored"><strong>Ignored: ${esc(type.label)}</strong><span>Not used at this site, so it is not required.</span>${toggle}</li>`
        : `<li class="doc-row doc-missing"><strong>Missing: ${esc(type.label)}</strong><span>Add this required job-book document.</span>${toggle}</li>`;
    });
    $('snc-docs').innerHTML = loaded.concat(missing).join('');
  }

  function showProgress(msg) { $('snc-progress').textContent = msg || ''; }

  async function collectFiles(list) {
    const out = [];
    for (const f of list) {
      if (/\.zip$/i.test(f.name)) {
        if (typeof JSZip === 'undefined') { showProgress('ZIP support failed to load; unzip the job book and drop the folder instead.'); continue; }
        showProgress(`Opening ${f.name}…`);
        const zip = await JSZip.loadAsync(f);
        for (const entry of Object.values(zip.files)) {
          if (entry.dir || !/\.pdf$/i.test(entry.name) || /(^|\/)__MACOSX\//.test(entry.name)) continue;
          out.push({ name: entry.name.split('/').pop(), get: () => entry.async('blob') });
        }
      } else if (/\.pdf$/i.test(f.name) || f.type === 'application/pdf') {
        out.push({ name: f.name, get: async () => f });
      }
    }
    return out;
  }

  // Walk dropped folders (drag & drop of a directory).
  async function filesFromDrop(dt) {
    const items = [...(dt.items || [])].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
    if (!items.length) return [...dt.files];
    const files = [];
    const walk = async entry => {
      if (entry.isFile) files.push(await new Promise((res, rej) => entry.file(res, rej)));
      else if (entry.isDirectory) {
        const reader = entry.createReader();
        let batch;
        do {
          batch = await new Promise((res, rej) => reader.readEntries(res, rej));
          for (const e of batch) await walk(e);
        } while (batch.length);
      }
    };
    for (const e of items) await walk(e);
    return files;
  }

  async function addFiles(list) {
    const files = await collectFiles([...list]);
    if (!files.length) { showProgress('No PDFs found.'); return; }
    // One document at a time keeps memory reasonable with large scanned PDFs.
    for (let k = 0; k < files.length; k++) {
      showProgress(`Reading ${k + 1} of ${files.length}: ${files[k].name}`);
      const blob = await files[k].get();
      const doc = await extract(blob, files[k].name);
      state.docs.push(doc);
      try { await saveDocument(doc); } catch (e) { console.warn('Could not save PDF for reload:', e); }
    }
    try { await saveSession(); } catch (e) { console.warn('Could not save PDF session:', e); }
    showProgress('');
    renderDocs();
    run();
  }

  const drop = $('snc-drop');
  drop.addEventListener('click', e => { if (!e.target.closest('button')) $('snc-file').click(); });
  $('snc-file').addEventListener('change', e => { addFiles(e.target.files); e.target.value = ''; });
  $('snc-folder').addEventListener('change', e => { addFiles(e.target.files); e.target.value = ''; });
  $('snc-pick-folder').addEventListener('click', e => { e.stopPropagation(); $('snc-folder').click(); });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('dragover'); }));
  drop.addEventListener('drop', async e => addFiles(await filesFromDrop(e.dataTransfer)));

  $('snc-docs').addEventListener('click', e => {
    const i = e.target.dataset && e.target.dataset.rm;
    if (i === undefined) return;
    const removed = state.docs[+i];
    state.docs.splice(+i, 1);
    if (+i === state.refIdx) { state.refAuto = true; state.refIdx = -1; } else if (+i < state.refIdx) state.refIdx--;
    renderDocs(); run();
    removeStoredDocument(removed.id).then(saveSession).catch(e => console.warn('Could not update saved PDFs:', e));
  });
  $('snc-docs').addEventListener('change', e => {
    const key = e.target.dataset && e.target.dataset.ignoreType;
    if (key) {
      const set = new Set(state.ignoredDocTypes);
      if (e.target.checked) set.add(key); else set.delete(key);
      state.ignoredDocTypes = [...set];
      renderDocs();
      saveSession().catch(err => console.warn('Could not save ignored document types:', err));
      return;
    }
    if (e.target.name !== 'snc-ref') return;
    state.refIdx = +e.target.value; state.refAuto = false;
    run();
    saveSession().catch(err => console.warn('Could not save reference selection:', err));
  });
  // Build the (possibly large) text view only when opened.
  $('snc-docs').addEventListener('toggle', e => {
    const det = e.target;
    if (!det.open || det.querySelector('textarea')) return;
    const d = state.docs[+det.dataset.i];
    const ta = document.createElement('textarea');
    ta.readOnly = true;
    ta.value = (d.fieldList.length ? '--- FORM FIELDS (label: value) ---\n' + d.fieldList.map(f => `p.${f.page}  ${f.label}  [${f.name}]: ${f.value}`).join('\n') + '\n\n' : '') +
      d.pages.map((p, i) => `--- PAGE ${i + 1}${p.hasText ? '' : ' (scanned / no text)'} ---\n${p.text}`).join('\n');
    det.appendChild(ta);
  }, true);

  $('snc-clear').addEventListener('click', () => {
    viewer.reset(); state.docs = []; state.refIdx = -1; state.refAuto = true; state.ignoredDocTypes = []; renderDocs(); run();
    clearStoredDocuments().catch(e => console.warn('Could not clear saved PDFs:', e));
  });

  // ---------- Viewer ----------
  // Where the evidence for a ROMC row should be: the required document's matching page,
  // a near-miss, its first scanned page, or anywhere else the value was found.
  function reviewTarget(e) {
    // sure: false = the page is a guess, so the viewer keeps the reviewer's place
    // if that document is already open.
    const fe = e.found.find(f => e.expected.includes(f.i));
    if (fe) return { doc: fe.i, page: fe.pages[0], hl: e.value, sure: true };
    if (e.near) return { doc: e.near.doc, page: e.near.page, hl: e.near.value, sure: true };
    if (e.scanPages && e.scanPages.length) return { doc: e.scanPages[0].i, page: e.scanPages[0].pages[0], hl: e.value, sure: false };
    if (e.expected.length) return { doc: e.expected[0], page: 1, hl: e.value, sure: false };
    if (e.found.length) return { doc: e.found[0].i, page: e.found[0].pages[0], hl: e.value, sure: true };
    return null;
  }
  function targetIndexFor(group, key) {
    const ix = group.targets.findIndex(e => reviewKey(e) === key);
    return ix < 0 ? 0 : ix;
  }
  function firstOpenTarget(group) {
    const ix = group.targets.findIndex(e => state.reviews.get(reviewKey(e)) !== 'verified');
    return ix < 0 ? 0 : ix;
  }
  function reviewGroups(scope = state.reviewScope) {
    const groups = state.jb && state.jb.groups || [];
    if (scope === 'all') return groups;
    if (scope === 'other') return groups.filter(group => group.entry.sourceKind !== 'romc' && group.entry.sourceKind !== 'sif');
    return groups.filter(group => group.entry.sourceKind === scope);
  }
  function rememberReviewPosition(entry, scope) {
    state.reviewPosition = { scope, field: fieldKey(entry), target: reviewKey(entry) };
    saveSession().catch(e => console.warn('Could not save review position:', e));
  }
  function resumeReview(scope = state.reviewScope) {
    const groups = reviewGroups(scope);
    if (!groups.length) return;
    state.reviewScope = scope;
    const hasPosition = state.reviewPosition && (!state.reviewPosition.scope || state.reviewPosition.scope === scope);
    let groupIndex = hasPosition ? groups.findIndex(g => g.key === state.reviewPosition.field) : -1;
    if (groupIndex < 0) groupIndex = groups.findIndex(g => g.targets.some(e => state.reviews.get(reviewKey(e)) !== 'verified'));
    if (groupIndex < 0) groupIndex = 0;
    const group = groups[groupIndex];
    const targetIndex = hasPosition ? targetIndexFor(group, state.reviewPosition.target) : firstOpenTarget(group);
    openReviewGroup(groupIndex, targetIndex, scope);
  }
  function nextUnresolvedReview(groupIndex, targetIndex, scope = state.reviewScope) {
    const groups = reviewGroups(scope);
    if (!groups.length) return;
    const current = groups[groupIndex];
    for (let ti = targetIndex + 1; ti < current.targets.length; ti++) {
      if (state.reviews.get(reviewKey(current.targets[ti])) !== 'verified') return openReviewGroup(groupIndex, ti, scope);
    }
    for (let offset = 1; offset <= groups.length; offset++) {
      const gi = (groupIndex + offset) % groups.length;
      if (gi === groupIndex) break;
      const group = groups[gi];
      const ti = group.targets.findIndex(e => state.reviews.get(reviewKey(e)) !== 'verified');
      if (ti >= 0) return openReviewGroup(gi, ti, scope);
    }
    for (let ti = 0; ti < targetIndex; ti++) {
      if (state.reviews.get(reviewKey(current.targets[ti])) !== 'verified') return openReviewGroup(groupIndex, ti, scope);
    }
    // Everything is verified; keep the current field open rather than closing the review.
    openReviewGroup(groupIndex, targetIndex, scope);
  }
  // Per-document progress for the viewer's Documents panel (queue order).
  function documentNav(scope, activeIdx) {
    const groups = reviewGroups(scope), byDoc = new Map();
    groups.forEach((group, gi) => {
      const idx = group.entry.sourceIdx;
      if (!byDoc.has(idx)) byDoc.set(idx, { idx, total: 0, done: 0, first: gi, open: -1 });
      const rec = byDoc.get(idx);
      group.targets.forEach(t => { rec.total++; if (state.reviews.get(reviewKey(t)) === 'verified') rec.done++; });
      if (rec.open < 0 && group.targets.some(t => state.reviews.get(reviewKey(t)) !== 'verified')) rec.open = gi;
    });
    return [...byDoc.values()].map(rec => {
      const d = state.docs[rec.idx], gi = rec.open >= 0 ? rec.open : rec.first;
      return { label: (d && (d.short || d.name)) || 'Document', name: d ? d.name : '', total: rec.total, done: rec.done, active: rec.idx === activeIdx,
        onSelect: () => openReviewGroup(gi, firstOpenTarget(groups[gi]), scope) };
    });
  }
  function openReviewGroup(groupIndex, targetIndex, scope = state.reviewScope) {
    const groups = reviewGroups(scope);
    const group = groups[groupIndex];
    if (!group) return;
    state.reviewScope = scope;
    targetIndex = Math.max(0, Math.min(targetIndex || 0, group.targets.length - 1));
    const entry = group.targets[targetIndex];
    rememberReviewPosition(entry, scope);
    viewer.open({
      title: `<strong>${esc(kindLabel(group.entry.sourceKind))}: ${esc(group.entry.label)}</strong> <span class="chip ${JB[entry.status].c}">${esc(group.entry.value)}</span>`,
      left: { doc: group.entry.sourceIdx, page: group.entry.page, hl: group.entry.pageReview ? [] : [group.entry.label, group.entry.value], rect: group.entry.rect, sure: true },
      targets: group.targets.map(target => ({
        label: target.targetLabel, right: reviewTarget(target),
        status: state.reviews.get(reviewKey(target)) || null,
      })),
      targetIndex,
      documents: viewerDocuments(),
      tip: reviewTip(entry),
      mark: { key: reviewKey(entry), status: state.reviews.get(reviewKey(entry)) || null, note: noteDetails(state.reviewNotes.get(reviewKey(entry))).text, noteDocument: noteDetails(state.reviewNotes.get(reviewKey(entry))).document, images: noteDetails(state.reviewNotes.get(reviewKey(entry))).images },
      step: { index: groupIndex, total: groups.length },
      docNav: () => documentNav(scope, group.entry.sourceIdx),
      onTarget: index => openReviewGroup(groupIndex, index, scope),
      onNextUnresolved: () => nextUnresolvedReview(groupIndex, targetIndex, scope),
    });
  }
  function openReview(k) {
    const e = state.jbRows[k];
    if (!e) return;
    const groups = reviewGroups('all');
    const groupIndex = groups.findIndex(group => group.key === fieldKey(e));
    if (groupIndex >= 0) return openReviewGroup(groupIndex, targetIndexFor(groups[groupIndex], reviewKey(e)), 'all');
    viewer.open({
      title: `<span class="st st-${JB[e.status].c}">${JB[e.status].t}</span> <strong>${esc(kindLabel(e.sourceKind))}: ${esc(e.label)}</strong> <span class="chip ${JB[e.status].c}">${esc(e.value)}</span> <span class="src">→ ${esc(e.targetLabel)}</span>`,
      left: { doc: e.sourceIdx, page: e.page, hl: [e.label, e.value], rect: e.rect, sure: true }, right: reviewTarget(e),
      documents: viewerDocuments(),
      tip: reviewTip(e),
      mark: { key: reviewKey(e), status: state.reviews.get(reviewKey(e)) || null, note: noteDetails(state.reviewNotes.get(reviewKey(e))).text, noteDocument: noteDetails(state.reviewNotes.get(reviewKey(e))).document, images: noteDetails(state.reviewNotes.get(reviewKey(e))).images },
    });
  }
  const viewer = window.createSncViewer({
    getDocs: () => state.docs,
    onMark: (key, status, note, noteDocument, preserveNote, images) => {
      if (status) state.reviews.set(key, status); else state.reviews.delete(key);
      if (status === 'issue') {
        if (note || noteDocument || (images && images.length)) state.reviewNotes.set(key, { text: note || '', document: noteDocument || null, images: images || [] }); else state.reviewNotes.delete(key);
      } else if (!preserveNote) state.reviewNotes.delete(key);
      saveReviews();
      saveSession().catch(e => console.warn('Could not save review progress:', e));
      renderJobBook(true); // same rows while the viewer is open, so "Next item" stays predictable
    },
    onClose: () => renderJobBook(),
    onStep: k => {
      const groups = reviewGroups();
      if (groups[k]) openReviewGroup(k, firstOpenTarget(groups[k]));
    },
  });
  $('snc').addEventListener('click', e => {
    const a = e.target.closest('a.open');
    if (a) {
      e.preventDefault();
      const i = +a.dataset.doc;
      viewer.open({ title: `<strong>${esc(state.docs[i].short)}</strong>`, left: { doc: i, page: +a.dataset.page, hl: a.dataset.hl } });
      return;
    }
    const r = e.target.closest('[data-review]');
    if (r) openReview(+r.dataset.review);
    if (e.target.closest('[data-start-review]')) resumeReview();
  });
  $('snc').addEventListener('change', e => {
    const skipKey = e.target.dataset && e.target.dataset.skip;
    if (skipKey && skipKey in state.skip) {
      state.skip[skipKey] = e.target.checked;
      saveSession().catch(err => console.warn('Could not save skip options:', err));
      run();
      return;
    }
    if (e.target.id !== 'snc-review-scope') return;
    state.reviewScope = e.target.value;
    saveSession().catch(err => console.warn('Could not save review scope:', err));
    renderJobBook();
  });
  document.querySelectorAll('input[name="snc-mode"]').forEach(r => r.addEventListener('change', e => { state.mode = e.target.value; run(); }));

  let timer;
  ['snc-usefields', 'snc-usetext', 'snc-mintwo', 'snc-ignorecase', 'snc-ignoresep', 'snc-fieldfilter',
   'snc-aliases', 'snc-rules', 'snc-issuesonly'].forEach(id =>
    $(id) && $(id).addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(run, 250); }));

  // =====================================================================
  // Job book answers snapshot (CSV export) and comparison with a previous job book
  // =====================================================================
  const SNAP_HEADER = ['Source', 'Section', 'Field name', 'Label', 'Value', 'State', 'Page', 'Category', 'Review', 'Issue note', 'Document'];
  const csvCell = s => `"${String(s == null ? '' : s).replace(/"/g, '""')}"`;
  // Category lets the comparison ignore fields that are expected to change (dates always).
  function fieldCategory(field, value) {
    if (isDateField(field, value)) return 'date';
    if (isInitialsField(field)) return 'initials';
    if (isCompanyField(field)) return 'company';
    if (isVersionField(field)) return 'version';
    if (isPersonNameField(field)) return 'name';
    if (value && isYesNoValue(value)) return 'yesno';
    return '';
  }
  function snapshotRows() {
    const byField = new Map();
    if (state.jb) for (const e of state.jb.entries) {
      const rk = reviewKey(e), mark = state.reviews.get(rk) || 'pending';
      const note = mark === 'issue' ? noteDetails(state.reviewNotes.get(rk)).text : '';
      const cur = byField.get(fieldKey(e)) || { marks: [], notes: [] };
      cur.marks.push(mark); if (note) cur.notes.push(note);
      byField.set(fieldKey(e), cur);
    }
    const rows = [];
    for (const [docIndex, d] of state.docs.entries()) {
      const kind = sourceKind(d) || 'document';
      if (d.error) continue;
      const sourceLabel = kind === 'document' ? (d.short || d.name) : kindLabel(kind);
      const seen = new Map();
      const add = (f, value, blank) => {
        const base = `${sourceLabel.toLowerCase()}|${f.name}`, n = seen.get(base) || 0;
        seen.set(base, n + 1);
        const r = byField.get(fieldKey({ sourceId: d.id || d.name, name: f.name, label: f.label, value: blank ? (f.checkbox ? '(unticked)' : '(blank)') : value, page: f.page }));
        rows.push({
          key: `${base}#${n}`, source: sourceLabel, section: f.section || '', name: f.name, label: f.label || '', value,
          state: blank ? 'Blank' : isNA(value) ? 'N/A' : 'Populated', page: f.page, category: fieldCategory(f, value),
          review: r ? (r.marks.includes('issue') ? 'issue' : r.marks.every(x => x === 'verified') ? 'verified' : 'pending') : '',
          note: r ? r.notes.join(' | ') : '', document: d.name, docIndex, rect: f.rect,
        });
      };
      d.fieldList.forEach(f => add(f, String(f.value || '').trim(), false));
      (d.blankFields || []).forEach(f => add(f, '', true));
    }
    return rows;
  }
  function parseCsv(text) {
    const rows = [];
    let row = [], cell = '', quoted = false;
    text = text.replace(/^﻿/, '');
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false; } else cell += c;
      } else if (c === '"') quoted = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); cell = ''; rows.push(row); row = []; }
      else cell += c;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }
  function baselineFromCsv(text) {
    const table = parseCsv(text).filter(r => r.some(c => c !== ''));
    const header = table.shift() || [];
    const col = name => header.indexOf(name);
    if (['Source', 'Field name', 'State', 'Value'].some(name => col(name) < 0)) throw new Error('This is not a job book answers CSV. Use "Export answers CSV" from a finished job book.');
    const seen = new Map();
    return table.map(r => {
      const get = name => (col(name) >= 0 ? r[col(name)] : '') || '';
      const base = `${get('Source').toLowerCase()}|${get('Field name')}`, n = seen.get(base) || 0;
      seen.set(base, n + 1);
      return { key: `${base}#${n}`, source: get('Source'), section: get('Section'), name: get('Field name'), label: get('Label'), value: get('Value'), state: get('State'), page: get('Page'), category: get('Category') };
    });
  }
  function loadBaseline() {
    try { const saved = JSON.parse(localStorage.getItem('snc-compare-baseline') || 'null'); if (saved && Array.isArray(saved.rows)) state.baseline = saved; } catch (e) { /* storage unavailable */ }
  }
  function saveBaseline() {
    try {
      if (state.baseline) localStorage.setItem('snc-compare-baseline', JSON.stringify(state.baseline)); else localStorage.removeItem('snc-compare-baseline');
    } catch (e) { console.warn('Could not store the comparison CSV:', e); }
  }
  const SKIP_FOR_CATEGORY = { initials: 'initials', name: 'names', company: 'company', version: 'version', yesno: 'yesno' };
  function compareSnapshots(prev, cur) {
    const prevMap = new Map(prev.map(r => [r.key, r])), curMap = new Map(cur.map(r => [r.key, r]));
    const count = rows => ({ total: rows.length, populated: rows.filter(r => r.state === 'Populated').length, na: rows.filter(r => r.state === 'N/A').length, blank: rows.filter(r => r.state === 'Blank').length });
    const diffs = { blankNow: [], filledNow: [], naNow: [], valueNow: [], changed: [], onlyCurrent: [], onlyPrevious: [] };
    for (const r of cur) {
      const p = prevMap.get(r.key);
      if (!p) { diffs.onlyCurrent.push({ r }); continue; }
      if (r.category === 'date' || p.category === 'date') continue; // dates always change
      if (r.state !== p.state) {
        if (r.state === 'Blank') diffs.blankNow.push({ r, p });
        else if (p.state === 'Blank') diffs.filledNow.push({ r, p });
        else if (r.state === 'N/A') diffs.naNow.push({ r, p });
        else diffs.valueNow.push({ r, p });
      } else if (r.state === 'Populated') {
        const cat = r.category || p.category;
        if (cat && SKIP_FOR_CATEGORY[cat] && state.skip[SKIP_FOR_CATEGORY[cat]]) continue;
        if (alnum(r.value) !== alnum(p.value)) diffs.changed.push({ r, p });
      }
    }
    for (const p of prev) if (!curMap.has(p.key)) diffs.onlyPrevious.push({ p });
    return { prev: count(prev), cur: count(cur), diffs };
  }
  function renderCompare() {
    const box = $('snc-compare-summary'), nameEl = $('snc-compare-name'), clearBtn = $('snc-compare-clear');
    if (!box) return;
    const b = state.baseline;
    nameEl.textContent = b ? `${b.name} (${b.rows.length} fields)` : 'No comparison CSV loaded';
    clearBtn.hidden = !b;
    if (!b) { box.innerHTML = ''; return; }
    const cur = snapshotRows();
    if (!cur.length) { box.innerHTML = '<p class="src">Load a job book (ROMC / SIF) to compare it with the CSV.</p>'; return; }
    const c = compareSnapshots(b.rows, cur);
    const delta = (a, z) => { const d = z - a; return d === 0 ? '0' : (d > 0 ? '+' : '') + d; };
    const metric = (label, key) => `<tr><td>${label}</td><td>${c.prev[key]}</td><td>${c.cur[key]}</td><td>${delta(c.prev[key], c.cur[key])}</td></tr>`;
    state.cmpLists = {};
    const list = (key, title, items, fmt) => {
      state.cmpLists[key] = { title, items, fmt };
      if (!items.length) return `<div class="cmp-line cmp-zero">${title}: <strong>0</strong></div>`;
      const shown = items.slice(0, 300);
      return `<details class="cmp-line"><summary>${title}: <strong>${items.length}</strong></summary><ul>${shown.map((i, n) => `<li>${i.r ? `<button type="button" class="cmp-open" data-cmp-list="${key}" data-cmp-i="${n}" title="Open in the viewer">${fmt(i)}</button>` : fmt(i)}</li>`).join('')}${items.length > shown.length ? `<li class="src">…and ${items.length - shown.length} more</li>` : ''}</ul></details>`;
    };
    const lbl = r => `<strong>${esc(r.source)}</strong> ${esc(r.label || r.name)}`;
    const val = r => r.state === 'Populated' ? `“${esc(r.value)}”` : r.state;
    const move = i => `${lbl(i.r)}: ${val(i.p)} → ${val(i.r)}`;
    const d = c.diffs;
    box.innerHTML = `<table class="cmp-table"><thead><tr><th></th><th>Previous</th><th>Current</th><th>Change</th></tr></thead><tbody>${metric('Fields', 'total')}${metric('Populated', 'populated')}${metric('N/A', 'na')}${metric('Blank', 'blank')}</tbody></table>` +
      list('blankNow', 'Blank now, was filled or N/A', d.blankNow, move) +
      list('filledNow', 'Filled now, was blank', d.filledNow, move) +
      list('naNow', 'N/A now, was filled', d.naNow, move) +
      list('valueNow', 'Filled now, was N/A', d.valueNow, move) +
      list('changed', 'Different value', d.changed, move) +
      list('onlyCurrent', 'Only in this job book', d.onlyCurrent, i => lbl(i.r)) +
      list('onlyPrevious', 'Only in the comparison CSV', d.onlyPrevious, i => lbl(i.p)) +
      '<p class="src">Click a change to open it in the viewer; use the arrows to step through that list. Date fields are ignored. Initials, names, company and Yes/No answers follow the Skip toggles above for the “Different value” list.</p>';
  }
  // Open one change from the comparison summary in the viewer (view only, no marks).
  function openCompareItem(key, index) {
    const group = state.cmpLists && state.cmpLists[key];
    const item = group && group.items[index];
    if (!item || !item.r) return;
    const r = item.r, p = item.p;
    const val = x => x.state === 'Populated' ? `\u201c${x.value}\u201d` : x.state;
    viewer.open({
      title: `<strong>${esc(r.source)}: ${esc(r.label || r.name)}</strong> <span class="src">${esc(group.title)}</span>`,
      left: { doc: r.docIndex, page: r.page, hl: [r.label, r.value].filter(Boolean), rect: r.rect, sure: Array.isArray(r.rect) },
      tip: p ? `Previous job book: ${val(p)}  \u2192  This job book: ${val(r)} (page ${r.page}).` : `Not in the previous job book (page ${r.page}).`,
      step: { index, total: group.items.length },
      onStep: k => openCompareItem(key, k),
    });
  }
  $('snc-compare-summary').addEventListener('click', e => {
    const btn = e.target.closest('[data-cmp-list]');
    if (btn) openCompareItem(btn.dataset.cmpList, +btn.dataset.cmpI);
  });
  $('snc-compare-file').addEventListener('change', async e => {
    const file = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!file) return;
    try {
      state.baseline = { name: file.name, savedAt: new Date().toISOString(), rows: baselineFromCsv(await file.text()) };
      saveBaseline();
      renderCompare();
    } catch (err) { showProgress(err.message || String(err)); }
  });
  $('snc-compare-clear').addEventListener('click', () => { state.baseline = null; saveBaseline(); renderCompare(); });
  $('snc-snapshot').addEventListener('click', () => {
    const rows = snapshotRows();
    if (!rows.length) return;
    const lines = [SNAP_HEADER.map(csvCell).join(',')].concat(rows.map(r =>
      [r.source, r.section, r.name, r.label, r.value, r.state, r.page, r.category, r.review, r.note, r.document].map(csvCell).join(',')));
    const wtg = (state.docs.map(d => /wtg[_\s-]*(\d{4,})/i.exec(d.name)).find(Boolean) || [])[1];
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv' }));
    a.download = `job-book-answers${wtg ? '-WTG' + wtg : ''}-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  $('snc-issues-csv').addEventListener('click', () => {
    if (!state.jb) return;
    const xml = s => String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;');
    const row = cells => `<Row>${cells.map(cell => `<Cell><Data ss:Type="String">${xml(cell)}</Data></Cell>`).join('')}</Row>`;
    const rows = [row(['Source', 'Field', 'Value', 'Source page', 'Review target', 'Found in', 'Issue note', 'Attached document'])];
    for (const e of state.jb.entries.filter(entry => state.reviews.get(reviewKey(entry)) === 'issue')) {
      const issue = noteDetails(state.reviewNotes.get(reviewKey(e)));
      rows.push(row([kindLabel(e.sourceKind), e.label, e.value, e.page, e.targetLabel,
        e.found.map(f => `${state.docs[f.i].short} p.${f.pages.join('/')}`).join('; '), issue.text, issue.document?.name || '']));
    }
    const workbook = `<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"><Styles><Style ss:ID="header"><Font ss:Bold="1"/></Style></Styles><Worksheet ss:Name="Issues"><Table>${rows[0].replace('<Row>', '<Row ss:StyleID="header">')}${rows.slice(1).join('')}</Table></Worksheet></Workbook>`;
    const blob = new Blob([workbook], { type: 'application/vnd.ms-excel' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `job-book-issues-${new Date().toISOString().slice(0, 10)}.xml`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('snc-issues-pdf').addEventListener('click', () => {
    if (!state.jb || !window.jspdf) return alert('The PDF report library did not load. Please check your connection and try again.');
    const { jsPDF } = window.jspdf;
    const issues = state.jb.entries.filter(entry => state.reviews.get(reviewKey(entry)) === 'issue');
    const pdf = new jsPDF({ unit: 'pt', format: 'letter' });
    const left = 48, right = 564, bottom = 742;
    let y = 0;
    const header = (continued = false) => {
      pdf.setFillColor(25, 49, 79); pdf.rect(0, 0, 612, 78, 'F');
      pdf.setTextColor(255, 255, 255); pdf.setFont('helvetica', 'bold'); pdf.setFontSize(20); pdf.text('Job Book Issues Report', left, 35);
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(9); pdf.text(`${issues.length} active issue${issues.length === 1 ? '' : 's'}  •  Generated ${new Date().toLocaleString()}${continued ? '  •  Continued' : ''}`, left, 55);
      pdf.setTextColor(32, 42, 56); y = 104;
    };
    const pageBreak = height => { if (y + height <= bottom) return; pdf.addPage(); header(true); };
    const field = (label, value) => {
      const lines = pdf.splitTextToSize(value || '—', right - left - 110);
      const height = Math.max(15, lines.length * 12) + 5;
      pageBreak(height);
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(8); pdf.setTextColor(89, 103, 120); pdf.text(label.toUpperCase(), left + 12, y + 10);
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(10); pdf.setTextColor(32, 42, 56); pdf.text(lines, left + 120, y + 10);
      y += height;
    };
    header();
    issues.forEach((entry, index) => {
      const issue = noteDetails(state.reviewNotes.get(reviewKey(entry)));
      const title = `${index + 1}. ${kindLabel(entry.sourceKind)} — ${entry.label}`;
      const titleLines = pdf.splitTextToSize(title, right - left - 24);
      const estimate = 42 + titleLines.length * 12 + 80;
      pageBreak(estimate);
      pdf.setFillColor(245, 247, 250); pdf.roundedRect(left, y, right - left, 27 + titleLines.length * 12, 4, 4, 'F');
      pdf.setDrawColor(204, 67, 57); pdf.setLineWidth(3); pdf.line(left, y + 2, left, y + 25 + titleLines.length * 12);
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(11); pdf.setTextColor(32, 42, 56); pdf.text(titleLines, left + 12, y + 17);
      y += 36 + titleLines.length * 12;
      field('Value', entry.value);
      field('Review target', entry.targetLabel);
      field('Source page', `Page ${entry.page}`);
      field('Found in', entry.found.map(f => `${state.docs[f.i].short} — page ${f.pages.join(', ')}`).join('; '));
      field('Attached document', issue.document?.name || '—');
      field('Issue note', issue.text || 'No note entered');
      // Screenshots attached to the note, two per row, directly under it.
      const shots = issue.images || [];
      for (let k = 0; k < shots.length; k += 2) {
        const pair = shots.slice(k, k + 2), maxW = (right - left - 36) / 2, maxH = 210;
        const sized = pair.map(img => { const r = Math.min(maxW / img.w, maxH / img.h, 1); return { img, w: img.w * r, h: img.h * r }; });
        const rowH = Math.max(...sized.map(x => x.h)) + 22;
        pageBreak(rowH);
        sized.forEach((x, col) => {
          const ix = left + 12 + col * (maxW + 12);
          try { pdf.addImage(x.img.data, 'JPEG', ix, y + 2, x.w, x.h); } catch (err) { console.warn('Could not add screenshot to the report:', err); }
          pdf.setDrawColor(200, 205, 212); pdf.setLineWidth(0.5); pdf.rect(ix, y + 2, x.w, x.h);
          pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7); pdf.setTextColor(89, 103, 120);
          pdf.text(pdf.splitTextToSize(`${x.img.doc || 'Document'} — page ${x.img.page}`, maxW)[0], ix, y + x.h + 12);
        });
        y += rowH;
      }
      y += 14;
    });
    const total = pdf.getNumberOfPages();
    for (let page = 1; page <= total; page++) {
      pdf.setPage(page); pdf.setDrawColor(220, 225, 231); pdf.line(left, 760, right, 760);
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(8); pdf.setTextColor(89, 103, 120);
      pdf.text('Bench • Job Book Review', left, 775); pdf.text(`Page ${page} of ${total}`, right, 775, { align: 'right' });
    }
    pdf.save(`job-book-issues-${new Date().toISOString().slice(0, 10)}.pdf`);
  });

  loadBaseline();
  run();
  restoreDocuments();
})();
