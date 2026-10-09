(function () {
  'use strict';
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const normName = s => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  const alnum = s => String(s).toUpperCase().replace(/[^A-Z0-9]/g, '');

  // docs: { name, short, numPages, pages:[{text, norm, hasText}], fieldList:[{name,value,page,label,section}],
  //         fields:{name:Set}, text, error }
  const state = { docs: [], refIdx: -1, refAuto: true, mode: 'jobbook', rowsB: [], jb: null, jbRows: [], reviews: new Map(), reviewsFor: '' };

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
      fieldList: d.fieldList, fields: Object.fromEntries(Object.entries(d.fields).map(([k, v]) => [k, [...v]])),
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
    await transaction(db, STORE_SESSION, 'readwrite', store => store.put({ ids: state.docs.map(d => d.id), refId: ref && ref.id, refAuto: state.refAuto }, SESSION_KEY));
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
      state.refAuto = session.refAuto !== false;
      state.refIdx = state.docs.findIndex(d => d.id === session.refId);
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

  // Reviewer marks ("verified" / "issue") per ROMC row, remembered per reference file.
  const reviewKey = e => e.label + '\u0000' + e.n;
  function loadReviews(ref) {
    if (state.reviewsFor === ref.name) return;
    state.reviewsFor = ref.name;
    state.reviews = new Map();
    try { Object.entries(JSON.parse(localStorage.getItem('snc-reviews:' + ref.name) || '{}')).forEach(([k, v]) => state.reviews.set(k, v)); } catch (e) { /* storage unavailable */ }
  }
  function saveReviews() {
    try { localStorage.setItem('snc-reviews:' + state.reviewsFor, JSON.stringify(Object.fromEntries(state.reviews))); } catch (e) { /* storage unavailable */ }
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
    const doc = { id: newDocumentId(), name: displayName || file.name, blob: file, numPages: 0, pages: [], fieldList: [], fields: {}, text: '', error: null };
    let pdf;
    try {
      const data = new Uint8Array(await file.arrayBuffer());
      pdf = await pdfjsLib.getDocument({ data }).promise;
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
          if (!a.fieldName || a.checkBox || a.radioButton || a.pushButton) continue;
          let v = a.fieldValue;
          if (v == null || v === '' || v === 'Off') continue;
          if (Array.isArray(v)) v = v.join(', ');
          v = String(v).trim();
          if (!v) continue;
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
        doc.fieldList.push({ name: w.name, value: w.value, page: w.page, label, section: sec ? sec.num : '' });
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
      useFields: $('snc-usefields').checked,
      useText: $('snc-usetext').checked,
      minTwo: $('snc-mintwo').checked,
      ignoreCase: $('snc-ignorecase').checked,
      ignoreSep: $('snc-ignoresep').checked,
      fieldFilter: $('snc-fieldfilter').value.trim(),
      skip: $('snc-skip').value.trim(),
    };
  }

  function parseAliases() {
    const map = new Map();
    for (const line of $('snc-aliases').value.split('\n')) {
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
  // "ROMC item regex | filename regex [| photo]" — "photo" means the evidence is usually a
  // picture, so a value missing from the text is a visual check rather than a failure.
  const parseJbRules = () => parsePipeLines($('snc-jbrules').value, $('snc-jberr'), (labelSrc, rest) => {
    const [docSrc, flag = ''] = rest.split(/\s+\|\s+/);
    return { labelSrc, docSrc: docSrc.trim(), photo: /photo/i.test(flag), labelRe: new RegExp(labelSrc, 'i'), docRe: new RegExp(docSrc.trim(), 'i') };
  });

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
  // Mode A: job book check against a reference document (ROMC)
  // =====================================================================
  function compareJobBook() {
    const o = opts();
    const ref = state.docs[state.refIdx];
    if (!ref || ref.error) { state.jb = null; return; }
    let skip = null;
    try { skip = o.skip ? new RegExp(o.skip, 'i') : null; } catch (e) { skip = null; }
    const rules = parseJbRules();
    const others = state.docs.map((d, i) => ({ d, i })).filter(x => x.i !== state.refIdx && !x.d.error);

    // 1. Reference entries
    const entries = [];
    const seen = new Set();
    for (const f of ref.fieldList) {
      if (skip && (skip.test(f.label) || skip.test(f.value))) continue;
      const parts = f.value.split(/\s*[,;&]\s*/).filter(Boolean);
      for (const part of parts.length > 1 ? parts : [f.value]) {
        const v = part.trim();
        if (!serialLike(v)) continue;
        const key = f.label + '\u0000' + alnum(v);
        if (seen.has(key)) continue;
        seen.add(key);
        entries.push({ label: f.label, value: v, n: alnum(v), page: f.page, section: f.section });
      }
    }

    // 1b. VUI first: where a component has a VUI, check the VUI and leave out its serial
    //     number. Components with no VUI (e.g. aviation lights) keep the serial.
    if ($('snc-vuifirst').checked) {
      const withVui = new Set(entries.filter(e => /\bvuis?\b/i.test(e.label)).map(e => e.section || e.label));
      for (let k = entries.length - 1; k >= 0; k--) {
        const e = entries[k];
        if (/serial/i.test(e.label) && !/\bvui/i.test(e.label) && withVui.has(e.section || e.label)) entries.splice(k, 1);
      }
    }

    // 2. Duplicates: same value under two different components (sections)
    const bySection = new Map();
    for (const e of entries) {
      if (!bySection.has(e.n)) bySection.set(e.n, new Map());
      bySection.get(e.n).set(e.section || e.label, e.label);
    }

    // 3. Search every other document; apply rules
    for (const e of entries) {
      e.found = [];
      for (const { d, i } of others) {
        const pages = [];
        d.pages.forEach((pg, pi) => { if (pg.norm.includes(e.n)) pages.push(pi + 1); });
        if (pages.length) e.found.push({ i, pages });
      }
      const dupMap = bySection.get(e.n);
      e.dupWith = dupMap.size > 1 ? [...dupMap.values()].filter(l => l !== e.label) : [];

      const matched = rules.filter(r => r.labelRe.test(e.label));
      e.ruleDocs = matched.map(r => r.docSrc);
      e.expected = others.filter(({ d }) => matched.some(r => r.docRe.test(d.name))).map(x => x.i);
      const foundExpected = e.found.filter(f => e.expected.includes(f.i));

      if (matched.length) {
        if (!e.expected.length) e.status = 'notloaded';
        else if (foundExpected.length) e.status = 'confirmed';
        else {
          e.scanPages = [];
          for (const i of e.expected) {
            const sp = state.docs[i].pages.map((p, k) => p.hasText ? 0 : k + 1).filter(Boolean);
            if (sp.length) e.scanPages.push({ i, pages: sp });
          }
          e.photo = matched.some(r => r.photo);
          for (const i of e.expected) {
            const nm = nearMiss(state.docs[i], e.value);
            if (nm) { e.near = { ...nm, doc: i }; break; }
          }
          // A near-identical value in the text is the strongest sign of a typo.
          e.status = e.near ? 'mismatch' : (e.scanPages.length || e.photo) ? 'visual' : 'notfound';
        }
      } else {
        e.status = e.found.length ? 'alsofound' : 'refonly';
      }
      if (e.dupWith.length) e.status = 'duplicate';
    }

    // 4. Serials named in other documents that the reference doesn't contain
    const refSet = new Set(entries.map(e => e.n));
    const refAll = alnum(ref.fieldList.map(f => f.value).join(' '));
    const extra = [];
    const extraSeen = new Set();
    for (const { d, i } of others) {
      for (const s of labeledSerials(d)) {
        const n = alnum(s.value);
        if (refSet.has(n) || refAll.includes(n)) continue;
        const k = i + '\u0000' + n;
        if (extraSeen.has(k)) continue;
        extraSeen.add(k);
        extra.push({ ...s, doc: i });
      }
    }

    state.jb = { entries, extra, refIdx: state.refIdx };
  }

  const JB = {
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
      case 'duplicate': return `Same value also entered for: ${e.dupWith.map(esc).join('; ')}`;
      case 'mismatch': return `${openLink(e.near.doc, e.near.page, e.near.value, `${docName(e.near.doc)} p.${e.near.page}`)} has <span class="chip bad">${esc(e.near.value)}</span> — ${e.near.d} character${e.near.d > 1 ? 's' : ''} different`;
      case 'notfound': return `Not in ${e.expected.map(docName).join(', ')}`;
      case 'visual': {
        const where = e.scanPages.map(s => openLink(s.i, s.pages[0], e.value, `${docName(s.i)} p.${rangeList(s.pages)}`));
        return `Not in the text of ${e.expected.map(docName).join(', ')}. ` +
          (where.length ? `Check the scanned page(s): ${where.join('; ')}` : 'Check the photo of the tag');
      }
      case 'notloaded': return `Expected in a document matching <code>${esc(e.ruleDocs.join(' / '))}</code>, which isn't loaded`;
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
    const sum = $('snc-jb-summary'), tbl = $('snc-jb-table'), ex = $('snc-jb-extra');
    const jb = state.jb;
    if (!jb) {
      sum.innerHTML = state.docs.length ? '<p class="none">Pick a reference document (the ROMC) in the list above.</p>' : '';
      tbl.innerHTML = ''; ex.innerHTML = '';
      return;
    }
    const ref = state.docs[jb.refIdx];
    const issuesOnly = $('snc-issuesonly').checked;
    const counts = {};
    jb.entries.forEach(e => { counts[e.status] = (counts[e.status] || 0) + 1; });
    sum.innerHTML = `<p>Reference: <strong>${esc(ref.short)}</strong> · ${jb.entries.length} serials/VUIs checked</p><p>` +
      Object.keys(JB).filter(k => counts[k]).map(k => `<span class="st-${JB[k].c}">${counts[k]} ${JB[k].t.toLowerCase()}</span>`).join('') + '</p>';

    loadReviews(ref);
    const reviewed = Object.values(Object.fromEntries(state.reviews));
    const nVer = reviewed.filter(v => v === 'verified').length, nIss = reviewed.filter(v => v === 'issue').length;
    if (nVer || nIss) sum.innerHTML += `<p class="src">Your review: ${nVer} verified · ${nIss} marked as issue</p>`;

    const rows = keepRows ? state.jbRows : jb.entries
      .filter(e => !issuesOnly || (['mismatch', 'duplicate', 'notfound', 'visual', 'notloaded'].includes(e.status) && state.reviews.get(reviewKey(e)) !== 'verified') || state.reviews.get(reviewKey(e)) === 'issue')
      .sort((a, b) => JB[a.status].o - JB[b.status].o || a.page - b.page);
    state.jbRows = rows;
    let h = '<thead><tr><th>Status</th><th>ROMC item</th><th>Value</th><th>Found in</th><th>Notes</th><th></th></tr></thead><tbody>';
    if (!rows.length) h += `<tr><td colspan="6" class="none">${issuesOnly ? 'No open issues. Untick “Issues only” to see every serial.' : 'No serial-like values found in the reference document.'}</td></tr>`;
    rows.forEach((e, k) => {
      const found = e.found.map(f => `<div class="${e.expected.includes(f.i) ? 'exp' : ''}">${esc(state.docs[f.i].short)} <span class="src">${f.pages.slice(0, 4).map(pn => openLink(f.i, pn, e.value, 'p.' + pn)).join(', ')}${f.pages.length > 4 ? '…' : ''}</span></div>`).join('') || '<span class="none">—</span>';
      const mark = state.reviews.get(reviewKey(e));
      const badge = mark ? `<div><span class="mark mark-${mark}">${mark === 'verified' ? '✓ Verified' : '⚠ Issue'}</span></div>` : '';
      h += `<tr${mark ? ` class="row-${mark}"` : ''}><td><span class="st st-${JB[e.status].c}">${JB[e.status].t}</span>${badge}</td>` +
        `<td>${esc(e.label)}<div class="src">${openLink(jb.refIdx, e.page, e.value, 'ROMC p.' + e.page)}</div></td>` +
        `<td><span class="chip ${JB[e.status].c}">${esc(e.value)}</span></td><td>${found}</td><td class="notes">${jbDetail(e)}</td>` +
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
    const jbMode = state.mode === 'jobbook';
    document.querySelectorAll('[data-mode]').forEach(el => { el.hidden = el.dataset.mode !== state.mode; });
    if (jbMode) { compareJobBook(); renderJobBook(); } else { compareFields(); renderFields(); }
    $('snc-csv').disabled = jbMode ? !(state.jb && state.jb.entries.length) : !state.rowsB.length;
  }

  function renderDocs() {
    setShortNames();
    autoPickReference();
    $('snc-docs').innerHTML = state.docs.map((d, i) => {
      if (d.error) return `<li><strong>${esc(d.name)}</strong> <span class="err">Could not read: ${esc(d.error)}</span> <button class="btn btn-small" data-rm="${i}">Remove</button></li>`;
      const scanned = d.pages.filter(p => !p.hasText).length;
      const textInfo = scanned === 0 ? 'text' : scanned === d.numPages ? '<span class="warn-text">scanned – no text</span>' : `<span class="warn-text">${scanned} of ${d.numPages} pages scanned</span>`;
      return `<li><label class="ref" title="Use as reference (master) document"><input type="radio" name="snc-ref" value="${i}" ${i === state.refIdx ? 'checked' : ''}> Reference</label>
        <span class="docname" title="${esc(d.name)}">${openLink(i, 1, '', esc(d.short))}</span>
        <span class="meta">${d.numPages} p · ${d.fieldList.length} fields · ${textInfo}</span>
        <details data-i="${i}"><summary class="meta">Text &amp; fields</summary></details>
        <button class="btn btn-small" data-rm="${i}">Remove</button></li>`;
    }).join('');
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
    viewer.reset(); state.docs = []; state.refIdx = -1; state.refAuto = true; renderDocs(); run();
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
  function openReview(k) {
    const e = state.jbRows[k];
    if (!e) return;
    viewer.open({
      title: `<span class="st st-${JB[e.status].c}">${JB[e.status].t}</span> <strong>${esc(e.label)}</strong> <span class="chip ${JB[e.status].c}">${esc(e.value)}</span>`,
      left: { doc: state.jb.refIdx, page: e.page, hl: e.value, sure: true },
      right: reviewTarget(e),
      mark: { key: reviewKey(e), status: state.reviews.get(reviewKey(e)) || null },
      step: { index: k, total: state.jbRows.length },
    });
  }
  const viewer = window.createSncViewer({
    getDocs: () => state.docs,
    onMark: (key, status) => {
      if (status) state.reviews.set(key, status); else state.reviews.delete(key);
      saveReviews();
      renderJobBook(true); // same rows while the viewer is open, so "Next item" stays predictable
    },
    onClose: () => renderJobBook(),
    onStep: k => openReview(k),
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
  });
  document.querySelectorAll('input[name="snc-mode"]').forEach(r => r.addEventListener('change', e => { state.mode = e.target.value; run(); }));

  let timer;
  ['snc-usefields', 'snc-usetext', 'snc-mintwo', 'snc-ignorecase', 'snc-ignoresep', 'snc-fieldfilter',
   'snc-aliases', 'snc-rules', 'snc-jbrules', 'snc-skip', 'snc-issuesonly', 'snc-vuifirst'].forEach(id =>
    $(id).addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(run, 250); }));

  $('snc-csv').addEventListener('click', () => {
    const q = s => `"${String(s).replace(/"/g, '""')}"`;
    const lines = [];
    if (state.mode === 'jobbook' && state.jb) {
      lines.push(['Status', 'ROMC item', 'Value', 'ROMC page', 'Found in', 'Notes', 'Review'].map(q).join(','));
      for (const e of state.jb.entries) {
        lines.push([JB[e.status].t, e.label, e.value, e.page,
          e.found.map(f => `${state.docs[f.i].short} p.${f.pages.join('/')}`).join('; '),
          jbDetail(e).replace(/<[^>]+>/g, '').replace(/&amp;/g, '&'), state.reviews.get(reviewKey(e)) || ''].map(q).join(','));
      }
      if (state.jb.extra.length) {
        lines.push('', ['Not in ROMC', 'Document', 'Value', 'Page', 'Context'].map(q).join(','));
        for (const s of state.jb.extra) lines.push(['', state.docs[s.doc].short, s.value, s.page, s.context].map(q).join(','));
      }
    } else {
      const docs = state.cmpDocs || [];
      lines.push(['Status', 'Source', 'Item', ...docs.map(d => d.short)].map(q).join(','));
      for (const r of state.rowsB) lines.push([LABEL_B[r.status], r.src, r.label, ...r.cells.map(c => c.join('; '))].map(q).join(','));
    }
    const blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `serial-crosscheck-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  run();
  restoreDocuments();
})();
