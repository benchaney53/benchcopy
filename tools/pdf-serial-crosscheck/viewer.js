// In-page PDF viewer for Job Book Review.
// One pane for "open this document", two side-by-side panes for reviewing a ROMC row
// against the document that should contain it. Matching values are highlighted.
(function () {
  'use strict';

  const alnum = s => String(s || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const STANDARD_FONT_DATA_URL = 'https://unpkg.com/pdfjs-dist@3.11.174/standard_fonts/';

  // ---- Shared PDF cache (a few documents open at once; oldest is closed) ----
  const cache = new Map(); // blob -> Promise<PDFDocumentProxy>
  const activePdfLeases = new Map(); // blob -> pane count; never evict an open pane's PDF
  const urls = new WeakMap(); // blob -> object URL for "open in new tab"
  const CACHE_LIMIT = 8;
  function retainPdf(blob) { activePdfLeases.set(blob, (activePdfLeases.get(blob) || 0) + 1); }
  function releasePdf(blob) {
    const next = (activePdfLeases.get(blob) || 1) - 1;
    if (next > 0) activePdfLeases.set(blob, next); else activePdfLeases.delete(blob);
  }
  function trimCache() {
    // Never call PDFDocumentProxy.destroy() on a PDF one of the viewer panes may
    // still use. Destroying it leaves its worker transport null and breaks later
    // highlight/page rendering with sendWithPromise errors.
    while (cache.size > CACHE_LIMIT) {
      const victim = [...cache.keys()].find(blob => !activePdfLeases.has(blob));
      if (!victim) return;
      const pdfPromise = cache.get(victim);
      cache.delete(victim);
      pdfPromise.then(pdf => pdf.destroy()).catch(() => {});
    }
  }
  function getPdf(blob) {
    if (cache.has(blob)) {
      const p = cache.get(blob);
      cache.delete(blob); cache.set(blob, p); // most recently used last
      return p;
    }
    const p = blob.arrayBuffer().then(buf => pdfjsLib.getDocument({
      data: new Uint8Array(buf), standardFontDataUrl: STANDARD_FONT_DATA_URL,
    }).promise);
    cache.set(blob, p);
    trimCache();
    return p;
  }
  function blobUrl(blob) {
    if (!urls.has(blob)) urls.set(blob, URL.createObjectURL(blob));
    return urls.get(blob);
  }

  // Rectangles (PDF user space) where `value` appears on a page: in the text layer,
  // in form field values, or in text typed onto the page as an annotation.
  async function findRects(page, value) {
    const t = alnum(value);
    if (t.length < 3) return [];
    const rects = [];
    const tc = await page.getTextContent();
    let concat = '';
    const spans = [];
    for (const it of tc.items) {
      const a = alnum(it.str);
      if (!a) continue;
      spans.push({ start: concat.length, end: concat.length + a.length, it });
      concat += a;
    }
    for (let idx = concat.indexOf(t); idx >= 0; idx = concat.indexOf(t, idx + t.length)) {
      const end = idx + t.length;
      for (const s of spans) {
        if (s.end <= idx || s.start >= end) continue;
        const tr = s.it.transform;
        const h = Math.hypot(tr[2], tr[3]) || 10;
        rects.push([tr[4] - 1, tr[5] - h * 0.3, tr[4] + (s.it.width || h) + 1, tr[5] + h * 0.95]);
      }
    }
    let annots = [];
    try { annots = await page.getAnnotations(); } catch (e) { /* none */ }
    for (const a of annots) {
      let v = a.fieldValue;
      if (Array.isArray(v)) v = v.join(' ');
      const text = [v, a.contentsObj && a.contentsObj.str].filter(Boolean).join(' ');
      if (text && alnum(text).includes(t)) rects.push(a.rect);
    }
    return rects;
  }

  // ---- One viewer pane: every page in one continuous, lazily rendered scroll ----
  function Pane(el, getDocs) {
    el.innerHTML =
      '<div class="pane-bar">' +
      '<select class="pane-doc" aria-label="Document"></select>' +
      '<span class="pane-nav"><button type="button" data-act="prev" title="Previous page">&#8249;</button>' +
      '<input class="pane-page" type="number" min="1" aria-label="Page"><span class="pane-of"></span>' +
      '<button type="button" data-act="next" title="Next page">&#8250;</button></span>' +
      '<span class="pane-nav pane-zoom"><button type="button" data-act="zout" title="Zoom out (−)">Zoom &minus; <kbd>&minus;</kbd></button>' +
      '<button type="button" data-act="zreset" title="Reset zoom (0)">Fit <kbd>0</kbd></button>' +
      '<button type="button" data-act="zin" title="Zoom in (+)">Zoom + <kbd>+</kbd></button></span>' +
      '<span class="pane-nav pane-pan" aria-label="Pan document"><button type="button" data-act="panup" title="Pan up (W)">&#8593;<kbd>W</kbd></button>' +
      '<button type="button" data-act="panleft" title="Pan left (A)">&#8592;<kbd>A</kbd></button>' +
      '<button type="button" data-act="pandown" title="Pan down (S)">&#8595;<kbd>S</kbd></button>' +
      '<button type="button" data-act="panright" title="Pan right (D)">&#8594;<kbd>D</kbd></button></span>' +
      '<span class="pane-gesture-hint"><kbd>Ctrl</kbd>+wheel zoom · <kbd>Shift</kbd>+wheel horizontal · middle-drag pan</span>' +
      '<a class="pane-tab" target="_blank" rel="noopener" title="Open in the browser\'s PDF viewer">New tab &#8599;</a>' +
      '</div><div class="pane-msg"></div>' +
      '<div class="pane-body" tabindex="0" aria-label="Document viewer. Ctrl plus mouse wheel zooms; Shift plus mouse wheel pans horizontally; middle-mouse drag pans. Keyboard: plus/minus zoom, 0 fits, W A S D pan."><div class="pane-pages"></div></div>';
    const q = s => el.querySelector(s);
    const sel = q('.pane-doc'), pageIn = q('.pane-page'), of = q('.pane-of'), body = q('.pane-body');
    const list = q('.pane-pages'), msg = q('.pane-msg'), tab = q('.pane-tab');
    // pages[k] = { div, w, h (unscaled), canvas, hl, scale (rendered at), busy }
    const st = { doc: -1, pdf: null, blob: null, hl: '', hlTerms: [], rect: null, hlPage: 1, zoom: 1, fitMode: 'width', scale: 1, pages: [], current: 1, token: 0, observer: null };

    // A review field has both a human-readable label and its value.  Highlighting
    // both makes the source (left) pane unambiguous even when the value appears
    // elsewhere on the page.
    function setHighlight(value, rect, page) {
      const values = Array.isArray(value) ? value : [value];
      st.hlTerms = [...new Set(values.map(v => String(v || '').trim()).filter(v => alnum(v).length >= 3))];
      st.rect = Array.isArray(rect) && rect.length === 4 ? rect : null;
      st.hlPage = Math.max(1, +page || 1);
      // Include the exact field rectangle in the render cache key. This prevents a
      // repeated value from reusing a highlight belonging to another field.
      st.hl = st.hlTerms.join('\u0001') + (st.rect ? '\u0002' + st.rect.join(',') : '') + '\u0003' + st.hlPage;
    }
    function highlightLabel() { return st.hlTerms.join(' / '); }

    function fillDocs() {
      const docs = getDocs();
      sel.innerHTML = docs.map((d, i) => d.error ? '' : `<option value="${i}">${esc(d.short || d.name)}</option>`).join('');
      sel.value = String(st.doc);
    }

    function teardown() {
      if (st.observer) st.observer.disconnect();
      st.observer = null;
      st.pages.forEach(p => { if (p.task) p.task.cancel(); });
      st.pages = [];
      list.innerHTML = '';
    }

    // Comparison panes fit to width. A single-document view fits the full page
    // height, so it opens centered rather than looking overly zoomed in.
    function pageScale(p) {
      const widthScale = Math.max(200, body.clientWidth - 24) / Math.max(1, p.w);
      const heightScale = Math.max(200, body.clientHeight - 24) / Math.max(1, p.h);
      return (st.fitMode === 'height' ? heightScale : widthScale) * st.zoom;
    }

    // Size every page placeholder for the current zoom (keeps the scrollbar honest).
    function layout() {
      for (const p of st.pages) {
        const sc = pageScale(p);
        p.div.style.width = Math.floor(p.w * sc) + 'px';
        p.div.style.height = Math.floor(p.h * sc) + 'px';
      }
    }

    async function load(docIdx) {
      const token = ++st.token;
      teardown();
      if (st.blob) releasePdf(st.blob);
      st.doc = docIdx; st.pdf = null; st.blob = null;
      const d = getDocs()[docIdx];
      if (!d || !d.blob) { msg.textContent = 'Document not available.'; return false; }
      msg.textContent = 'Loading…';
      let pdf;
      retainPdf(d.blob);
      try { pdf = await getPdf(d.blob); } catch (e) { releasePdf(d.blob); msg.textContent = 'Could not open: ' + e.message; return false; }
      if (token !== st.token) { releasePdf(d.blob); return false; }
      const views = await Promise.all(Array.from({ length: pdf.numPages }, (_, k) =>
        pdf.getPage(k + 1).then(pg => pg.getViewport({ scale: 1 }))));
      if (token !== st.token) { releasePdf(d.blob); return false; }
      st.pdf = pdf; st.blob = d.blob;
      st.pages = views.map((v, k) => {
        const div = document.createElement('div');
        div.className = 'pane-page-box';
        div.dataset.page = k + 1;
        div.innerHTML = `<div class="page-num">${k + 1}</div><div class="pane-hl"></div>`;
        list.appendChild(div);
        return { div, w: v.width, h: v.height, canvas: null, scale: 0, hl: null, task: null };
      });
      pageIn.max = pdf.numPages; of.textContent = `/ ${pdf.numPages}`;
      fillDocs();
      layout();
      // Render pages as they come near the viewport; free them when far away.
      st.observer = new IntersectionObserver(entries => {
        for (const en of entries) {
          const p = st.pages[+en.target.dataset.page - 1];
          if (!p) continue;
          if (en.isIntersecting) renderPage(p, +en.target.dataset.page);
          else release(p);
        }
      }, { root: body, rootMargin: '1500px 0px' });
      st.pages.forEach(p => st.observer.observe(p.div));
      return true;
    }

    function release(p) {
      if (p.task) { p.task.cancel(); p.task = null; }
      if (p.canvas) { p.canvas.width = 0; p.canvas.remove(); p.canvas = null; }
      p.scale = 0; p.hl = null; p.hlPromise = null; p.pending = null;
    }

    // One render per page at a time: callers (scrolling, "go to page") share the same promise.
    function renderPage(p, n) {
      if (!st.pdf) return Promise.resolve();
      const want = pageScale(p);
      if (p.canvas && p.scale === want) return (p.hl === st.hl && p.hlPromise) ? p.hlPromise : highlight(p, n);
      if (p.pending && p.pendingScale === want) return p.pending;
      if (p.task) p.task.cancel();
      p.pendingScale = want;
      const prom = drawPage(p, n).finally(() => { if (p.pending === prom) p.pending = null; });
      p.pending = prom;
      return prom;
    }

    async function drawPage(p, n) {
      const token = st.token;
      let page;
      try { page = await st.pdf.getPage(n); }
      catch (e) { if (token === st.token) msg.textContent = 'This PDF needs to be reopened.'; return; }
      if (token !== st.token) return;
      const scale = pageScale(p);
      const vp = page.getViewport({ scale });
      const dpr = window.devicePixelRatio || 1;
      const canvas = document.createElement('canvas');
      canvas.width = Math.floor(vp.width * dpr);
      canvas.height = Math.floor(vp.height * dpr);
      canvas.style.width = Math.floor(vp.width) + 'px';
      canvas.style.height = Math.floor(vp.height) + 'px';
      const task = page.render({
        canvasContext: canvas.getContext('2d'), viewport: vp,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null,
        annotationMode: pdfjsLib.AnnotationMode.ENABLE,
      });
      p.task = task;
      try { await task.promise; } catch (e) { return; } // cancelled (scrolled away / zoomed)
      if (token !== st.token || p.task !== task) return;
      p.task = null;
      // Swap in the finished canvas so there's never a blank flash on re-render.
      if (p.canvas) p.canvas.remove();
      p.div.insertBefore(canvas, p.div.firstChild);
      p.canvas = canvas; p.scale = scale; p.hl = null;
      await highlight(p, n);
    }

    function highlight(p, n) {
      const want = st.hl;
      p.hl = want;
      p.hlPromise = drawHighlights(p, n, want, st.hlTerms.slice(), st.rect);
      return p.hlPromise;
    }

    async function drawHighlights(p, n, want, terms, exactRect) {
      const layer = p.div.querySelector('.pane-hl');
      layer.innerHTML = '';
      p.rects = [];
      if (!want || !st.pdf) return;
      // A review calls out one source page (or one target page), never every
      // rendered occurrence in a continuous-scroll document.
      if (n !== st.hlPage) return;
      let page;
      try { page = await st.pdf.getPage(n); }
      catch (e) { return; }
      // Form-field coordinates are the authoritative source location. Fall back
      // to text matching only for entries that did not originate in a PDF form.
      const rects = exactRect ? [exactRect] : (await Promise.all(terms.map(term => findRects(page, term)))).flat();
      if (p.hl !== want || !p.canvas) return;
      const out = [];
      const seen = new Set();
      const vp = page.getViewport({ scale: p.scale });
      for (const r of rects) {
        const [a, b, c, d] = vp.convertToViewportRectangle(r);
        const box = { left: Math.min(a, c), top: Math.min(b, d), width: Math.abs(c - a), height: Math.abs(d - b) };
        const key = [box.left, box.top, box.width, box.height].map(v => Math.round(v)).join(':');
        if (seen.has(key)) continue;
        seen.add(key);
        const div = document.createElement('div');
        div.className = 'hl';
        div.style.cssText = `left:${box.left}px;top:${box.top}px;width:${box.width}px;height:${box.height}px`;
        layer.appendChild(div);
        out.push(box);
      }
      p.rects = out;
    }

    // Page whose area is under the top third of the pane.
    function pageAtView() {
      const y = body.scrollTop + body.clientHeight * 0.3;
      let n = 1;
      for (let k = 0; k < st.pages.length; k++) { if (st.pages[k].div.offsetTop <= y) n = k + 1; else break; }
      return n;
    }
    function updateBar() {
      if (!st.pdf) return;
      st.current = pageAtView();
      if (document.activeElement !== pageIn) pageIn.value = st.current;
      tab.href = blobUrl(st.blob) + '#page=' + st.current;
    }
    let raf = 0;
    body.addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; updateBar(); }); });

    function scrollToPage(n, offset) {
      const p = st.pages[n - 1];
      if (!p) return;
      body.scrollTop = Math.max(0, p.div.offsetTop - 8 + (offset || 0));
      updateBar();
    }

    // Go to page n; once it's drawn, bring the first highlight into view.
    async function goTo(n) {
      n = Math.min(Math.max(1, n || 1), st.pages.length);
      scrollToPage(n);
      const p = st.pages[n - 1];
      await renderPage(p, n);
      if (st.hl && p.rects && p.rects.length) {
        const top = Math.min(...p.rects.map(r => r.top));
        scrollToPage(n, Math.max(0, top - body.clientHeight * 0.3));
      }
      return p;
    }

    function setMsg(p, n, kept) {
      if (!st.hl) { msg.textContent = ''; return; }
      const label = highlightLabel();
      if (kept) msg.innerHTML = `Stayed on p.${n} — <code>${esc(label)}</code> isn't in this document's text, so its location is a guess`;
      else if (p && p.rects && p.rects.length) msg.innerHTML = `Highlighted <code>${esc(label)}</code> on p.${n}`;
      else msg.innerHTML = `<code>${esc(label)}</code> isn't in the text of p.${n} — look for it in the image`;
    }

    // Re-layout for a new zoom or width while keeping the same spot in view.
    function relayout() {
      if (!st.pdf) return;
      const n = pageAtView(), p = st.pages[n - 1];
      const frac = (body.scrollTop - p.div.offsetTop) / Math.max(1, p.div.offsetHeight);
      layout();
      body.scrollTop = p.div.offsetTop + frac * p.div.offsetHeight;
      st.pages.forEach((pp, k) => { if (pp.canvas) renderPage(pp, k + 1); });
    }

    function zoomTo(next, anchor) {
      if (!st.pdf) return;
      next = Math.max(0.4, Math.min(4, next));
      if (Math.abs(next - st.zoom) < 0.001) return;
      const n = pageAtView(), p = st.pages[n - 1];
      const x = anchor && Number.isFinite(anchor.x) ? anchor.x : body.clientWidth / 2;
      const y = anchor && Number.isFinite(anchor.y) ? anchor.y : body.clientHeight / 2;
      const xFrac = (body.scrollLeft + x - p.div.offsetLeft) / Math.max(1, p.div.offsetWidth);
      const yFrac = (body.scrollTop + y - p.div.offsetTop) / Math.max(1, p.div.offsetHeight);
      st.zoom = next;
      layout();
      body.scrollLeft = Math.max(0, p.div.offsetLeft + xFrac * p.div.offsetWidth - x);
      body.scrollTop = Math.max(0, p.div.offsetTop + yFrac * p.div.offsetHeight - y);
      st.pages.forEach((pp, k) => { if (pp.canvas) renderPage(pp, k + 1); });
    }
    function pan(dx, dy) {
      // Keep panning at the actual document boundary. The small viewer gutter is
      // intentional, but no extra white space can be reached beyond it.
      const maxLeft = Math.max(0, body.scrollWidth - body.clientWidth);
      const maxTop = Math.max(0, body.scrollHeight - body.clientHeight);
      body.scrollLeft = Math.max(0, Math.min(maxLeft, body.scrollLeft + (dx || 0)));
      body.scrollTop = Math.max(0, Math.min(maxTop, body.scrollTop + (dy || 0)));
    }

    el.addEventListener('click', e => {
      const act = e.target.closest('[data-act]');
      if (!act || !st.pdf) return;
      const a = act.dataset.act;
      if (a === 'prev') scrollToPage(Math.max(1, st.current - 1));
      else if (a === 'next') scrollToPage(Math.min(st.pages.length, st.current + 1));
      else if (a === 'zin') zoomTo(st.zoom * 1.25);
      else if (a === 'zout') zoomTo(st.zoom / 1.25);
      else if (a === 'zreset') zoomTo(1);
      else if (a === 'panup') pan(0, -180);
      else if (a === 'panleft') pan(-180, 0);
      else if (a === 'pandown') pan(0, 180);
      else if (a === 'panright') pan(180, 0);
    });
    pageIn.addEventListener('change', () => scrollToPage(Math.min(Math.max(1, +pageIn.value || 1), st.pages.length)));
    sel.addEventListener('change', async () => {
      if (await load(+sel.value)) { const p = await goTo(1); setMsg(p, 1); }
    });
    body.addEventListener('wheel', e => {
      if (!st.pdf) return;
      if (e.shiftKey) { e.preventDefault(); pan(e.deltaY || e.deltaX, 0); return; }
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const bounds = body.getBoundingClientRect();
      zoomTo(st.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12), { x: e.clientX - bounds.left, y: e.clientY - bounds.top });
    }, { passive: false });
    body.addEventListener('keydown', e => {
      if (!st.pdf || e.ctrlKey || e.metaKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === '+' || key === '=') { e.preventDefault(); zoomTo(st.zoom * 1.25); }
      else if (key === '-' || key === '_') { e.preventDefault(); zoomTo(st.zoom / 1.25); }
      else if (key === '0') { e.preventDefault(); zoomTo(1); }
      else if (key === 'w') { e.preventDefault(); pan(0, -180); }
      else if (key === 'a') { e.preventDefault(); pan(-180, 0); }
      else if (key === 's') { e.preventDefault(); pan(0, 180); }
      else if (key === 'd') { e.preventDefault(); pan(180, 0); }
    });
    let dragging = null;
    body.addEventListener('pointerdown', e => {
      if (e.button !== 1 || !st.pdf) return;
      e.preventDefault();
      body.focus({ preventScroll: true });
      dragging = { x: e.clientX, y: e.clientY, left: body.scrollLeft, top: body.scrollTop };
      body.classList.add('panning');
      body.setPointerCapture(e.pointerId);
    });
    body.addEventListener('pointermove', e => {
      if (!dragging) return;
      body.scrollLeft = dragging.left - (e.clientX - dragging.x);
      body.scrollTop = dragging.top - (e.clientY - dragging.y);
    });
    const stopDrag = e => {
      if (!dragging) return;
      dragging = null;
      body.classList.remove('panning');
      if (body.hasPointerCapture(e.pointerId)) body.releasePointerCapture(e.pointerId);
    };
    body.addEventListener('pointerup', stopDrag);
    body.addEventListener('pointercancel', stopDrag);
    body.addEventListener('auxclick', e => { if (e.button === 1) e.preventDefault(); });

    return {
      // spec: { doc, page, hl, rect?, sure }. sure === false means "page is only a guess":
      // if this document is already open, stay where the reviewer is.
      async show(spec) {
        const same = st.pdf && spec.doc === st.doc;
        const fitChanged = st.fitMode !== (spec.fit || 'width');
        st.fitMode = spec.fit || 'width';
        setHighlight(spec.hl, spec.rect, spec.page);
        fillDocs();
        if (!same) {
          st.zoom = 1;
          if (!(await load(spec.doc))) return;
          const p = await goTo(spec.page);
          setMsg(p, spec.page);
          return;
        }
        if (fitChanged) relayout();
        // Same document: refresh highlights on drawn pages.
        st.pages.forEach((p, k) => { if (p.canvas) highlight(p, k + 1); });
        if (spec.sure === false) { setMsg(null, st.current, true); return; }
        const p = await goTo(spec.page);
        setMsg(p, spec.page);
      },
      relayout,
    };
  }

  // ---- The overlay with one or two panes ----
  window.createSncViewer = function ({ getDocs, onMark, onStep: defaultOnStep, onClose }) {
    // An opened item may bring its own step handler (e.g. a list of comparison changes).
    const onStep = k => (current && current.onStep ? current.onStep(k) : defaultOnStep(k));
    const root = document.createElement('div');
    root.className = 'snc-viewer';
    root.hidden = true;
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.innerHTML =
      '<div class="viewer-box">' +
      '<div class="viewer-bar">' +
      '<div class="viewer-title"></div>' +
      '<div class="viewer-targets" aria-label="Required review targets"></div>' +
      '<div class="viewer-actions">' +
      '<span class="viewer-step"><button type="button" class="btn btn-small" data-v="prev">&#8249; Previous <kbd>P</kbd></button>' +
      '<span class="viewer-count"></span>' +
      '<button type="button" class="btn btn-small" data-v="next">Next <kbd>N</kbd> &#8250;</button></span>' +
      '<span class="viewer-mark"><button type="button" class="btn btn-small mark-ok" data-v="verified">&#10003; Verified <kbd>V</kbd></button>' +
      '<button type="button" class="btn btn-small mark-bad" data-v="issue">&#9888; Issue <kbd>I</kbd></button>' +
      '<button type="button" class="btn btn-small" data-v="clear">Clear <kbd>U</kbd></button></span>' +
      '<button type="button" class="btn btn-small" data-v="unresolved">Next open <kbd>Space</kbd></button>' +
      '<button type="button" class="btn btn-small" data-v="close" title="Close (Esc)">Close <kbd>Esc</kbd></button>' +
      '</div></div>' +
      '<div class="viewer-tip" hidden></div>' +
      '<form class="viewer-note" hidden><label><strong>Issue note <span class="src">(optional &middot; one per item)</span></strong><textarea placeholder="Describe what is wrong or what needs follow-up…" aria-label="Issue note"></textarea></label><label><strong>Attach to document</strong><select aria-label="Attach issue note to document"></select></label><span class="note-error">Saved automatically</span></form>' +
      '<div class="viewer-panes"><div class="pane"></div><div class="pane"></div></div>' +
      '</div>';
    document.body.appendChild(root);
    const [elA, elB] = root.querySelectorAll('.pane');
    const paneA = Pane(elA, getDocs), paneB = Pane(elB, getDocs);
    const title = root.querySelector('.viewer-title'), tip = root.querySelector('.viewer-tip');
    const markBox = root.querySelector('.viewer-mark'), stepBox = root.querySelector('.viewer-step'), targetBox = root.querySelector('.viewer-targets');
    const noteBox = root.querySelector('.viewer-note'), noteInput = noteBox.querySelector('textarea'), noteDoc = noteBox.querySelector('select'), noteError = noteBox.querySelector('.note-error');
    let current = null;
    let lastFocus = null;

    function setMarkButtons(status) {
      markBox.querySelector('.mark-ok').classList.toggle('active', status === 'verified');
      markBox.querySelector('.mark-bad').classList.toggle('active', status === 'issue');
    }
    function renderTargets() {
      const targets = current && current.targets;
      targetBox.hidden = !targets || !targets.length;
      if (targetBox.hidden) { targetBox.innerHTML = ''; return; }
      targetBox.innerHTML = targets.map((target, index) =>
        `<button type="button" class="target-tab${index === current.targetIndex ? ' active' : ''}${target.status ? ' target-' + target.status : ''}" data-v="target" data-target="${index}"><kbd>${index + 1}</kbd> ${esc(target.label)}</button>`
      ).join('');
    }

    function close() {
      flushNote();
      root.hidden = true;
      document.body.classList.remove('snc-viewer-open');
      if (onClose) onClose();
      if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
    }
    function hideNote() { flushNote(); noteBox.hidden = true; }
    function showIssueNote(focus) {
      if (!current || !current.mark) return;
      noteInput.value = current.mark.note || '';
      const docs = current.documents || [];
      const targetDoc = current.targets && current.targets[current.targetIndex] && current.targets[current.targetIndex].right && current.targets[current.targetIndex].right.doc;
      const defaultDoc = docs.find(doc => doc.index === targetDoc) || docs.find(doc => current.right && doc.index === current.right.doc) || docs.find(doc => current.left && doc.index === current.left.doc);
      const selected = current.mark.noteDocument && current.mark.noteDocument.id || (defaultDoc && defaultDoc.id);
      noteDoc.innerHTML = docs.map(doc => `<option value="${esc(String(doc.id))}">${esc(doc.name)}</option>`).join('');
      noteDoc.hidden = !docs.length;
      if (docs.some(doc => String(doc.id) === String(selected))) noteDoc.value = String(selected);
      noteBox.hidden = false;
      if (focus) noteInput.focus();
    }
    // The note autosaves: one note (and one attached document) per review item.
    let noteTimer = null;
    function commitNote() {
      clearTimeout(noteTimer); noteTimer = null;
      if (!current || !current.mark || current.mark.status !== 'issue') return;
      const note = noteInput.value.trim();
      const doc = (current.documents || []).find(item => String(item.id) === noteDoc.value);
      const attachment = doc ? { id: doc.id, name: doc.name } : null;
      current.mark.note = note; current.mark.noteDocument = attachment;
      onMark(current.mark.key, 'issue', note, attachment);
      noteError.textContent = 'Saved automatically';
    }
    function flushNote() { if (noteTimer) commitNote(); }
    function queueNote() { noteError.textContent = 'Saving…'; clearTimeout(noteTimer); noteTimer = setTimeout(commitNote, 400); }

    root.addEventListener('click', e => {
      if (e.target === root) return close();
      const b = e.target.closest('[data-v]');
      if (!b) return;
      const v = b.dataset.v;
      if (v === 'close') close();
      else if ((v === 'prev' || v === 'next') && current && current.step) onStep(current.step.index + (v === 'next' ? 1 : -1));
      else if (v === 'target' && current && current.onTarget) current.onTarget(+b.dataset.target);
      else if (v === 'unresolved' && current && current.onNextUnresolved) current.onNextUnresolved();
      else if (v === 'clear' && current && current.mark) {
        hideNote();
        current.mark.status = null;
        if (current.targets) current.targets[current.targetIndex].status = null;
        current.mark.note = ''; current.mark.noteDocument = null;
        setMarkButtons(null); renderTargets(); onMark(current.mark.key, null);
      }
      else if ((v === 'verified' || v === 'issue') && current && current.mark) {
        if (v === 'issue') {
          if (current.mark.status !== 'issue') {
            current.mark.status = 'issue';
            if (current.targets) current.targets[current.targetIndex].status = 'issue';
            setMarkButtons('issue'); renderTargets();
            showIssueNote(true); commitNote();
            return;
          }
          hideNote();
          current.mark.status = null;
          if (current.targets) current.targets[current.targetIndex].status = null;
          setMarkButtons(null); renderTargets(); onMark(current.mark.key, null, null, null, true);
          return;
        }
        const next = current.mark.status === v ? null : v;
        current.mark.status = next;
        if (current.targets) current.targets[current.targetIndex].status = next;
        setMarkButtons(next);
        renderTargets();
        onMark(current.mark.key, next);
        if (next === 'verified' && current.onNextUnresolved) current.onNextUnresolved();
      }
    });
    noteBox.addEventListener('submit', e => e.preventDefault());
    noteInput.addEventListener('input', queueNote);
    noteDoc.addEventListener('change', commitNote);
    document.addEventListener('keydown', e => {
      if (root.hidden) return;
      if (e.target.matches('textarea')) {
        if (e.key === 'Escape') { e.preventDefault(); flushNote(); e.target.blur(); }
        return;
      }
      if (e.target.matches('input, select, [contenteditable="true"]')) return;
      const key = e.key.toLowerCase();
      if (key === 'escape') { e.preventDefault(); close(); }
      else if ((key === 'arrowleft' || key === 'p') && current && current.step && current.step.index > 0) { e.preventDefault(); onStep(current.step.index - 1); }
      else if ((key === 'arrowright' || key === 'n') && current && current.step && current.step.index < current.step.total - 1) { e.preventDefault(); onStep(current.step.index + 1); }
      else if (key === 'v' && current && current.mark) { e.preventDefault(); root.querySelector('[data-v="verified"]').click(); }
      else if (key === 'i' && current && current.mark) { e.preventDefault(); if (current.mark.status === 'issue') { showIssueNote(true); } else root.querySelector('[data-v="issue"]').click(); }
      else if (key === 'u' && current && current.mark) { e.preventDefault(); root.querySelector('[data-v="clear"]').click(); }
      else if (e.key === ' ') { e.preventDefault(); root.querySelector('[data-v="unresolved"]').click(); }
      else if (/^[1-9]$/.test(e.key) && current && current.targets && +e.key <= current.targets.length && current.onTarget) { e.preventDefault(); current.onTarget(+e.key - 1); }
    });
    let rt;
    window.addEventListener('resize', () => {
      if (root.hidden) return;
      clearTimeout(rt);
      rt = setTimeout(() => { paneA.relayout(); if (!elB.hidden) paneB.relayout(); }, 200);
    });

    return {
      // spec: { title, left, right?, documents?, targets?:[{label,right,status}], targetIndex?, mark?, step? }
      open(spec) {
        hideNote();
        current = spec;
        if (spec.targets) {
          spec.targetIndex = Math.max(0, Math.min(spec.targetIndex || 0, spec.targets.length - 1));
          spec.right = spec.targets[spec.targetIndex].right;
        }
        if (root.hidden) lastFocus = document.activeElement;
        title.innerHTML = spec.title || '';
        tip.textContent = spec.tip || '';
        tip.hidden = !spec.tip;
        renderTargets();
        markBox.hidden = !spec.mark;
        stepBox.hidden = !spec.step;
        if (spec.mark) { spec.mark.note = spec.mark.note || ''; spec.mark.noteDocument = spec.mark.noteDocument || null; setMarkButtons(spec.mark.status); if (spec.mark.status === 'issue') showIssueNote(false); }
        if (spec.step) root.querySelector('.viewer-count').textContent = `${spec.step.index + 1} of ${spec.step.total}`;
        if (spec.step) {
          stepBox.querySelector('[data-v="prev"]').disabled = spec.step.index <= 0;
          stepBox.querySelector('[data-v="next"]').disabled = spec.step.index >= spec.step.total - 1;
        }
        elB.hidden = !spec.right;
        root.classList.toggle('split', !!spec.right);
        root.hidden = false;
        document.body.classList.add('snc-viewer-open');
        root.querySelector('[data-v="close"]').focus();
        // Panes size themselves to their width, so render after layout.
        requestAnimationFrame(() => {
          paneA.show({ ...spec.left, fit: spec.right ? 'width' : 'height' });
          if (spec.right) paneB.show({ ...spec.right, fit: 'width' });
        });
      },
      close,
      reset() { cache.forEach(p => p.then(pdf => pdf.destroy()).catch(() => {})); cache.clear(); activePdfLeases.clear(); },
    };
  };
})();
