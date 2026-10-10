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
      if (!st.hl || (Array.isArray(st.hl) && !st.hl.filter(Boolean).length)) { msg.textContent = ''; return; }
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

    // ---- Area capture: drag a rectangle on a page to get a picture of it ----
    let captureCb = null, capSel = null;
    const clampTo = (value, max) => Math.max(0, Math.min(value, max));
    function startCapture(cb) { captureCb = cb; body.classList.add('capturing'); }
    function stopCapture() {
      captureCb = null; body.classList.remove('capturing');
      if (capSel) { capSel.el.remove(); capSel = null; }
    }
    function drawSel() {
      Object.assign(capSel.el.style, {
        left: Math.min(capSel.x0, capSel.x1) + 'px', top: Math.min(capSel.y0, capSel.y1) + 'px',
        width: Math.abs(capSel.x1 - capSel.x0) + 'px', height: Math.abs(capSel.y1 - capSel.y0) + 'px',
      });
    }
    body.addEventListener('pointerdown', e => {
      if (!captureCb || e.button !== 0 || !st.pdf) return;
      const box = e.target.closest && e.target.closest('.pane-page-box');
      if (!box) return;
      e.preventDefault();
      const r = box.getBoundingClientRect();
      const el = document.createElement('div');
      el.className = 'pane-capture-capSel';
      box.appendChild(el);
      capSel = { box, el, x0: clampTo(e.clientX - r.left, r.width), y0: clampTo(e.clientY - r.top, r.height) };
      capSel.x1 = capSel.x0; capSel.y1 = capSel.y0;
      drawSel();
      try { body.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or already-ended pointer */ }
    });
    body.addEventListener('pointermove', e => {
      if (!capSel) return;
      const r = capSel.box.getBoundingClientRect();
      capSel.x1 = clampTo(e.clientX - r.left, r.width); capSel.y1 = clampTo(e.clientY - r.top, r.height);
      drawSel();
    });
    function finishSel(e, cancelled) {
      if (!capSel) return;
      const { box, el } = capSel, cb = captureCb;
      if (!cancelled && Number.isFinite(e.clientX)) {
        // Use the release position too, so a quick drag without move events still counts.
        const br = box.getBoundingClientRect();
        capSel.x1 = clampTo(e.clientX - br.left, br.width); capSel.y1 = clampTo(e.clientY - br.top, br.height);
      }
      const x = Math.min(capSel.x0, capSel.x1), y = Math.min(capSel.y0, capSel.y1), w = Math.abs(capSel.x1 - capSel.x0), h = Math.abs(capSel.y1 - capSel.y0);
      const page = +box.dataset.page, p = st.pages[page - 1];
      el.remove(); capSel = null;
      try { if (body.hasPointerCapture(e.pointerId)) body.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }
      if (cancelled || w < 8 || h < 8 || !p || !p.canvas) return; // too small: stay in capture mode
      // Map the selection (relative to the page box) onto the rendered canvas through the canvas's
      // actual on-screen size, so zoom level, scroll position and a canvas that is mid-re-render all crop correctly.
      const br = box.getBoundingClientRect(), cr = p.canvas.getBoundingClientRect();
      const sx = p.canvas.width / cr.width, sy = p.canvas.height / cr.height;
      const left = Math.max(0, x - (cr.left - br.left)), top = Math.max(0, y - (cr.top - br.top));
      const right = Math.min(cr.width, x + w - (cr.left - br.left)), bottom = Math.min(cr.height, y + h - (cr.top - br.top));
      if (right - left < 4 || bottom - top < 4) return;
      const cw = Math.max(1, Math.round((right - left) * sx)), ch = Math.max(1, Math.round((bottom - top) * sy)), shrink = Math.min(1, 1100 / cw);
      const out = document.createElement('canvas');
      out.width = Math.max(1, Math.round(cw * shrink)); out.height = Math.max(1, Math.round(ch * shrink));
      const ctx = out.getContext('2d');
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, out.width, out.height);
      ctx.drawImage(p.canvas, left * sx, top * sy, cw, ch, 0, 0, out.width, out.height);
      const d = getDocs()[st.doc];
      stopCapture();
      if (cb) cb({ data: out.toDataURL('image/jpeg', 0.82), w: out.width, h: out.height, doc: d ? d.name : '', page });
    }
    body.addEventListener('pointerup', e => finishSel(e, false));
    body.addEventListener('pointercancel', e => finishSel(e, true));

    return {
      startCapture, stopCapture,
      zoomIn() { if (st.pdf) zoomTo(st.zoom * 1.25); },
      zoomOut() { if (st.pdf) zoomTo(st.zoom / 1.25); },
      zoomReset() { if (st.pdf) zoomTo(1); },
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
      '<aside class="viewer-rail" aria-label="Review tools">' +
      '<div class="viewer-actions">' +
      '<span class="viewer-step"><button type="button" class="btn btn-small" data-v="prev">&#8249; Previous <kbd>P</kbd> <kbd>&uarr;</kbd></button>' +
      '<span class="viewer-count"></span>' +
      '<button type="button" class="btn btn-small" data-v="next">Next <kbd>N</kbd> <kbd>&darr;</kbd> &#8250;</button></span>' +
      '<span class="viewer-mark"><button type="button" class="btn btn-small mark-ok" data-v="verified">&#10003; Verified <kbd>V</kbd></button>' +
      '<button type="button" class="btn btn-small mark-bad" data-v="issue">&#9888; Issue <kbd>I</kbd></button>' +
      '<button type="button" class="btn btn-small" data-v="clear">Clear <kbd>U</kbd></button></span>' +
      '<button type="button" class="btn btn-small" data-v="unresolved">Next open <kbd>Space</kbd></button>' +
      '</div>' +
      '<div class="viewer-docs-wrap" hidden><div class="viewer-docs-title">Documents</div><div class="viewer-docs"></div></div>' +
      '</aside>' +
      '<div class="viewer-main">' +
      '<div class="viewer-bar">' +
      '<button type="button" class="viewer-close" data-v="close" title="Close (Esc)" aria-label="Close">&times;</button>' +
      '<div class="viewer-title"></div>' +
      '<div class="viewer-targets" aria-label="Required review targets"></div>' +
      '</div>' +
      '<div class="viewer-tip" hidden></div>' +
      '<form class="viewer-note" hidden><label><strong>Issue note <span class="src">(optional &middot; one per item)</span></strong><textarea placeholder="Describe what is wrong or what needs follow-up…" aria-label="Issue note"></textarea></label><label><strong>Attach to document</strong><select aria-label="Attach issue note to document"></select></label><button type="button" class="btn btn-small" data-v="capture" title="Click, then drag over a PDF to attach a picture of that area to this note">&#9986; Attach area</button><span class="note-error">Saved automatically</span><div class="note-images"></div></form>' +
      '<div class="viewer-panes"><div class="pane"></div><div class="pane"></div><div class="pane"></div><div class="pane"></div></div>' +
      '</div></div>';
    document.body.appendChild(root);
    const paneEls = [...root.querySelectorAll('.pane')];
    const panes = paneEls.map(el => Pane(el, getDocs));
    // Every pane after the first shows one required review target; a small label says which.
    paneEls.slice(1).forEach(el => el.insertAdjacentHTML('afterbegin', '<div class="pane-target-label" hidden></div>'));
    const paneA = panes[0];
    // + / - / 0 zoom the pane under the mouse (or every open pane) from anywhere in the viewer.
    let hoverPane = -1;
    paneEls.forEach((el, i) => el.addEventListener('pointerenter', () => { hoverPane = i; }));
    root.addEventListener('pointerleave', () => { hoverPane = -1; });
    function zoomKey(action) {
      const targets = hoverPane >= 0 && !paneEls[hoverPane].hidden ? [panes[hoverPane]] : panes.filter((p, i) => !paneEls[i].hidden);
      targets.forEach(p => p[action]());
    }
    const targetPanes = panes.slice(1), targetEls = paneEls.slice(1);
    const title = root.querySelector('.viewer-title'), tip = root.querySelector('.viewer-tip');
    const markBox = root.querySelector('.viewer-mark'), stepBox = root.querySelector('.viewer-step'), targetBox = root.querySelector('.viewer-targets');
    const noteBox = root.querySelector('.viewer-note'), noteInput = noteBox.querySelector('textarea'), noteDoc = noteBox.querySelector('select'), noteError = noteBox.querySelector('.note-error');
    let current = null;
    let lastFocus = null;
    const noteImages = noteBox.querySelector('.note-images'), captureBtn = noteBox.querySelector('[data-v="capture"]');
    let capturing = false;

    // Documents panel: every document in the review with its progress; click to jump to its first open item.
    const docsWrap = root.querySelector('.viewer-docs-wrap'), docsBox = root.querySelector('.viewer-docs');
    function renderDocNav() {
      const items = current && current.docNav ? current.docNav() : null;
      docsWrap.hidden = !items || !items.length;
      if (docsWrap.hidden) { docsBox.innerHTML = ''; return; }
      docsBox.innerHTML = items.map((d, i) =>
        `<button type="button" class="doc-nav${d.active ? ' active' : ''}${d.done >= d.total ? ' complete' : ''}" data-v="doc" data-i="${i}" title="${esc(d.name)}"><span class="doc-nav-name">${esc(d.label)}</span><span class="doc-nav-count">${d.done}/${d.total}</span></button>`).join('');
      const active = docsBox.querySelector('.doc-nav.active');
      if (active) active.scrollIntoView({ block: 'nearest' });
    }
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
    function hideNote() { flushNote(); cancelCapture(); noteBox.hidden = true; }
    function renderNoteImages() {
      const images = (current && current.mark && current.mark.images) || [];
      noteImages.hidden = !images.length;
      noteImages.innerHTML = images.map((img, i) =>
        `<span class="note-img"><img src="${img.data}" alt="Attached area"><small>${esc(img.doc || '')} p.${img.page}</small><button type="button" data-v="rm-image" data-i="${i}" aria-label="Remove picture" title="Remove picture">&times;</button></span>`).join('');
    }
    function cancelCapture() {
      panes.forEach(p => p.stopCapture());
      capturing = false;
      if (captureBtn) captureBtn.innerHTML = '&#9986; Attach area';
    }
    function startCaptureMode() {
      capturing = true;
      captureBtn.textContent = 'Drag over a PDF… (Esc cancels)';
      const done = img => {
        panes.forEach(p => p.stopCapture());
        capturing = false; captureBtn.innerHTML = '&#9986; Attach area';
        current.mark.images = (current.mark.images || []).concat(img);
        renderNoteImages(); commitNote();
      };
      panes.forEach(p => p.startCapture(done));
    }
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
      renderNoteImages();
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
      onMark(current.mark.key, 'issue', note, attachment, false, current.mark.images || []);
      renderDocNav();
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
      else if (v === 'doc' && current && current.docNav) { const items = current.docNav(); if (items[+b.dataset.i]) items[+b.dataset.i].onSelect(); }
      else if (v === 'capture') { if (capturing) cancelCapture(); else startCaptureMode(); }
      else if (v === 'rm-image' && current && current.mark) { current.mark.images.splice(+b.dataset.i, 1); renderNoteImages(); commitNote(); }
      else if ((v === 'prev' || v === 'next') && current && current.step) onStep(current.step.index + (v === 'next' ? 1 : -1));
      else if (v === 'target' && current && current.onTarget) current.onTarget(+b.dataset.target);
      else if (v === 'unresolved' && current && current.onNextUnresolved) current.onNextUnresolved();
      else if (v === 'clear' && current && current.mark) {
        hideNote();
        current.mark.status = null;
        if (current.targets) current.targets[current.targetIndex].status = null;
        current.mark.note = ''; current.mark.noteDocument = null; current.mark.images = [];
        setMarkButtons(null); renderTargets(); onMark(current.mark.key, null); renderDocNav();
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
          setMarkButtons(null); renderTargets(); onMark(current.mark.key, null, null, null, true); renderDocNav();
          return;
        }
        const next = current.mark.status === v ? null : v;
        current.mark.status = next;
        if (current.targets) current.targets[current.targetIndex].status = next;
        setMarkButtons(next);
        renderTargets();
        onMark(current.mark.key, next);
        renderDocNav();
        if (next === 'verified' && current.onNextUnresolved) current.onNextUnresolved();
      }
    });
    noteBox.addEventListener('submit', e => e.preventDefault());
    noteInput.addEventListener('input', queueNote);
    noteDoc.addEventListener('change', commitNote);
    document.addEventListener('keydown', e => {
      if (root.hidden) return;
      if (capturing && e.key === 'Escape') { e.preventDefault(); cancelCapture(); return; }
      if (e.target.matches('textarea')) {
        if (e.key === 'Escape') { e.preventDefault(); flushNote(); e.target.blur(); }
        return;
      }
      if (e.target.matches('input, select, [contenteditable="true"]')) return;
      const key = e.key.toLowerCase();
      if (!e.ctrlKey && !e.metaKey && !e.altKey && !e.target.closest('.pane-body')) {
        if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomKey('zoomIn'); return; }
        if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomKey('zoomOut'); return; }
        if (e.key === '0') { e.preventDefault(); zoomKey('zoomReset'); return; }
      }
      if (key === 'escape') { e.preventDefault(); close(); }
      else if ((key === 'arrowleft' || key === 'arrowup' || key === 'p') && current && current.step && current.step.index > 0) { e.preventDefault(); onStep(current.step.index - 1); }
      else if ((key === 'arrowright' || key === 'arrowdown' || key === 'n') && current && current.step && current.step.index < current.step.total - 1) { e.preventDefault(); onStep(current.step.index + 1); }
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
      rt = setTimeout(() => { paneA.relayout(); targetPanes.forEach((p, i) => { if (!targetEls[i].hidden) p.relayout(); }); }, 200);
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
        renderDocNav();
        markBox.hidden = !spec.mark;
        stepBox.hidden = !spec.step;
        if (spec.mark) { spec.mark.note = spec.mark.note || ''; spec.mark.noteDocument = spec.mark.noteDocument || null; spec.mark.images = spec.mark.images || []; setMarkButtons(spec.mark.status); if (spec.mark.status === 'issue') showIssueNote(false); }
        if (spec.step) root.querySelector('.viewer-count').textContent = `${spec.step.index + 1} of ${spec.step.total}`;
        if (spec.step) {
          stepBox.querySelector('[data-v="prev"]').disabled = spec.step.index <= 0;
          stepBox.querySelector('[data-v="next"]').disabled = spec.step.index >= spec.step.total - 1;
        }
        // Show every document the item must be checked against at the same time.
        const shown = spec.targets
          ? spec.targets.map((target, index) => ({ index, label: target.label, right: target.right, status: target.status })).filter(item => item.right).slice(0, targetEls.length)
          : (spec.right ? [{ index: null, label: '', right: spec.right }] : []);
        targetEls.forEach((el, i) => {
          const item = shown[i];
          el.hidden = !item;
          const label = el.querySelector('.pane-target-label');
          label.hidden = !item || shown.length < 2 && !spec.targets;
          if (item) {
            label.textContent = item.index == null ? '' : `${item.index + 1}. ${item.label}`;
            label.className = 'pane-target-label' + (item.index === spec.targetIndex ? ' active' : '') + (item.status ? ' target-' + item.status : '');
          }
          el.classList.toggle('active-target', !!item && item.index === spec.targetIndex && shown.length > 1);
        });
        root.classList.toggle('split', shown.length > 0);
        root.dataset.panes = String(1 + shown.length);
        root.hidden = false;
        document.body.classList.add('snc-viewer-open');
        root.querySelector('[data-v="close"]').focus();
        // Panes size themselves to their width, so render after layout.
        requestAnimationFrame(() => {
          paneA.show({ ...spec.left, fit: shown.length ? 'width' : 'height' });
          shown.forEach((item, i) => targetPanes[i].show({ ...item.right, fit: 'width' }));
        });
      },
      close,
      reset() { cache.forEach(p => p.then(pdf => pdf.destroy()).catch(() => {})); cache.clear(); activePdfLeases.clear(); },
    };
  };
})();
