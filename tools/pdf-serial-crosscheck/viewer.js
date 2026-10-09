// In-page PDF viewer for PDF Serial Cross-Check.
// One pane for "open this document", two side-by-side panes for reviewing a ROMC row
// against the document that should contain it. Matching values are highlighted.
(function () {
  'use strict';

  const alnum = s => String(s || '').toUpperCase().replace(/[^A-Z0-9]/g, '');
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  // ---- Shared PDF cache (a few documents open at once; oldest is closed) ----
  const cache = new Map(); // blob -> Promise<PDFDocumentProxy>
  const urls = new WeakMap(); // blob -> object URL for "open in new tab"
  function getPdf(blob) {
    if (cache.has(blob)) {
      const p = cache.get(blob);
      cache.delete(blob); cache.set(blob, p); // most recently used last
      return p;
    }
    const p = blob.arrayBuffer().then(buf => pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise);
    cache.set(blob, p);
    while (cache.size > 4) {
      const [oldBlob, oldP] = cache.entries().next().value;
      cache.delete(oldBlob);
      oldP.then(pdf => pdf.destroy()).catch(() => {});
    }
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
      '<span class="pane-nav"><button type="button" data-act="zout" title="Zoom out">&minus;</button>' +
      '<button type="button" data-act="zin" title="Zoom in">+</button></span>' +
      '<a class="pane-tab" target="_blank" rel="noopener" title="Open in the browser\'s PDF viewer">New tab &#8599;</a>' +
      '</div><div class="pane-msg"></div>' +
      '<div class="pane-body"><div class="pane-pages"></div></div>';
    const q = s => el.querySelector(s);
    const sel = q('.pane-doc'), pageIn = q('.pane-page'), of = q('.pane-of'), body = q('.pane-body');
    const list = q('.pane-pages'), msg = q('.pane-msg'), tab = q('.pane-tab');
    // pages[k] = { div, w, h (unscaled), canvas, hl, scale (rendered at), busy }
    const st = { doc: -1, pdf: null, blob: null, hl: '', zoom: 1, scale: 1, pages: [], current: 1, token: 0, observer: null };

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

    // Each page fits the pane width on its own, so one huge photo page doesn't shrink the rest.
    function pageScale(p) {
      return Math.max(200, body.clientWidth - 24) / Math.max(1, p.w) * st.zoom;
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
      st.doc = docIdx; st.pdf = null;
      const d = getDocs()[docIdx];
      if (!d || !d.blob) { msg.textContent = 'Document not available.'; return false; }
      msg.textContent = 'Loading…';
      let pdf;
      try { pdf = await getPdf(d.blob); } catch (e) { msg.textContent = 'Could not open: ' + e.message; return false; }
      if (token !== st.token) return false;
      const views = await Promise.all(Array.from({ length: pdf.numPages }, (_, k) =>
        pdf.getPage(k + 1).then(pg => pg.getViewport({ scale: 1 }))));
      if (token !== st.token) return false;
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
      const page = await st.pdf.getPage(n);
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
      p.hlPromise = drawHighlights(p, n, want);
      return p.hlPromise;
    }

    async function drawHighlights(p, n, want) {
      const layer = p.div.querySelector('.pane-hl');
      layer.innerHTML = '';
      p.rects = [];
      if (!want || !st.pdf) return;
      const page = await st.pdf.getPage(n);
      const rects = await findRects(page, want);
      if (p.hl !== want || !p.canvas) return;
      const out = [];
      const vp = page.getViewport({ scale: p.scale });
      for (const r of rects) {
        const [a, b, c, d] = vp.convertToViewportRectangle(r);
        const box = { left: Math.min(a, c), top: Math.min(b, d), width: Math.abs(c - a), height: Math.abs(d - b) };
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
      if (kept) msg.innerHTML = `Stayed on p.${n} — <code>${esc(st.hl)}</code> isn't in this document's text, so its location is a guess`;
      else if (p && p.rects && p.rects.length) msg.innerHTML = `Highlighted <code>${esc(st.hl)}</code> on p.${n}`;
      else msg.innerHTML = `<code>${esc(st.hl)}</code> isn't in the text of p.${n} — look for it in the image`;
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

    el.addEventListener('click', e => {
      const act = e.target.closest('[data-act]');
      if (!act || !st.pdf) return;
      const a = act.dataset.act;
      if (a === 'prev') scrollToPage(Math.max(1, st.current - 1));
      else if (a === 'next') scrollToPage(Math.min(st.pages.length, st.current + 1));
      else if (a === 'zin') { st.zoom = Math.min(4, st.zoom * 1.25); relayout(); }
      else if (a === 'zout') { st.zoom = Math.max(0.4, st.zoom / 1.25); relayout(); }
    });
    pageIn.addEventListener('change', () => scrollToPage(Math.min(Math.max(1, +pageIn.value || 1), st.pages.length)));
    sel.addEventListener('change', async () => {
      if (await load(+sel.value)) { const p = await goTo(1); setMsg(p, 1); }
    });

    return {
      // spec: { doc, page, hl, sure }. sure === false means "page is only a guess":
      // if this document is already open, stay where the reviewer is.
      async show(spec) {
        const same = st.pdf && spec.doc === st.doc;
        st.hl = spec.hl || '';
        fillDocs();
        if (!same) {
          st.zoom = 1;
          if (!(await load(spec.doc))) return;
          const p = await goTo(spec.page);
          setMsg(p, spec.page);
          return;
        }
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
  window.createSncViewer = function ({ getDocs, onMark, onStep, onClose }) {
    const root = document.createElement('div');
    root.className = 'snc-viewer';
    root.hidden = true;
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.innerHTML =
      '<div class="viewer-box">' +
      '<div class="viewer-bar">' +
      '<div class="viewer-title"></div>' +
      '<div class="viewer-actions">' +
      '<span class="viewer-step"><button type="button" class="btn btn-small" data-v="prev">&#8249; Prev item</button>' +
      '<span class="viewer-count"></span>' +
      '<button type="button" class="btn btn-small" data-v="next">Next item &#8250;</button></span>' +
      '<span class="viewer-mark"><button type="button" class="btn btn-small mark-ok" data-v="verified">&#10003; Verified</button>' +
      '<button type="button" class="btn btn-small mark-bad" data-v="issue">&#9888; Issue</button></span>' +
      '<button type="button" class="btn btn-small" data-v="close" title="Close (Esc)">Close &#10005;</button>' +
      '</div></div>' +
      '<div class="viewer-panes"><div class="pane"></div><div class="pane"></div></div>' +
      '</div>';
    document.body.appendChild(root);
    const [elA, elB] = root.querySelectorAll('.pane');
    const paneA = Pane(elA, getDocs), paneB = Pane(elB, getDocs);
    const title = root.querySelector('.viewer-title');
    const markBox = root.querySelector('.viewer-mark'), stepBox = root.querySelector('.viewer-step');
    let current = null;
    let lastFocus = null;

    function setMarkButtons(status) {
      markBox.querySelector('.mark-ok').classList.toggle('active', status === 'verified');
      markBox.querySelector('.mark-bad').classList.toggle('active', status === 'issue');
    }

    function close() {
      root.hidden = true;
      document.body.classList.remove('snc-viewer-open');
      if (onClose) onClose();
      if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
    }

    root.addEventListener('click', e => {
      if (e.target === root) return close();
      const b = e.target.closest('[data-v]');
      if (!b) return;
      const v = b.dataset.v;
      if (v === 'close') close();
      else if ((v === 'prev' || v === 'next') && current && current.step) onStep(current.step.index + (v === 'next' ? 1 : -1));
      else if ((v === 'verified' || v === 'issue') && current && current.mark) {
        const next = current.mark.status === v ? null : v;
        current.mark.status = next;
        setMarkButtons(next);
        onMark(current.mark.key, next);
      }
    });
    document.addEventListener('keydown', e => {
      if (root.hidden) return;
      if (e.key === 'Escape') close();
    });
    let rt;
    window.addEventListener('resize', () => {
      if (root.hidden) return;
      clearTimeout(rt);
      rt = setTimeout(() => { paneA.relayout(); if (!elB.hidden) paneB.relayout(); }, 200);
    });

    return {
      // spec: { title (html), left:{doc,page,hl}, right?:{doc,page,hl}, mark?:{key,status}, step?:{index,total} }
      open(spec) {
        current = spec;
        if (root.hidden) lastFocus = document.activeElement;
        title.innerHTML = spec.title || '';
        markBox.hidden = !spec.mark;
        stepBox.hidden = !spec.step;
        if (spec.mark) setMarkButtons(spec.mark.status);
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
          paneA.show(spec.left);
          if (spec.right) paneB.show(spec.right);
        });
      },
      close,
      reset() { cache.forEach(p => p.then(pdf => pdf.destroy()).catch(() => {})); cache.clear(); },
    };
  };
})();
