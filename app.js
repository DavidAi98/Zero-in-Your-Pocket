/* =====================================================================
   app.js: the page. Connects engine.js (the AI) to the screen.
   Sections: board drawing, loop diagram, training + chart,
   search stepper, play against the AI, results bars.
   Everything sits inside one { } block so no names leak into the page.
   ===================================================================== */
{
  const $ = id => document.getElementById(id);
  const NAMES = ['top-left', 'top', 'top-right', 'left', 'centre', 'right', 'bottom-left', 'bottom', 'bottom-right'];
  const who = p => (p === 1 ? 'Black' : 'White');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const wait = ms => new Promise(r => setTimeout(r, reduced ? Math.min(ms, 120) : ms));

  /* ---------- board rendering ----------
     makeBoard builds the 9 points once; drawBoard repaints them.
     Options for drawBoard: b (stones), N (visit counts -> number + glow),
     Q (average result), P (prior -> purple bar), ghosts (dashed stones
     showing a search path), order (move numbers), last (drop animation). */
  function makeBoard(el, onClick) {
    el.innerHTML = '';
    for (let i = 0; i < 9; i++) {
      const c = document.createElement(onClick ? 'button' : 'div');
      c.className = 'pt r' + ((i / 3) | 0) + ' c' + (i % 3);
      if (onClick) { c.type = 'button'; c.addEventListener('click', () => onClick(i)); }
      el.appendChild(c);
    }
  }
  function drawBoard(el, o) {
    const b = o.b, win = AGZ.winLine(b) || [], max = o.N ? Math.max(1, ...o.N.map(x => x || 0)) : 1;
    [...el.children].forEach((c, i) => {
      let h = '';
      const n = o.N && o.N[i], heat = n ? 0.12 + 0.5 * n / max : 0;
      h += '<span class="heat" style="opacity:' + heat.toFixed(2) + '"></span>';
      if (o.N && !b[i] && o.N[i] != null) h += '<span class="num">' + o.N[i] + '</span>';
      if (o.Q && !b[i] && o.Q[i] != null && o.N[i]) h += '<span class="q">' + (o.Q[i] >= 0 ? '+' : '−') + Math.abs(o.Q[i]).toFixed(2) + '</span>';
      if (o.P && !b[i] && o.P[i] != null) h += '<span class="pbar"><i style="width:' + Math.round(o.P[i] * 100) + '%"></i></span>';
      const g = o.ghosts && o.ghosts.find(x => x.i === i);
      if (b[i]) {
        const lab = o.order && o.order[i] ? o.order[i] : '';
        h += '<div class="stone ' + (b[i] === 1 ? 'bk' : 'wh') + (win.includes(i) ? ' win' : '') + '">' + lab + '</div>';
      } else if (g) {
        h += '<div class="stone ghostst ' + (g.p === 1 ? 'bk' : 'wh') + '">' + g.label + '</div>';
      }
      c.innerHTML = h;
      c.classList.toggle('lastm', o.last === i);
      const st = b[i] ? who(b[i]).toLowerCase() + ' stone' : 'empty';
      c.setAttribute('aria-label', NAMES[i] + ', ' + st + (o.N && o.N[i] ? ', ' + o.N[i] + ' visits' : ''));
      if (c.tagName === 'BUTTON') c.disabled = !!b[i] || !!o.lock;
    });
  }

  /* ---------- the loop diagram: lights up the active stage (net/search/play/learn) ---------- */
  const steps = [...document.querySelectorAll('.step')];
  function setLoop(s) { steps.forEach(x => x.classList.toggle('on', x.dataset.s === s)); }

  /* ---------- training ----------
     KEY FEATURE: every self-play game trains the network (inside AGZ.selfPlay).
     Every 10 games we test the network alone against a perfect player
     (80 games) and add a point to the chart. */
  let net = new AGZ.Net(), games = 0, curve = [], busy = false;
  const TEST = 80, measure = () => AGZ.vsPerfect(net, 0, TEST);
  const trainBoard = $('trainBoard'); makeBoard(trainBoard);

  function tiles() {
    $('tGames').textContent = games.toLocaleString();
    const l = curve.length ? curve[curve.length - 1].l : null;
    if (l != null) {
      const ok = Math.round((1 - l) * TEST);
      $('tPerf').textContent = Math.round(ok / TEST * 100) + '%';
      $('tFrac').textContent = ok + ' of ' + TEST + ' test games vs a perfect player';
      $('tBar').style.width = (ok / TEST * 100) + '%';
    }
  }
  function orderOf(hist) { const o = {}; hist.forEach((h, k) => { o[h.move] = k + 1; }); return o; }
  function lock(on) { busy = on; ['watchBtn', 'trainBtn', 'resetBtn'].forEach(id => { $(id).disabled = on; }); }

  // Watch one self-play game slowly: show the search's visits, then place the stone.
  async function watchOne() {
    if (busy) return; lock(true);
    const r = AGZ.selfPlay(net, 40);
    const b = new Array(9).fill(0), order = {};
    for (let k = 0; k < r.hist.length; k++) {
      const h = r.hist[k];
      setLoop('search');
      drawBoard(trainBoard, { b, N: h.stats.map(s => (s ? s.N : null)), order });
      $('trainStatus').textContent = who(h.player) + ': 40 searches → ' + NAMES[h.move];
      await wait(900);
      setLoop('play');
      b[h.move] = h.player; order[h.move] = k + 1;
      drawBoard(trainBoard, { b, order, last: h.move });
      await wait(450);
    }
    setLoop('learn');
    const res = r.w === 0 ? 'Draw (z = 0).' : who(r.w) + ' wins (z = ±1).';
    $('trainStatus').textContent = res + ' Learned.';
    games++; curve.push({ g: games, l: measure() }); tiles(); drawChart();
    await wait(1400); setLoop('net');
    lock(false); resetSearch();
  }

  // Train n games quickly, 3 games per frame, so the page stays responsive.
  function trainBatch(n) {
    if (busy) return; lock(true);
    const target = games + n; let frame = 0, last = null;
    const cycle = ['net', 'search', 'play', 'learn'];
    const tick = () => {
      setLoop(cycle[frame++ % 4]);
      for (let k = 0; k < 3 && games < target; k++) {
        last = AGZ.selfPlay(net, 40); games++;
        if (games % 10 === 0) curve.push({ g: games, l: measure() });
      }
      if (last) drawBoard(trainBoard, { b: last.final, order: orderOf(last.hist) });
      tiles(); drawChart();
      $('trainStatus').textContent = 'Training… ' + games.toLocaleString() + ' / ' + target.toLocaleString();
      if (games < target) setTimeout(tick, reduced ? 0 : 40);
      else {
        setLoop('net'); lock(false); resetSearch();
        const l = curve[curve.length - 1].l;
        $('trainStatus').textContent = l === 0
          ? 'Done. It no longer loses. Try to beat it below.'
          : 'Done. Train more to keep improving.';
      }
    };
    tick();
  }
  function resetAll() {
    if (busy) return;
    net = new AGZ.Net(); games = 0; curve = [{ g: 0, l: measure() }];
    drawBoard(trainBoard, { b: new Array(9).fill(0) }); setLoop(null); tiles(); drawChart(); resetSearch(); newGame();
    $('trainStatus').textContent = 'Untrained. Watch one game or train 100.';
  }
  $('watchBtn').addEventListener('click', watchOne);
  $('trainBtn').addEventListener('click', () => trainBatch(100));
  $('resetBtn').addEventListener('click', resetAll);

  /* ---------- chart: perfect-game % vs games played (redrawn on resize) ---------- */
  const chartSvg = document.querySelector('#chart svg');
  function drawChart() {
    const W = chartSvg.clientWidth, H = chartSvg.clientHeight; if (!W || !H) return;
    const L = 44, R = 14, T = 14, B = 30, gx = Math.max(100, games), w = W - L - R, h = H - T - B;
    const X = g => L + w * g / gx, Y = l => T + h * (1 - l);
    let s = '';
    for (const v of [0, 0.5, 1]) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="#1d2129"/><text x="' + (L - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + v * 100 + '%</text>';
    s += '<text x="' + L + '" y="' + (H - 8) + '">0</text><text x="' + (W - R) + '" y="' + (H - 8) + '" text-anchor="end">' + gx + ' games</text>';
    if (curve.length) {
      const pts = curve.map(p => X(p.g).toFixed(1) + ',' + Y(1 - p.l).toFixed(1)).join(' ');
      s += '<polyline points="' + pts + '" fill="none" stroke="#5cc8ff" stroke-opacity=".25" stroke-width="7" stroke-linejoin="round"/>';
      s += '<polyline points="' + pts + '" fill="none" stroke="#5cc8ff" stroke-width="2" stroke-linejoin="round"/>';
      const p = curve[curve.length - 1];
      s += '<circle cx="' + X(p.g) + '" cy="' + Y(1 - p.l) + '" r="4" fill="#5cc8ff"/>';
    }
    chartSvg.innerHTML = s;
  }
  const chartRO = new ResizeObserver(drawChart); chartRO.observe(chartSvg);

  /* ---------- search stepper ----------
     KEY FEATURE: runs MCTS one simulation at a time on a fixed position
     (black must block white's line at the right edge) and shows P, N, Q
     on each point, plus the path the simulation took. */
  const B0 = [1, 0, -1, 0, 1, 0, 0, 0, -1];
  const searchBoard = $('searchBoard'); makeBoard(searchBoard);
  const traceLis = [...document.querySelectorAll('#trace li')];
  let search, lastTrace = null, stepping = false;
  function setTrace(t) { traceLis.forEach(li => li.classList.toggle('on', li.dataset.t === t)); }
  function drawSearch() {
    const st = search.stats(), ghosts = [];
    if (lastTrace) lastTrace.path.forEach((m, k) => { if (!B0[m.move]) ghosts.push({ i: m.move, p: m.player, label: k + 1 }); });
    drawBoard(searchBoard, {
      b: B0, ghosts,
      N: st.map(s => (s ? s.N : null)), Q: st.map(s => (s ? s.Q : null)), P: st.map(s => (s ? s.P : null))
    });
  }
  function describe(t) {
    const n = search.sims;
    if (!t.path.length) return 'Sim 1: the network sets priors p (purple bars).';
    const route = t.path.map(m => who(m.player).toLowerCase() + ' ' + NAMES[m.move]).join(' → ');
    const ev = t.kind === 'end'
      ? 'game over, exact result.'
      : 'v = ' + (t.v >= 0 ? '+' : '−') + Math.abs(t.v).toFixed(2) + '.';
    return 'Sim ' + n + ': ' + route + ' → ' + ev;
  }
  async function stepSearch(k) {
    if (stepping) return; stepping = true;
    if (k === 1) {
      setTrace('sel'); await wait(260);
      lastTrace = search.step();
      setTrace('eval'); drawSearch(); await wait(320);
      setTrace('back');
    } else {
      for (let i = 0; i < k; i++) lastTrace = search.step();
      setTrace('back'); drawSearch();
    }
    drawSearch();
    $('searchStatus').textContent = describe(lastTrace);
    stepping = false;
  }
  function resetSearch() {
    search = new AGZ.Search(net, B0); lastTrace = null; setTrace(null);
    drawBoard(searchBoard, { b: B0 });
    $('searchStatus').textContent = 'Press +1 to run a simulation.';
  }
  $('s1Btn').addEventListener('click', () => stepSearch(1));
  $('s10Btn').addEventListener('click', () => stepSearch(10));
  $('sResetBtn').addEventListener('click', resetSearch);

  /* ---------- play vs AI ----------
     The slider sets searches per move. 0 = the raw network only, which
     shows how much search adds to the network. */
  const playBoard = $('playBoard');
  let pb, human = 1, over = false, thinking = false, tally = { you: 0, d: 0, ai: 0 }, lastN = null;
  makeBoard(playBoard, humanMove);
  const sims = () => +$('sims').value;
  $('sims').addEventListener('input', () => { $('simsOut').textContent = sims(); });
  function drawPlay(last) { drawBoard(playBoard, { b: pb, N: lastN, last, lock: over || thinking || AGZ.toMove(pb) !== human }); }
  function finish() {
    const w = AGZ.winner(pb); if (w === null) return false;
    over = true;
    if (w === 0) { tally.d++; $('playStatus').textContent = 'Draw. The best anyone can do.'; }
    else if (w === human) { tally.you++; $('playStatus').textContent = 'You win! Train it more and retry.'; }
    else { tally.ai++; $('playStatus').textContent = 'The AI wins.'; }
    $('tally').textContent = 'you ' + tally.you + ' · draws ' + tally.d + ' · AI ' + tally.ai;
    return true;
  }
  async function aiMove() {
    thinking = true; drawPlay();
    $('playStatus').textContent = 'Thinking…';
    await wait(350);
    const k = sims(), r = AGZ.chooseMove(net, pb, k);
    lastN = r.stats ? r.stats.map(s => (s ? s.N : null)) : null;
    pb[r.move] = -human; thinking = false;
    drawPlay(r.move);
    if (!finish()) $('playStatus').textContent = (k ? k + ' searches → ' + NAMES[r.move] + ' (numbers = visits).' : 'Network only → ' + NAMES[r.move] + '.') + ' Your move.';
  }
  function humanMove(i) {
    if (over || thinking || pb[i] || AGZ.toMove(pb) !== human) return;
    pb[i] = human; lastN = null; drawPlay(i);
    if (!finish()) aiMove();
  }
  function newGame() {
    pb = new Array(9).fill(0); over = false; lastN = null; thinking = false;
    drawPlay();
    $('playStatus').textContent = 'Your move. You are ' + who(human).toLowerCase() + '.';
    if (human === -1) aiMove();
  }
  function setColour(c) {
    human = c; $('pBlack').setAttribute('aria-pressed', c === 1); $('pWhite').setAttribute('aria-pressed', c === -1); newGame();
  }
  $('pBlack').addEventListener('click', () => setColour(1));
  $('pWhite').addEventListener('click', () => setColour(-1));
  $('newGame').addEventListener('click', newGame);

  /* ---------- results bars ----------
     Elo ratings from Silver et al. (2017). Check against the paper before printing. */
  const ELO = [
    { n: 'AlphaGo Zero', s: '40 days', v: 5185, cls: 'zero' },
    { n: 'AlphaGo Master', s: '', v: 4858 },
    { n: 'AlphaGo Lee', s: 'beat Lee Sedol', v: 3739 },
    { n: 'AlphaGo Fan', s: '', v: 3144 },
    { n: 'Zero, no search', s: 'network only', v: 3055, cls: 'raw' }
  ];
  const bars = $('bars');
  bars.innerHTML = ELO.map(e => '<div class="bar ' + (e.cls || '') + '"><div class="nm"><span><b>' + e.n + '</b>' + (e.s ? ' · ' + e.s : '') + '</span><b>' + e.v.toLocaleString() + '</b></div><div class="track"><div class="fill" data-w="' + (e.v / 5400 * 100).toFixed(1) + '"></div></div></div>').join('');
  const fillBars = () => bars.querySelectorAll('.fill').forEach(f => { f.style.width = f.dataset.w + '%'; });
  if ('IntersectionObserver' in window) {
    const barIO = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { fillBars(); barIO.disconnect(); } }, { threshold: 0.3 });
    barIO.observe(bars);
  } else fillBars();

  /* ---------- start: measure the untrained AI and draw everything ---------- */
  curve = [{ g: 0, l: measure() }];
  drawBoard(trainBoard, { b: new Array(9).fill(0) });
  tiles(); drawChart(); resetSearch(); newGame();
}
