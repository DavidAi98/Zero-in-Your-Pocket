/* =====================================================================
   engine.js: a mini AlphaGo Zero
   ---------------------------------------------------------------------
   Same method as Silver et al. (2017), shrunk to fit on a phone:
     - Game: three-in-a-row on a 3x3 board with Go stones.
     - "Network": a lookup table (one entry per board position) that
       outputs a policy p (move probabilities) and a value v (who is
       winning). AlphaGo Zero used a deep residual neural network instead.
     - Search: Monte Carlo Tree Search (MCTS) with the PUCT rule, guided
       by p and v, with no random rollouts.
     - Learning: self-play, then train p toward the search's choice (pi)
       and v toward the game's result (z).

   Board format: an array of 9 numbers, indexes 0..8, row by row.
     0 = empty, 1 = black, -1 = white. Black always moves first.
   ===================================================================== */
var AGZ = (function () {

  /* ---------- 1. Game rules: the only knowledge the AI is given ---------- */

  // The 8 winning lines: 3 rows, 3 columns, 2 diagonals.
  const LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];

  // Whose turn it is: count the stones; an even count means black to move.
  function toMove(b) { let n = 0; for (const c of b) if (c) n++; return n % 2 ? -1 : 1; }

  // Result: 1 = black won, -1 = white won, 0 = draw, null = game still going.
  function winner(b) {
    for (const [a, c, d] of LINES) if (b[a] && b[a] === b[c] && b[a] === b[d]) return b[a];
    for (const c of b) if (!c) return null;
    return 0;
  }
  // The winning line itself (used by the page to make those stones glow).
  function winLine(b) {
    for (const L of LINES) if (b[L[0]] && b[L[0]] === b[L[1]] && b[L[0]] === b[L[2]]) return L;
    return null;
  }
  // Legal moves = every empty point.
  function legal(b) { const m = []; for (let i = 0; i < 9; i++) if (!b[i]) m.push(i); return m; }
  // A text key for a position, e.g. "b.w.b...." (used by the lookup table).
  function key(b) { let s = ''; for (const x of b) s += x === 1 ? 'b' : x === -1 ? 'w' : '.'; return s; }

  /* ---------- 2. Symmetry (data augmentation) ----------
     A square board looks the same after 4 rotations x 2 reflections = 8
     versions. AlphaGo Zero trained on all 8 versions of each Go position;
     we do the same, so every game teaches the AI 8 times as much. */
  const SYMS = [
    (r, c) => [r, c], (r, c) => [c, 2 - r], (r, c) => [2 - r, 2 - c], (r, c) => [2 - c, r],
    (r, c) => [r, 2 - c], (r, c) => [2 - r, c], (r, c) => [c, r], (r, c) => [2 - c, 2 - r]
  ].map(f => { const p = []; for (let i = 0; i < 9; i++) { const [r, c] = f((i / 3) | 0, i % 3); p[i] = r * 3 + c; } return p; });
  function permute(arr, p) { const o = new Array(9); for (let i = 0; i < 9; i++) o[p[i]] = arr[i]; return o; }

  /* ---------- 3. Random helpers for exploration noise ----------
     AlphaGo Zero mixes Dirichlet noise into the root's priors during
     self-play so it keeps trying new openings. Dirichlet samples are
     built from Gamma samples (Marsaglia-Tsang method). */
  function randn() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function gamma(a) {
    if (a < 1) return gamma(a + 1) * Math.pow(Math.random(), 1 / a);
    const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = randn(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = Math.random();
      if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  }
  function dirichlet(alpha, n) { const g = []; let s = 0; for (let i = 0; i < n; i++) { g.push(gamma(alpha)); s += g[i]; } return g.map(x => x / s); }

  /* ---------- 4. The "network": f(board) -> (p, v) ----------
     Each position stores 9 logits (raw move scores) and 1 value.
     Positions never seen before return a uniform policy and v = 0,
     i.e. "I have no idea", which is how an untrained network behaves. */
  class Net {
    constructor() { this.t = new Map(); this.lr = 0.3; } // lr = learning rate (step size)

    // Policy head: softmax over legal moves -> probabilities p that sum to 1.
    // Value head: v in [-1, 1], from the point of view of the player to move.
    predict(b) {
      const e = this.t.get(key(b)), m = legal(b), p = new Array(9).fill(0);
      let mx = -Infinity; for (const i of m) mx = Math.max(mx, e ? e.l[i] : 0);
      let s = 0; for (const i of m) { p[i] = Math.exp((e ? e.l[i] : 0) - mx); s += p[i]; }
      for (const i of m) p[i] /= s;
      return { p, v: e ? e.v : 0 };
    }

    /* KEY FEATURE: the training step.
       Loss (as in the paper): l = (z - v)^2 - pi . log(p)
         (z - v)^2      -> make the value predict the real winner z
         -pi . log(p)   -> make the policy copy the search's choice pi
       For a softmax, the gradient of the cross-entropy on the logits is
       (p - pi), so we move each logit by lr * (pi - p), and move v toward z.
       (The paper also adds weight decay c*||theta||^2; a table doesn't need it.)
       This is done for all 8 symmetric copies of the position. */
    train(b, pi, z) {
      let loss = 0;
      for (const P of SYMS) {
        const bb = permute(b, P), pp = permute(pi, P), k = key(bb);
        let e = this.t.get(k);
        if (!e) { e = { l: new Float32Array(9), v: 0 }; this.t.set(k, e); }
        const { p, v } = this.predict(bb);
        let ce = 0;
        for (const i of legal(bb)) {
          if (pp[i] > 0) ce -= pp[i] * Math.log(p[i] + 1e-9);
          e.l[i] += this.lr * (pp[i] - p[i]);          // policy update: p -> pi
        }
        loss += (z - v) ** 2 + ce;
        e.v += this.lr * (z - e.v);                      // value update:  v -> z
      }
      return loss / SYMS.length;
    }
    get size() { return this.t.size; } // how many positions the AI has learned about
  }

  /* ---------- 5. Monte Carlo Tree Search (MCTS) ----------
     KEY FEATURE: each edge (position, move) in the tree stores
       P = prior from the network, N = visit count,
       W = total value,           Q = W / N (average value).
     One simulation = Select -> Expand & Evaluate -> Back up. */
  function node(b) { return { b, term: winner(b), kids: null }; }

  class Search {
    constructor(net, b, opt) {
      opt = opt || {};
      this.net = net;
      this.c = opt.c || 1.5;        // c_puct: how much to favour exploration
      this.noise = !!opt.noise;     // add Dirichlet noise at the root (self-play only)
      this.root = node(b.slice());
      this.sims = 0;
    }

    // EXPAND & EVALUATE: ask the network once for (p, v) at a new leaf.
    // There are no random playouts; v replaces them (a big change from the original AlphaGo).
    expand(n) {
      const { p, v } = this.net.predict(n.b);
      n.kids = {};
      for (const i of legal(n.b)) n.kids[i] = { P: p[i], N: 0, W: 0, child: null };
      return v;
    }

    // One simulation. Returns a trace (the path taken) so the page can show it.
    step() {
      let n = this.root; const path = [];

      // SELECT: walk down the tree. At each node pick the move maximising
      //   Q + U,   U = c_puct * P * sqrt(total visits) / (1 + N)
      // Q = "this move has worked so far"; U = "the network likes it but we
      // haven't tried it much". U shrinks as a move gets visited.
      while (n.kids && n.term === null) {
        let tot = 0; for (const a in n.kids) tot += n.kids[a].N;
        const sq = Math.sqrt(Math.max(1, tot));
        let best = -Infinity, ba = null;
        for (const a in n.kids) {
          const e = n.kids[a], Q = e.N ? e.W / e.N : 0, U = this.c * e.P * sq / (1 + e.N);
          if (Q + U > best) { best = Q + U; ba = a; }
        }
        const e = n.kids[ba];
        if (!e.child) { const nb = n.b.slice(); nb[+ba] = toMove(n.b); e.child = node(nb); }
        path.push({ edge: e, move: +ba, player: toMove(n.b) });
        n = e.child;
      }

      // EVALUATE the leaf: a finished game has an exact result;
      // otherwise the network gives its estimate v.
      let v, kind;
      if (n.term !== null) { v = n.term === 0 ? 0 : -1; kind = 'end'; } // the player to move has lost
      else {
        v = this.expand(n); kind = 'net';
        // Root exploration noise: P = 0.75 * p + 0.25 * Dirichlet(0.3)
        if (n === this.root && this.noise) {
          const ks = Object.keys(n.kids), d = dirichlet(0.3, ks.length);
          ks.forEach((a, j) => { n.kids[a].P = 0.75 * n.kids[a].P + 0.25 * d[j]; });
        }
      }
      const leafV = v;

      // BACK UP: walk back to the root. Flip the sign at each level
      // (good for me = bad for my opponent), add 1 visit, add v to W.
      for (let j = path.length - 1; j >= 0; j--) { v = -v; path[j].edge.N++; path[j].edge.W += v; }

      this.sims++;
      return { path: path.map(x => ({ move: x.move, player: x.player })), v: leafV, kind, leaf: n.b };
    }

    run(k) { for (let i = 0; i < k; i++) this.step(); return this; }

    // Root statistics per point (null for occupied points); the page draws these.
    stats() {
      const out = []; const k = this.root.kids || {};
      for (let i = 0; i < 9; i++) { const e = k[i]; out.push(e ? { P: e.P, N: e.N, Q: e.N ? e.W / e.N : 0 } : null); }
      return out;
    }

    // KEY FEATURE: search probabilities pi, proportional to visit counts N.
    // These are better than the raw network's p, so they become the training target.
    pi() {
      const s = this.stats(); let t = 0; for (const e of s) if (e) t += e.N;
      return s.map(e => (e && t ? e.N / t : 0));
    }
  }

  // Pick the highest value; break ties at random.
  function argmaxRandom(arr, ok) {
    let best = -Infinity, c = [];
    for (let i = 0; i < arr.length; i++) { if (ok && !ok(i)) continue; if (arr[i] > best + 1e-12) { best = arr[i]; c = [i]; } else if (Math.abs(arr[i] - best) <= 1e-12) c.push(i); }
    return c[(Math.random() * c.length) | 0];
  }
  // Sample a move from a probability distribution.
  function sample(pi) { let r = Math.random(), a = 0; for (let i = 0; i < 9; i++) { r -= pi[i]; if (r <= 0 && pi[i] > 0) return i; if (pi[i] > 0) a = i; } return a; }

  // Choose a move to PLAY (not to train): with search if sims > 0,
  // or with the raw network alone if sims = 0.
  function chooseMove(net, b, sims) {
    if (!sims) { const { p } = net.predict(b); return { move: argmaxRandom(p, i => !b[i]), stats: null }; }
    const s = new Search(net, b).run(sims);
    return { move: argmaxRandom(s.pi(), i => !b[i]), stats: s.stats() };
  }

  /* ---------- 6. Self-play: the training loop ----------
     KEY FEATURE: the whole AlphaGo Zero idea in one function.
       1. Before every move, run MCTS (40 simulations here; 1,600 in the paper).
       2. Temperature: for the first 3 moves, SAMPLE from pi (tau = 1) for
          variety; after that, play the most-visited move (tau -> 0).
          (The paper samples for the first 30 moves of a much longer game.)
       3. Record (position s, search probabilities pi, player).
       4. When the game ends, label every position with z = +1 if that
          player went on to win, -1 if they lost, 0 for a draw.
       5. Train the network on every (s, pi, z). */
  const SAMPLE_MOVES = 3;
  function selfPlay(net, sims) {
    const b = new Array(9).fill(0), hist = [];
    while (winner(b) === null) {
      const s = new Search(net, b, { noise: true }).run(sims), pi = s.pi();
      const a = hist.length < SAMPLE_MOVES ? sample(pi) : argmaxRandom(pi, i => !b[i]);
      hist.push({ b: b.slice(), pi, player: toMove(b), move: a, stats: s.stats() });
      b[a] = toMove(b);
    }
    const w = winner(b); let loss = 0;
    for (const h of hist) { h.z = w === 0 ? 0 : h.player === w ? 1 : -1; loss += net.train(h.b, h.pi, h.z); }
    return { w, hist, final: b.slice(), loss: loss / hist.length };
  }

  /* ---------- 7. A perfect player, used ONLY to measure progress ----------
     Minimax solves this small game exactly. It is never used for training:
     the AI learns only from its own self-play, as in the paper. */
  const memo = new Map();
  function solve(b) {
    const k = key(b); if (memo.has(k)) return memo.get(k);
    const w = winner(b); let r;
    if (w !== null) r = { v: w === 0 ? 0 : -1, best: [] };
    else {
      const pl = toMove(b); let bv = -2, bm = [];
      for (const i of legal(b)) { b[i] = pl; const x = -solve(b).v; b[i] = 0; if (x > bv) { bv = x; bm = [i]; } else if (x === bv) bm.push(i); }
      r = { v: bv, best: bm };
    }
    memo.set(k, r); return r;
  }
  solve(new Array(9).fill(0)); // pre-solve every position once at start-up

  // Test: the network alone (sims = 0) plays `games` games against the perfect
  // player, half as black and half as white. Returns the share of games LOST.
  // The page shows 1 - this as "perfect games" (a draw is perfect play here).
  function vsPerfect(net, sims, games) {
    let lost = 0;
    for (let g = 0; g < games; g++) {
      const b = new Array(9).fill(0), me = g % 2 ? -1 : 1;
      while (winner(b) === null) {
        const pl = toMove(b);
        if (pl === me) b[chooseMove(net, b, sims).move] = pl;
        else { const bs = memo.get(key(b)).best; b[bs[(Math.random() * bs.length) | 0]] = pl; }
      }
      if (winner(b) === -me) lost++;
    }
    return lost / games;
  }

  // Everything the page (app.js) uses.
  return { LINES, toMove, winner, winLine, legal, key, Net, Search, selfPlay, chooseMove, vsPerfect, solve };
})();

// Lets you also load this file in Node.js for testing: const AGZ = require('./engine.js')
if (typeof module !== 'undefined') module.exports = AGZ;
