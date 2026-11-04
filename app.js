// EyeChess: scan a real board with the camera, suggest the best move.
// Idea: pixels only tell us empty / white / black per square. Piece types come
// from game history: we find the legal move(s) whose result matches the scan.

// ---------- pure logic (also runs in node for tests) ----------
const FILES = 'abcdefgh';
const sqName = i => FILES[i % 8] + (8 - Math.floor(i / 8)); // i=0 is a8, i=63 is h1
const isLight = i => ((i % 8) + Math.floor(i / 8)) % 2 === 0;
const BOARD_CORNERS = [[0, 0], [8, 0], [8, 8], [0, 8]]; // a8, h8, h1, a1 in board units

// 4-point homography src -> dst, solved with Gaussian elimination.
function homography(src, dst) {
  const A = [], b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
    for (let r = 0; r < 8; r++) if (r !== c) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  return [...b.map((v, i) => v / A[i][i]), 1];
}

function project(h, x, y) {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

// Per square: [mean luminance, std dev] over the centre 50% of the square.
function squareFeatures(img, h) {
  const out = [];
  for (let i = 0; i < 64; i++) {
    const f = i % 8, r = Math.floor(i / 8);
    let s = 0, s2 = 0, n = 0;
    for (let a = 0; a < 10; a++) for (let c = 0; c < 10; c++) {
      const [x, y] = project(h, f + 0.25 + a * 0.05, r + 0.25 + c * 0.05);
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= img.width || yi >= img.height) continue;
      const p = (yi * img.width + xi) * 4;
      const l = 0.299 * img.data[p] + 0.587 * img.data[p + 1] + 0.114 * img.data[p + 2];
      s += l; s2 += l * l; n++;
    }
    const m = n ? s / n : 0;
    out.push([m, n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0]);
  }
  return out;
}

// Centroid per (class, square colour), learned from a known position.
function buildRefs(feats, grid) {
  const acc = {};
  feats.forEach((ft, i) => {
    const k = grid[i] + isLight(i);
    const a = (acc[k] ||= [0, 0, 0]);
    a[0] += ft[0]; a[1] += ft[1]; a[2]++;
  });
  const refs = {};
  for (const k in acc) refs[k] = [acc[k][0] / acc[k][2], acc[k][1] / acc[k][2]];
  return refs;
}

// ponytail: nearest-centroid on brightness+texture; swap for a tiny CNN if lighting breaks it.
function classify(feats, refs) {
  return feats.map((ft, i) => {
    let best = '.', bd = Infinity;
    for (const c of '.wb') {
      const ref = refs[c + isLight(i)] || refs[c + !isLight(i)];
      if (!ref) continue;
      const d = (ft[0] - ref[0]) ** 2 + 9 * (ft[1] - ref[1]) ** 2; // texture weighted 3x: highlights shift brightness, not texture
      if (d < bd) { bd = d; best = c; }
    }
    return best;
  }).join('');
}

function gridOf(chess) {
  let g = '';
  for (let i = 0; i < 64; i++) { const p = chess.get(sqName(i)); g += p ? p.color : '.'; }
  return g;
}

function diff(a, b) { let n = 0; for (let i = 0; i < 64; i++) if (a[i] !== b[i]) n++; return n; }

// Find the shortest run of legal moves (up to 4, for fast play / premoves) that best explains the scanned grid.
// Pruning: one move changes at most 4 squares (castling), so with `left` moves to go a line that is still
// more than 4*left squares away can never beat the best found so far.
function findMoves(chess, scanned, maxDepth = 4) {
  let best = { moves: [], diff: diff(gridOf(chess), scanned) };
  if (best.diff === 0) return best;
  const t0 = Date.now(), path = [];
  const dfs = left => {
    for (const m of chess.moves({ verbose: true })) {
      chess.move(m); path.push(m);
      const d = diff(gridOf(chess), scanned);
      if (d < best.diff) best = { moves: path.slice(), diff: d };
      if (left > 1 && d - 4 * (left - 1) < best.diff && Date.now() - t0 < 400) dfs(left - 1);
      path.pop(); chess.undo();
    }
  };
  for (let depth = 1; depth <= maxDepth; depth++) {
    if (depth > 1 && best.diff < 2) break; // a 1-off is scan noise, not another move
    if (depth > 2 && best.diff > 10) break; // that many squares off is a misread, not fast play
    const before = best;
    best = { moves: [], diff: before.diff }; // only a strictly better, longer line replaces a shorter one
    dfs(depth);
    if (!best.moves.length || best.diff >= before.diff) best = before;
    if (best.diff === 0) break;
  }
  return best;
}

// Did the screen go back to an earlier position (takeback)? Returns {n plies undone, diff}; leaves chess unchanged.
function findTakeback(chess, scanned, max = 4) {
  const undone = [];
  let best = { n: 0, diff: Infinity };
  for (let n = 1; n <= max; n++) {
    const m = chess.undo();
    if (!m) break;
    undone.push(m);
    const d = diff(gridOf(chess), scanned);
    if (d < best.diff) best = { n, diff: d };
  }
  while (undone.length) chess.move(undone.pop());
  return best;
}

// ---------- full position reading (piece shapes), used to resync ----------
// Each square -> 16x16 patch. Background = median of the patch border, so it works on any square
// colour and on highlighted squares. Patch vector = [mask (piece pixel?) , luminance of piece pixels].
const PN = 16;
const rgbDist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
function squarePatches(img, h) {
  const out = [];
  for (let i = 0; i < 64; i++) {
    const f = i % 8, r = Math.floor(i / 8), px = [], ring = [];
    for (let a = 0; a < PN; a++) for (let c = 0; c < PN; c++) {
      const [x, y] = project(h, f + (c + 0.5) / PN, r + (a + 0.5) / PN);
      const xi = Math.min(img.width - 1, Math.max(0, Math.round(x))), yi = Math.min(img.height - 1, Math.max(0, Math.round(y)));
      const p = (yi * img.width + xi) * 4, q = [img.data[p], img.data[p + 1], img.data[p + 2]];
      px.push(q);
      if (a === 0 || c === 0 || a === PN - 1 || c === PN - 1) ring.push(q);
    }
    const med = j => ring.map(q => q[j]).sort((x, y) => x - y)[ring.length >> 1];
    const bg = [med(0), med(1), med(2)], v = new Array(2 * PN * PN).fill(0);
    px.forEach((q, k) => {
      if (rgbDist(q, bg) > 30) { v[k] = 255; v[PN * PN + k] = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2]; }
    });
    out.push({ v, bg });
  }
  return out;
}

// Average patch per piece letter ('.' = empty) from a known position, plus normal light/dark square colours.
function buildPieceModel(patches, pieces) {
  const acc = {}, bgs = { l: [], d: [] };
  patches.forEach((p, i) => {
    const a = (acc[pieces[i]] ||= { v: new Array(p.v.length).fill(0), n: 0 });
    p.v.forEach((x, j) => { a.v[j] += x; }); a.n++;
    if (pieces[i] === '.') bgs[isLight(i) ? 'l' : 'd'].push(p.bg);
  });
  const avg = list => [0, 1, 2].map(j => list.reduce((s, q) => s + q[j], 0) / (list.length || 1));
  const tpl = {};
  for (const k in acc) tpl[k] = acc[k].v.map(x => Math.round(x / acc[k].n));
  return { tpl, bgL: avg(bgs.l), bgD: avg(bgs.d) };
}

// ponytail: nearest template per square; piece types never seen at calibration can't be read (e.g. a 2nd queen reads as Q only if a Q existed).
// Result also carries .unsure: squares where the runner-up shape was nearly as close (not a clear read).
function readPieces(patches, model) {
  const unsure = new Set();
  const out = patches.map((p, i) => {
    let best = '.', bd = Infinity, second = Infinity;
    for (const k in model.tpl) {
      const t = model.tpl[k]; let d = 0;
      for (let j = 0; j < t.length; j++) d += (p.v[j] - t[j]) ** 2;
      if (d < bd) { second = bd; bd = d; best = k; } else if (d < second) second = d;
    }
    if (bd > 0.7 * second) unsure.add(i);
    return best;
  });
  out.unsure = unsure;
  return out;
}

// Colour of the side that just moved, from the site's last-move highlight (null if none seen).
function lastMover(patches, pieces, model) {
  let king = null;
  for (let i = 0; i < 64; i++) {
    const p = pieces[i];
    if (p === '.' || rgbDist(patches[i].bg, isLight(i) ? model.bgL : model.bgD) < 25) continue;
    const col = p === p.toUpperCase() ? 'w' : 'b';
    if (p.toLowerCase() !== 'k') return col;
    king = col; // king is the only highlighted piece: it just moved (in check, the checker's square wins above)
  }
  return king;
}

// Resync helper: a square that still holds the same colour as before keeps its known piece type;
// only changed squares take the (less reliable) shape reading.
function mergeRead(oldPieces, read) {
  const col = p => p === '.' ? '.' : p === p.toUpperCase() ? 'w' : 'b';
  return read.map((p, i) => oldPieces[i] !== '.' && col(oldPieces[i]) === col(p) ? oldPieces[i] : p);
}

// Could this position occur in a real game? Rejects misreads before they reach the engine.
function plausible(pieces, turn) {
  const n = p => pieces.filter(x => x === p).length;
  if (n('K') !== 1 || n('k') !== 1) return false;
  if ([...pieces.slice(0, 8), ...pieces.slice(56)].some(p => p === 'P' || p === 'p')) return false; // pawn on back rank
  for (const up of [true, false]) {
    const c = p => n(up ? p : p.toLowerCase()), pawns = c('P');
    const extra = Math.max(0, c('Q') - 1) + Math.max(0, c('R') - 2) + Math.max(0, c('B') - 2) + Math.max(0, c('N') - 2);
    if (pawns > 8 || extra > 8 - pawns) return false; // extra pieces can only come from promoted pawns
  }
  const other = new Chess();
  if (!other.load(fenFromGrid(pieces, turn === 'w' ? 'b' : 'w'))) return false;
  return !other.in_check(); // the side that just moved can't be left in check
}

// ---------- engine output ----------
// "info depth 12 multipv 2 score cp -35 ... pv e7e5 g1f3" -> { k: 2, score: { type: 'cp', v: -35 }, uci: 'e7e5' }
function parseInfo(line) {
  const k = line.match(/ multipv (\d+)/), s = line.match(/ score (cp|mate) (-?\d+)/), pv = line.match(/ pv (.+)$/);
  if (!s || !pv) return null;
  const moves = pv[1].trim().split(/\s+/);
  return { k: k ? +k[1] : 1, score: { type: s[1], v: +s[2] }, uci: moves[0], pv: moves };
}

// Engine scores are for the side to move; show them from White's view: +1.2, -0.4, M3, -M2.
function formatScore(score, turn) {
  const v = turn === 'w' ? score.v : -score.v;
  if (score.type === 'mate') return v > 0 ? `M${v}` : `-M${-v}`;
  const x = v / 100;
  return (x >= 0 ? '+' : '') + x.toFixed(1);
}

const PIECE_NAMES = { N: 'Knight', B: 'Bishop', R: 'Rook', Q: 'Queen', K: 'King' };
function sanToSpeech(san) {
  const tail = san.includes('#') ? ', checkmate' : san.includes('+') ? ', check' : '';
  if (san.startsWith('O-O-O')) return 'Castle queenside' + tail;
  if (san.startsWith('O-O')) return 'Castle kingside' + tail;
  const m = san.match(/^([NBRQK])?[a-h]?[1-8]?(x)?([a-h][1-8])(?:=([NBRQ]))?/);
  if (!m) return san;
  return `${m[1] ? PIECE_NAMES[m[1]] : 'Pawn'} ${m[2] ? 'takes' : 'to'} ${m[3]}` + (m[4] ? `, promotes to ${PIECE_NAMES[m[4]]}` : '') + tail;
}

// ---------- game review ----------
// Engine score (side to move's view) -> centipawns, mates capped at ±10000.
const toCp = s => s.type === 'mate' ? (s.v > 0 ? 10000 : -10000) : s.v;
// Winning chance 0..100 for the side whose view cp is (same curve lichess uses).
const winPct = cp => 50 + 50 * (2 / (1 + Math.exp(-0.00368208 * cp)) - 1);

// before: eval of the position the move was played from {cp, best: uci, second: cp of 2nd-best move};
// after: eval of the next position (cp from the opponent's view, since they are to move).
// extra: { prevBad: opponent's last move was a mistake/blunder, sacrifice: move gives up material }.
function reviewMove(before, after, playedUci, extra = {}) {
  // engine's own choice = no loss (separate short searches wobble a little between positions)
  const loss = playedUci === before.best ? 0 : Math.max(0, winPct(before.cp) - winPct(-after.cp));
  const accuracy = Math.min(100, Math.max(0, 103.1668 * Math.exp(-0.04354 * loss) - 3.1669));
  let label = playedUci === before.best ? 'Best' : loss < 2 ? 'Excellent' : loss < 5 ? 'Good'
    : loss < 10 ? 'Inaccuracy' : loss < 20 ? 'Mistake' : 'Blunder';
  // ponytail: Brilliant/Great/Miss are heuristics on top of short engine searches, not ground truth.
  if (loss < 2 && extra.sacrifice && winPct(before.cp) < 90 && winPct(-after.cp) >= 45) label = 'Brilliant';
  else if (label === 'Best' && before.second != null && winPct(before.cp) - winPct(before.second) >= 15) label = 'Great';
  else if (loss >= 5 && extra.prevBad && winPct(before.cp) >= 65) label = 'Miss';
  return { loss, accuracy, label };
}

// Opening book: openings.tsv lines are "position <tab> ECO Name", name empty for a book position on the
// way to a named one. Made from the lichess chess-openings list (CC0) by playing every line once, because
// replaying 3,865 lines with chess.js takes ~8s. Keyed by position, so transpositions still get the name.
const posKey = fen => fen.split(' ').slice(0, 4).join(' ');
const openingBook = tsv => new Map(tsv.split('\n').filter(l => l.includes('\t')).map(l => { const [k, n] = l.split('\t'); return [k, n || null]; }));

// Material for the side to move minus the opponent (P1 N3 B3 R5 Q9), in a position given by FEN.
function materialBalance(fen, side) {
  const V = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  let s = 0;
  for (const ch of fen.split(' ')[0]) {
    const v = V[ch.toLowerCase()];
    if (v != null) s += (ch === ch.toUpperCase()) === (side === 'w') ? v : -v;
  }
  return s;
}

// Does the move give up material for good? Play it, the opponent's best reply, and our follow-up
// (so a plain exchange that we win back straight away doesn't count).
function isSacrifice(fen, playedUci, replyUci, followUci) {
  const side = fen.split(' ')[1], c = new Chess(fen), before = materialBalance(fen, side);
  const play = u => u && c.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] || 'q' });
  if (!play(playedUci)) return false;
  play(replyUci) && play(followUci);
  return materialBalance(c.fen(), side) <= before - 2;
}

// ---------- PGN ----------
// PGN text (chess.com, lichess, …) -> moves in the shape EyeChess records, or null if unreadable.
function gameFromPgn(text) {
  const c = new Chess();
  if (!text || !c.load_pgn(text.trim(), { sloppy: true })) return null;
  const headers = c.header(), replay = new Chess(headers.FEN || undefined);
  const moves = c.history({ verbose: true }).map(m => {
    const fen = replay.fen(); replay.move(m.san);
    return { fen, san: m.san, uci: m.from + m.to + (m.promotion || ''), color: m.color };
  });
  return { moves, headers, result: headers.Result || '*' };
}

// Recorded game -> PGN. complete=false when EyeChess lost track mid-game (PGN then stops at the gap).
function gameToPgn(g) {
  const c = new Chess(g.moves[0].fen), key = f => f.split(' ').slice(0, 2).join(' ');
  let complete = true;
  for (const m of g.moves) if (key(m.fen) !== key(c.fen()) || !c.move(m.san)) { complete = false; break; }
  const d = new Date(g.date || Date.now()), pad = n => String(n).padStart(2, '0');
  c.header('Event', g.title || 'EyeChess game', 'Date', `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())}`,
    'Result', complete ? g.result || '*' : '*');
  return { pgn: c.pgn({ max_width: 80, newline_char: '\n' }), complete };
}

// UCI line -> SAN list from a position (stops at the first move that doesn't fit).
function pvSan(fen, ucis, n) {
  const c = new Chess(fen), out = [];
  for (const u of (ucis || []).slice(0, n)) {
    const m = c.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] || 'q' });
    if (!m) break;
    out.push(m.san);
  }
  return out;
}

// 64 chars: '.' or FEN letter
function piecesOf(chess) {
  return [...Array(64)].map((_, i) => {
    const p = chess.get(sqName(i));
    return p ? (p.color === 'w' ? p.type.toUpperCase() : p.type) : '.';
  });
}

function fenFromGrid(pieces, turn) {
  const rows = [];
  for (let r = 0; r < 8; r++) {
    let row = '', e = 0;
    for (let f = 0; f < 8; f++) {
      const p = pieces[r * 8 + f];
      if (p === '.') e++; else { if (e) row += e; row += p; e = 0; }
    }
    rows.push(row + (e || ''));
  }
  let castle = '';
  if (pieces[60] === 'K' && pieces[63] === 'R') castle += 'K';
  if (pieces[60] === 'K' && pieces[56] === 'R') castle += 'Q';
  if (pieces[4] === 'k' && pieces[7] === 'r') castle += 'k';
  if (pieces[4] === 'k' && pieces[0] === 'r') castle += 'q';
  return `${rows.join('/')} ${turn} ${castle || '-'} - 0 1`;
}

// ---------- auto board detection ----------
function toGray(img) {
  const g = new Float32Array(img.width * img.height);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) g[i] = 0.299 * img.data[p] + 0.587 * img.data[p + 1] + 0.114 * img.data[p + 2];
  return g;
}

function integral(g, W, H) {
  const I = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) { row += g[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; }
  }
  return I;
}

// Checkerboard correlation: +mean of "a8-colour" squares, -mean of the others, over an N x N grid of boxes.
function gridScore(I, W, x0, y0, sx, sy, N = 8) {
  const W1 = W + 1; let S = 0;
  for (let r = 0; r < N; r++) {
    const ya = Math.round(y0 + r * sy), yb = Math.round(y0 + (r + 1) * sy);
    for (let f = 0; f < N; f++) {
      const xa = Math.round(x0 + f * sx), xb = Math.round(x0 + (f + 1) * sx);
      const sum = I[yb * W1 + xb] - I[ya * W1 + xb] - I[yb * W1 + xa] + I[ya * W1 + xa];
      S += ((f + r) & 1 ? -1 : 1) * sum / ((xb - xa) * (yb - ya));
    }
  }
  return S;
}

function sampleGray(g, W, H, x, y) { // bilinear, clamped to the image
  x = Math.min(W - 1.001, Math.max(0, x)); y = Math.min(H - 1.001, Math.max(0, y));
  const xi = x | 0, yi = y | 0, fx = x - xi, fy = y - yi, p = yi * W + xi;
  return (g[p] * (1 - fx) + g[p + 1] * fx) * (1 - fy) + (g[p + W] * (1 - fx) + g[p + W + 1] * fx) * fy;
}

const unitCorners = N => [[0, 0], [N, 0], [N, N], [0, N]];

// Checkerboard correlation for an N x N grid inside any 4-corner outline (handles rotation and perspective).
function quadScore(g, W, H, quad, N = 8, n = 8) {
  const h = homography(unitCorners(N), quad); let S = 0;
  for (let r = 0; r < N; r++) for (let f = 0; f < N; f++) {
    let s = 0;
    for (let a = 0; a < n; a++) for (let c = 0; c < n; c++) {
      const [x, y] = project(h, f + (c + 0.5) / n, r + (a + 0.5) / n);
      s += sampleGray(g, W, H, x, y);
    }
    S += ((f + r) & 1 ? -1 : 1) * s / (n * n);
  }
  return S;
}

// Move each corner separately until the grid sits exactly on the squares (this is what fits perspective).
// Big steps use 4x4 samples per square (fast); fine steps use more (precise).
function climb(g, W, H, quad, N, fine = true) {
  const sq = Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]) / N;
  const stages = [[sq / 2, 4], [sq / 4, 4], [sq / 8, 4], [sq / 16, 4]].concat(fine ? [[sq / 32, 8], [sq / 64, 16]] : []);
  let cur = 0, n = 0;
  for (const [step, sn] of stages) {
    if (sn !== n) { n = sn; cur = Math.abs(quadScore(g, W, H, quad, N, n)); }
    for (let improved = true, it = 0; improved && it < 40; it++) {
      improved = false;
      for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) for (const d of [-step, step]) {
        const q = quad.map(p => p.slice()); q[i][j] += d;
        const sc = Math.abs(quadScore(g, W, H, q, N, n));
        if (sc > cur) { cur = sc; quad = q; improved = true; }
      }
    }
  }
  return { quad, cur };
}

// Board at any rotation and tilt:
// 1) coarse: in a small copy rotated in 7.5° steps, look for upright/squashed 8x8 grids (works for rotation)
//    and 4x4 patches (a small patch stays nearly square under perspective tilt);
// 2) fit each candidate exactly (4x4 patches are extended to every 8x8 placement first);
// 3) keep the candidate whose full 8x8 grid matches best. That final check is what rejects look-alikes,
//    e.g. a row of identical pieces seen diagonally can out-score a 4x4 patch but never a whole board.
// ponytail: steep camera tilt (> ~45°) turns the board into a strong trapezoid the rectangle search misses;
// Set corners covers it. Fix would be keystone-shaped boxes in the coarse search (~4x slower).
function detectBoard(g, W, H) {
  const k = Math.max(1, Math.round(Math.max(W, H) / 180)), w = Math.floor(W / k), h = Math.floor(H / k);
  const small = new Float32Array(w * h);
  for (let y = 0; y < h * k; y++) for (let x = 0; x < w * k; x++) small[Math.floor(y / k) * w + Math.floor(x / k)] += g[y * W + x] / (k * k);
  let mean = 0; for (const v of small) mean += v / small.length;
  const D = Math.ceil(Math.hypot(w, h)), rot = new Float32Array(D * D);
  const cands = []; // best per (angle, aspect, size N)
  for (let deg = 0; deg < 90; deg += 7.5) {
    const t = deg * Math.PI / 180, cs = Math.cos(t), sn = Math.sin(t);
    const back = (X, Y) => { const u = X - D / 2, v = Y - D / 2; return [w / 2 + u * cs - v * sn, h / 2 + u * sn + v * cs]; };
    for (let Y = 0; Y < D; Y++) for (let X = 0; X < D; X++) {
      const [x, y] = back(X + 0.5, Y + 0.5), xi = Math.floor(x), yi = Math.floor(y);
      rot[Y * D + X] = xi >= 0 && yi >= 0 && xi < w && yi < h ? small[yi * w + xi] : mean;
    }
    const I = integral(rot, D, D);
    for (const N of [8, 4]) for (const asp of [1, 0.7, 0.5]) {
      let best = null, bestSc = 0;
      for (let s = 5; s * N <= D; s++) {
        const sy = s * asp, step = Math.max(1, Math.round(s / 4));
        if (sy < 3) continue;
        for (let y0 = 0; y0 + N * sy <= D; y0 += step) for (let x0 = 0; x0 + N * s <= D; x0 += step) {
          const sc = Math.abs(gridScore(I, D, x0, y0, s, sy, N));
          if (sc > bestSc) { bestSc = sc; best = [x0, y0, s, sy]; }
        }
      }
      if (best) {
        const [x0, y0, s, sy] = best;
        cands.push({ N, sc: bestSc / (N * N), quad: [[x0, y0], [x0 + N * s, y0], [x0 + N * s, y0 + N * sy], [x0, y0 + N * sy]].map(([X, Y]) => back(X, Y).map(v => v * k)) });
      }
    }
  }
  if (!cands.length) return null;
  const place = (hm, ox, oy) => BOARD_CORNERS.map(([x, y]) => project(hm, x - ox, y - oy));
  // quick fit of every candidate into an 8x8 outline, scored with few samples
  const quick = cands.map(c => {
    if (c.N === 8) { const f = climb(g, W, H, c.quad, 8, false); return { quad: f.quad, est: f.cur }; }
    const h4 = homography(unitCorners(4), climb(g, W, H, c.quad, 4, false).quad);
    let best = { est: -1 };
    for (let oy = 0; oy <= 4; oy++) for (let ox = 0; ox <= 4; ox++) {
      const q = place(h4, ox, oy), est = Math.abs(quadScore(g, W, H, q, 8, 4));
      if (est > best.est) best = { quad: q, est };
    }
    return best;
  }).sort((a, b) => b.est - a.est);
  // full fit of the 3 best, with one-square shifts (a diagonal shift keeps the colours and can trap the climb)
  let fit = null;
  for (const c of quick.slice(0, 3)) {
    let f = climb(g, W, H, c.quad, 8);
    for (let tries = 0; tries < 4; tries++) {
      const hm = homography(BOARD_CORNERS, f.quad);
      let shifted = null, scS = f.cur * 1.02;
      for (const [dx, dy] of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) {
        const q = place(hm, dx, dy), sc = Math.abs(quadScore(g, W, H, q));
        if (sc > scS) { scS = sc; shifted = q; }
      }
      if (!shifted) break;
      f = climb(g, W, H, shifted, 8);
    }
    if (!fit || f.cur > fit.cur) fit = f;
  }
  return { quad: fit.quad, contrast: fit.cur / 32, tlLight: quadScore(g, W, H, fit.quad) > 0, cover: coverage(g, W, H, fit.quad) };
}

// Weakest rank or file, as a share of the average one. A board that is half off-screen (or covered)
// still fits a strong grid on the visible half, but its missing ranks show no checker pattern at all.
function coverage(g, W, H, quad) {
  const h = homography(unitCorners(8), quad), row = Array(8).fill(0), col = Array(8).fill(0);
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    let s = 0;
    for (let a = 0; a < 4; a++) for (let c = 0; c < 4; c++) s += sampleGray(g, W, H, ...project(h, f + (c + 0.5) / 4, r + (a + 0.5) / 4));
    const v = ((f + r) & 1 ? -1 : 1) * s / 16;
    row[r] += v; col[f] += v;
  }
  const all = [...row, ...col].map(Math.abs), avg = all.reduce((a, b) => a + b) / 16;
  return avg ? Math.min(...all) / avg : 0;
}

// Turn a detected outline into a8,h8,h1,a1 corners: a8 is a light square, and White's pieces sit on the bright side.
function orientBoard(img, b, grid) {
  const quad = b.quad; // clockwise, quad[0] is the corner of the square whose colour tlLight gives
  let best = null;
  for (const o of b.tlLight ? [0, 2] : [1, 3]) {
    const c = [0, 1, 2, 3].map(k => quad[(k + o) % 4]);
    const feats = squareFeatures(img, homography(BOARD_CORNERS, c));
    const avg = col => { let s = 0, n = 0; feats.forEach((ft, i) => { if (grid[i] === col) { s += ft[0]; n++; } }); return n ? s / n : 0; };
    const sc = avg('w') - avg('b');
    if (!best || sc > best.sc) best = { sc, corners: c, feats };
  }
  return best;
}

// Calibrate against whichever known position (e.g. current game vs. start) the picture fits best.
// A wrong position gives inconsistent references, so its self-classification error (err) is high.
function bestCalibration(img, b, grids) {
  let best = null;
  for (const grid of grids) {
    const o = orientBoard(img, b, grid);
    o.grid = grid; o.refs = buildRefs(o.feats, grid);
    o.err = diff(classify(o.feats, o.refs), grid);
    if (!best || o.err < best.err) best = o;
  }
  return best;
}

function selfTest() {
  const ok = (c, m) => { if (!c) throw new Error('FAIL: ' + m); };
  const pts = [[100, 50], [500, 60], [520, 480], [90, 470]];
  const h = homography(BOARD_CORNERS, pts);
  BOARD_CORNERS.forEach(([x, y], i) => {
    const [u, v] = project(h, x, y);
    ok(Math.abs(u - pts[i][0]) < 1e-6 && Math.abs(v - pts[i][1]) < 1e-6, 'homography');
  });
  const play = (fen, sans) => { const c = new Chess(fen); sans.forEach(s => c.move(s)); return gridOf(c); };
  const START = new Chess().fen();
  let r = findMoves(new Chess(), play(START, ['e4']));
  ok(r.diff === 0 && r.moves[0].san === 'e4', 'e4');
  r = findMoves(new Chess(), play(START, ['e4', 'e5']));
  ok(r.diff === 0 && r.moves.map(m => m.san).join() === 'e4,e5', '2-ply');
  const mid = 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3';
  let t = Date.now();
  r = findMoves(new Chess(mid), play(mid, ['Bb5', 'a6', 'Bxc6']));
  ok(r.diff === 0 && r.moves.map(m => m.san).join() === 'Bb5,a6,Bxc6', '3 fast moves');
  r = findMoves(new Chess(mid), play(mid, ['Bc4', 'Nf6', 'O-O', 'Bc5']));
  ok(r.diff === 0 && r.moves.length === 4, '4 fast moves incl. castling');
  r = findMoves(new Chess(mid), [...gridOf(new Chess(mid))].map((c, i) => i % 3 ? c : '.').join(''));
  ok(Date.now() - t < 3000, 'deep search stays fast, even on a garbage scan (' + (Date.now() - t) + 'ms)');
  const cf = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
  r = findMoves(new Chess(cf), play(cf, ['O-O']));
  ok(r.diff === 0 && r.moves[0].san === 'O-O', 'castle');
  const ef = '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1';
  r = findMoves(new Chess(ef), play(ef, ['exd6']));
  ok(r.diff === 0 && r.moves[0].san === 'exd6', 'en passant');
  const pf = '4k3/P7/8/8/8/8/8/4K3 w - - 0 1';
  r = findMoves(new Chess(pf), play(pf, ['a8=Q']));
  ok(r.moves[0].promotion === 'q', 'promote to queen by default');
  const tc = new Chess(); ['e4', 'e5', 'Nf3', 'Nc6'].forEach(s => tc.move(s));
  const tb = findTakeback(tc, play(START, ['e4', 'e5']));
  ok(tb.n === 2 && tb.diff === 0 && tc.history().length === 4, 'takeback of 2 plies, game left intact');
  const grid = gridOf(new Chess());
  const feats = [...grid].map((g, i) => [g === '.' ? (isLight(i) ? 200 : 80) : g === 'w' ? 230 : 30, g === '.' ? 5 : 40]);
  ok(classify(feats, buildRefs(feats, grid)) === grid, 'classify');
  const lit = feats.map((ft, i) => i === 36 || i === 44 ? [ft[0] + 40, ft[1]] : ft); // e4, e3 coloured as last move
  ok(classify(lit, buildRefs(feats, grid)) === grid, 'last-move highlight stays empty');
  ok(fenFromGrid(piecesOf(new Chess()), 'w') === START, 'fen');
  const inf = parseInfo('info depth 12 seldepth 16 multipv 2 score cp -35 nodes 1 pv e7e5 g1f3');
  ok(inf.k === 2 && inf.score.v === -35 && inf.uci === 'e7e5' && inf.pv.join() === 'e7e5,g1f3', 'parse engine line');
  ok(reviewMove({ cp: 0, best: 'a1a2', second: -400 }, { cp: 0 }, 'a1a2').label === 'Great', 'review: only good move = Great');
  ok(reviewMove({ cp: 400, best: 'a1a2' }, { cp: 0 }, 'b1b2', { prevBad: true }).label === 'Miss', 'review: failing to punish = Miss');
  ok(reviewMove({ cp: 50, best: 'd1d5' }, { cp: -50 }, 'd1d5', { sacrifice: true }).label === 'Brilliant', 'review: sound sacrifice = Brilliant');
  const sacFen = '4k3/8/2p5/8/8/8/8/3RK3 w - - 0 1';
  ok(isSacrifice(sacFen, 'd1d5', 'c6d5', 'e1e2') && !isSacrifice(sacFen, 'd1d2', 'e8e7', 'e1e2'), 'detect sacrifice');
  ok(pvSan(START, ['e2e4', 'e7e5', 'g1f3'], 2).join(' ') === 'e4 e5', 'engine line to SAN');
  const site = '[Event "Live Chess"]\n[White "me"]\n[Black "you"]\n[Result "1-0"]\n\n1. e4 {[%clk 0:02:59]} 1... e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0';
  const imp = gameFromPgn(site);
  ok(imp && imp.moves.length === 7 && imp.moves[6].san === 'Qxf7#' && imp.result === '1-0' && imp.moves[2].fen.includes(' w '), 'import PGN with headers and clock comments');
  const exp = gameToPgn({ ...imp, date: Date.UTC(2026, 10, 3) });
  ok(exp.complete && exp.pgn.includes('4. Qxf7# 1-0') && gameFromPgn(exp.pgn).moves.length === 7, 'export PGN round-trips');
  const gap = { moves: [imp.moves[0], imp.moves[1], imp.moves[4]], result: '1-0' };
  ok(!gameToPgn(gap).complete && gameToPgn(gap).pgn.includes('1. e4 e5 *'), 'export stops at a gap');
  const fromFen = gameFromPgn('[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]\n\n1. e4 *');
  ok(fromFen && fromFen.moves[0].fen.startsWith('4k3/') && gameToPgn(fromFen).pgn.includes('[FEN "4k3/'), 'PGN from a set-up position');
  ok(gameFromPgn('not a game at all ###') === null, 'reject junk PGN');
  ok(formatScore({ type: 'cp', v: -35 }, 'b') === '+0.3' && formatScore({ type: 'mate', v: 3 }, 'b') === '-M3', 'score from White view');
  // the garbled resync from the field: pawns on a8/b8/f8, three white bishops, rook next to the king
  const garbled = piecesOf(new Chess('pp1rkpnn/pp1ppppp/1p6/2p5/4P3/P1N2N2/1PPP1PPP/B1BrKB1R w - - 0 1'));
  ok(!plausible(garbled, 'w') && !plausible(garbled, 'b'), 'reject impossible position');
  ok(plausible(piecesOf(new Chess()), 'w'), 'accept start position');
  ok(!plausible(piecesOf(new Chess('4k3/8/8/8/8/8/8/R3K2r w - - 0 1')), 'b'), 'reject: side that just moved left in check');
  const before = piecesOf(new Chess()), misread = before.slice();
  misread[0] = 'p'; misread[52] = '.'; misread[36] = 'B'; // a8 rook read as pawn; e2 pawn moved to e4 but read as bishop
  const merged = mergeRead(before, misread);
  ok(merged[0] === 'r' && merged[36] === 'B' && merged[52] === '.', 'merge keeps known types on unchanged squares');
  ok(reviewMove({ cp: 30, best: 'e2e4' }, { cp: -30 }, 'e2e4').label === 'Best', 'review: best move');
  ok(reviewMove({ cp: 30, best: 'e2e4' }, { cp: -10 }, 'd2d4').label === 'Excellent', 'review: tiny loss');
  ok(reviewMove({ cp: 50, best: 'e2e4' }, { cp: 300 }, 'f2f3').label === 'Blunder', 'review: +0.5 to -3 is a blunder');
  ok(reviewMove({ cp: 800, best: 'e2e4' }, { cp: -750 }, 'd2d4').label === 'Excellent', 'review: still totally winning');
  ok(Math.abs(reviewMove({ cp: 0, best: 'a' }, { cp: 0 }, 'a').accuracy - 100) < 0.01, 'review: perfect accuracy');
  ok(sanToSpeech('Nxf3+') === 'Knight takes f3, check' && sanToSpeech('exd8=Q#') === 'Pawn takes d8, promotes to Queen, checkmate' &&
     sanToSpeech('O-O') === 'Castle kingside', 'speech text');

  // synthetic camera frame: noisy background, board at (101,57) with 37px squares, White at the top (board flipped)
  const W = 640, Ht = 480, S = 37, X = 101, Y = 57, img = { width: W, height: Ht, data: new Uint8ClampedArray(W * Ht * 4) };
  let seed = 1; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let y = 0; y < Ht; y++) for (let x = 0; x < W; x++) {
    let v = 100 + 50 * rnd();
    const bx = x - X, by = y - Y;
    if (bx >= 0 && by >= 0 && bx < 8 * S && by < 8 * S) {
      const f = Math.floor(bx / S), r = Math.floor(by / S), cx = bx % S - S / 2, cy = by % S - S / 2;
      v = (f + r) & 1 ? 70 : 200;
      if (cx * cx + cy * cy < (S * 0.3) ** 2) { if (r < 2) v = 245; if (r > 5) v = 20; } // pieces
    }
    img.data.fill(v, (y * W + x) * 4, (y * W + x) * 4 + 3);
  }
  const det = detectBoard(toGray(img), W, Ht);
  const near = (quad, truth, tol) => truth.every(t => quad.some(p => Math.hypot(p[0] - t[0], p[1] - t[1]) < tol));
  ok(det && near(det.quad, [[X, Y], [X + 8 * S, Y], [X + 8 * S, Y + 8 * S], [X, Y + 8 * S]], 2), 'detect upright board');
  ok(det.cover >= 0.25, 'whole board counts as fully visible');
  // same board with its lower half off-screen (dark): must be refused, not fitted onto the dark area
  const half = { width: W, height: Ht, data: img.data.slice() };
  for (let y = Y + 4 * S; y < Ht; y++) half.data.fill(20, y * W * 4, (y + 1) * W * 4);
  const dh = detectBoard(toGray(half), W, Ht);
  ok(!dh || dh.contrast < 15 || dh.cover < 0.25, 'half-visible board is refused');

  // rotated + tilted board (perspective), White at the top
  const T = [[170, 60], [480, 130], [470, 430], [110, 380]], toBoard = homography(T, BOARD_CORNERS);
  const img2 = { width: W, height: Ht, data: new Uint8ClampedArray(W * Ht * 4) };
  for (let y = 0; y < Ht; y++) for (let x = 0; x < W; x++) {
    let v = 100 + 50 * rnd();
    const [u, w] = project(toBoard, x + 0.5, y + 0.5);
    if (u >= 0 && w >= 0 && u < 8 && w < 8) {
      const f = Math.floor(u), r = Math.floor(w), cx = u - f - 0.5, cy = w - r - 0.5;
      v = (f + r) & 1 ? 70 : 200;
      if (cx * cx + cy * cy < 0.09) { if (r < 2) v = 245; if (r > 5) v = 20; }
    }
    img2.data.fill(v, (y * W + x) * 4, (y * W + x) * 4 + 3);
  }
  const det2 = detectBoard(toGray(img2), W, Ht);
  ok(det2 && near(det2.quad, T, 4), 'detect rotated + tilted board');
  ok(det2.cover >= 0.25, 'tilted board counts as fully visible');
  const o2 = orientBoard(img2, det2, grid);
  ok(Math.hypot(o2.corners[0][0] - T[2][0], o2.corners[0][1] - T[2][1]) < 4, 'orient rotated board');
  ok(classify(o2.feats, buildRefs(o2.feats, grid)) === grid, 'calibrate rotated board');
  const o = orientBoard(img, det, grid);
  ok(Math.abs(o.corners[0][0] - (X + 8 * S)) < 2 && Math.abs(o.corners[0][1] - (Y + 8 * S)) < 2, 'orient: a8 bottom-right when White is on top');
  ok(classify(o.feats, buildRefs(o.feats, grid)) === grid, 'auto calibrate');
  const stale = play(START, ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'O-O']); // old game left in memory
  const cal = bestCalibration(img, det, [stale, grid]);
  ok(cal.grid === grid && cal.err === 0, 'calibrate picks the start position over a stale game');

  // full position reading: render boards whose pieces have a distinct shape per type
  const drawBoard = (pieces, hl = []) => {
    const Z = 40, im = { width: 8 * Z, height: 8 * Z, data: new Uint8ClampedArray(64 * Z * Z * 4) };
    for (let y = 0; y < 8 * Z; y++) for (let x = 0; x < 8 * Z; x++) {
      const i = Math.floor(y / Z) * 8 + Math.floor(x / Z), p = pieces[i], t = 'pnbrqk'.indexOf(p.toLowerCase());
      let c = hl.includes(i) ? [246, 246, 105] : isLight(i) ? [238, 238, 210] : [118, 150, 86];
      const dx = Math.abs(x % Z - Z / 2), dy = Math.abs(y % Z - Z / 2);
      if (p !== '.' && dx < 4 + 2 * t && dy < 14 - t) c = p === p.toUpperCase() ? [250, 250, 250] : [25, 25, 25];
      im.data.set([...c, 255], (y * 8 * Z + x) * 4);
    }
    return im;
  };
  const hb = homography(BOARD_CORNERS, [[0, 0], [320, 0], [320, 320], [0, 320]]);
  const model = buildPieceModel(squarePatches(drawBoard(piecesOf(new Chess())), hb), piecesOf(new Chess()));
  const later = new Chess(); ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5', 'a6', 'Bxc6'].forEach(s => later.move(s));
  const sq = s => FILES.indexOf(s[0]) + (8 - +s[1]) * 8;
  const pt = squarePatches(drawBoard(piecesOf(later), [sq('b5'), sq('c6')]), hb);
  const read = readPieces(pt, model);
  ok(read.join('') === piecesOf(later).join(''), 'read whole position from piece shapes');
  ok(read.unsure.size === 0, 'clean picture: every square read with confidence');
  ok(lastMover(pt, read, model) === 'w', 'last-move highlight shows White just moved');

  // opening book: named end positions, unnamed steps on the way, transposition finds the name
  const key = (...sans) => { const c = new Chess(); sans.forEach(s => c.move(s)); return posKey(c.fen()); };
  const bk = openingBook([[key('e4'), ''], [key('e4', 'e5'), "C20 King's Pawn Game"], [key('e4', 'e5', 'Nf3'), ''],
    [key('e4', 'e5', 'Nf3', 'Nc6'), ''], [key('e4', 'e5', 'Nf3', 'Nc6', 'Bc4'), 'C50 Italian Game']].map(r => r.join('\t')).join('\n') + '\n');
  const at = (...sans) => bk.get(key(...sans));
  ok(at('e4', 'e5') === "C20 King's Pawn Game" && at('e4', 'e5', 'Nf3') === null && at('e4', 'e5', 'Nf3', 'Nc6', 'Bc4') === 'C50 Italian Game', 'opening book names');
  ok(at('e4', 'e5', 'Bc4', 'Nc6', 'Nf3') === 'C50 Italian Game' && at('d4') === undefined, 'opening book: transposition, and out of book');
  return 'all tests passed';
}

if (typeof module !== 'undefined') module.exports = { selfTest };

// ---------- browser app ----------
if (typeof document !== 'undefined') (() => {
  const $ = id => document.getElementById(id);
  const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d');
  const boardEl = $('board'), statusEl = $('status'), bestEl = $('best');
  const GLYPH = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' };
  const CORNER_NAMES = ['a8', 'h8', 'h1', 'a1'];
  const load = k => { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } };
  const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } };

  let chess = new Chess();
  const savedFen = load('fen');
  if (savedFen && chess.load(savedFen) && !plausible(piecesOf(chess), chess.turn())) chess.reset(); // drop a garbled saved position
  let corners = load('corners'), refs = load('refs'), model = load('model');
  let tapping = null, editing = null, brush = 'P', highlight = [], lines = [], lastScan = null;
  $('speak').checked = load('speak') ?? true;
  $('speak').onchange = () => save('speak', $('speak').checked);

  const status = t => { statusEl.textContent = t; };
  const H = () => homography(BOARD_CORNERS, corners);

  if (location.search.includes('test')) {
    try { status(selfTest()); } catch (e) { status(e.message); }
  }

  // video source: camera or shared screen/window; in the Android app also the phone's own screen (shot)
  // the page is loaded from the website, so only Capacitor's injected bridge is here (no registerPlugin)
  const cap = window.Capacitor;
  const Phone = cap?.isNativePlatform?.() ? {
    start: () => cap.nativePromise('Screen', 'start', {}),
    stop: () => cap.nativePromise('Screen', 'stop', {}),
    bubble: o => cap.nativePromise('Screen', 'bubble', o),
    addListener: (ev, cb) => cap.addListener('Screen', ev, cb),
  } : null;
  let stream = null, shot = null; // shot = canvas with the latest phone-screen picture
  const frameW = () => shot ? shot.width : video.videoWidth;
  function stopPhone() { // leave phone-screen pictures (read on the phone, or watched on the PC)
    if (peer) { peer.destroy(); peer = null; viewers.clear(); $('stream').textContent = Phone ? '📡 Stream to PC' : '📡 Watch phone'; }
    if (shot) { if (Phone) Phone.stop(); shot.remove(); shot = null; video.hidden = false; }
  }
  // switch to phone-screen pictures as the source (they arrive through showFrame)
  function useFrames(msg) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; corners = refs = model = null; lastScan = null; drawOverlay();
    status(msg);
    startLive();
  }
  async function showFrame(data) {
    const img = new Image(); img.src = data;
    try { await img.decode(); } catch { return; }
    if (!shot) { shot = document.createElement('canvas'); shot.id = 'shot'; $('cam').prepend(shot); video.hidden = true; }
    if (shot.width !== img.width || shot.height !== img.height) { shot.width = img.width; shot.height = img.height; fit(); }
    shot.getContext('2d').drawImage(img, 0, 0);
    if (liveTimer) tick(); // frames drive the checks: Android slows timers while the chess app is in front
  }
  function useStream(s, label) {
    stopPhone();
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = s; video.srcObject = s;
    corners = refs = model = null; lastScan = null; drawOverlay(); // old corners belong to the old source
    s.getVideoTracks()[0].onended = () => { if (stream === s) status(label + ' stopped.'); };
    status(label + ' on. Show the board at the position below — scanning automatically.');
    startLive();
  }
  const startCamera = () => navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } } })
    .then(s => useStream(s, 'Camera'))
    .catch(e => status('Camera error: ' + e.message + ' (needs https or localhost)'));
  $('camera').onclick = startCamera;
  $('screen').onclick = () => navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 10 }, audio: false })
    .then(s => useStream(s, 'Screen share'))
    .catch(e => status('Screen share cancelled: ' + e.message));
  // keep saved corners on first load: the camera is the default source (in the app: wait for a button)
  if (!Phone) navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 } } })
    .then(s => { stream = s; video.srcObject = s; startLive(); })
    .catch(() => status('No camera. Press Share screen to read a board on your screen.'));
  const fit = () => { overlay.width = frameW(); overlay.height = shot ? shot.height : video.videoHeight; drawOverlay(); };
  video.addEventListener('loadedmetadata', fit);
  video.addEventListener('resize', fit); // shared window was resized

  // Android app: read the phone's own screen; the best move shows in a floating bubble over the chess app
  if (Phone) {
    $('screen').textContent = '📱 Read this phone';
    $('screen').onclick = () => Phone.start()
      .then(() => useFrames('Reading this phone\'s screen. Open your chess app; the best move shows in the blue bubble (drag it off the board).'))
      .catch(e => status('Screen reading cancelled: ' + e.message));
    Phone.addListener('frame', ({ data }) => {
      showFrame(data);
      // also to any PC watching; skip a picture while the line is still busy with earlier ones
      viewers.forEach(c => { if (c.open && c.dataChannel.bufferedAmount < 1e6) c.send(data); });
    });
    new MutationObserver(() => Phone.bubble({ text: bestEl.textContent }))
      .observe(bestEl, { childList: true, characterData: true, subtree: true });
  }

  // live stream phone -> PC page. The free PeerJS broker only introduces the two; the pictures (the same JPEGs
  // the phone reads, 4 a second) then go straight from phone to PC over an encrypted WebRTC data channel.
  // (Not a video track: the Android WebView sends no frames from a canvas stream.) The code is the only key.
  const peerId = code => 'eyechess-' + code.toLowerCase();
  const viewers = new Set();
  let peer = null;
  $('stream').textContent = Phone ? '📡 Stream to PC' : '📡 Watch phone';
  $('stream').onclick = () => {
    if (typeof Peer === 'undefined') return status('Streaming needs internet (couldn\'t load PeerJS).');
    if (Phone) {
      if (!shot) return status('Press 📱 Read this phone first, then 📡 Stream to PC.');
      if (peer) { peer.destroy(); viewers.clear(); }
      const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789', c = [...crypto.getRandomValues(new Uint8Array(8))].map(b => A[b % 32]).join('');
      const code = c.slice(0, 4) + '-' + c.slice(4);
      peer = new Peer(peerId(c));
      $('stream').textContent = '📡 Code ' + code; // stays visible; the status line keeps changing
      peer.on('open', () => status(`Streaming. On your PC open the EyeChess site, press 📡 Watch phone and enter ${code}`));
      peer.on('connection', conn => {
        conn.on('open', () => { viewers.add(conn); status(`PC connected and watching (code ${code}).`); });
        conn.on('close', () => viewers.delete(conn));
      });
      peer.on('error', e => status('Streaming error: ' + e.type));
    } else {
      const code = (prompt('Code shown on the phone (e.g. K7QM-2XPA):') || '').replace(/[^a-z0-9]/gi, '');
      if (!code) return;
      stopPhone(); // drop an earlier watch
      peer = new Peer();
      peer.on('open', () => {
        status('Connecting to the phone…');
        const conn = peer.connect(peerId(code));
        conn.on('open', () => { $('stream').textContent = '📡 Watching'; useFrames('Watching the phone. Open your chess app on the phone; the best move shows here.'); });
        conn.on('data', d => { if (typeof d === 'string' && d.startsWith('data:image/')) showFrame(d); });
        conn.on('close', () => status('The phone stopped streaming.'));
      });
      peer.on('error', e => status(e.type === 'peer-unavailable' ? 'No phone is streaming with that code.' : 'Streaming error: ' + e.type));
    }
  };

  function grabFrame() {
    const c = document.createElement('canvas');
    c.width = frameW(); c.height = shot ? shot.height : video.videoHeight;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(shot || video, 0, 0);
    return g.getImageData(0, 0, c.width, c.height);
  }

  function drawOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = Math.max(2, overlay.width / 400);
    const pts = tapping || corners;
    if (!pts) return;
    ctx.fillStyle = '#f33';
    pts.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, ctx.lineWidth * 3, 0, 7); ctx.fill(); ctx.fillText(CORNER_NAMES[i], x + 8, y - 8); });
    if (tapping) return;
    const h = H();
    ctx.strokeStyle = 'rgba(0,255,0,.7)';
    for (let k = 0; k <= 8; k++) {
      ctx.beginPath(); ctx.moveTo(...project(h, k, 0)); ctx.lineTo(...project(h, k, 8)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(...project(h, 0, k)); ctx.lineTo(...project(h, 8, k)); ctx.stroke();
    }
    if (lastScan) for (let i = 0; i < 64; i++) {
      if (lastScan[i] === '.') continue;
      const [x, y] = project(h, i % 8 + 0.5, Math.floor(i / 8) + 0.5);
      ctx.fillStyle = lastScan[i] === 'w' ? '#fff' : '#000';
      ctx.strokeStyle = '#f0f';
      ctx.beginPath(); ctx.arc(x, y, ctx.lineWidth * 4, 0, 7); ctx.fill(); ctx.stroke();
    }
    // move arrows on the camera view: best = strong, alternatives = faint and thinner (drawn first, under it)
    const base = ctx.lineWidth;
    [...lines].reverse().forEach((ln, j, all) => {
      const best = j === all.length - 1;
      const c = s => project(h, FILES.indexOf(s[0]) + 0.5, 8 - +s[1] + 0.5);
      const [x1, y1] = c(ln.from), [x2, y2] = c(ln.to);
      const a = Math.atan2(y2 - y1, x2 - x1), L = base * (best ? 8 : 6);
      ctx.strokeStyle = ctx.fillStyle = `rgba(59,130,246,${best ? 0.9 : 0.4})`;
      ctx.lineWidth = base * (best ? 3 : 2);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - L * Math.cos(a), y2 - L * Math.sin(a)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - L * 1.5 * Math.cos(a - 0.5), y2 - L * 1.5 * Math.sin(a - 0.5));
      ctx.lineTo(x2 - L * 1.5 * Math.cos(a + 0.5), y2 - L * 1.5 * Math.sin(a + 0.5));
      ctx.fill();
    });
  }

  overlay.addEventListener('click', e => {
    if (!tapping) return;
    const r = overlay.getBoundingClientRect();
    tapping.push([(e.clientX - r.left) * overlay.width / r.width, (e.clientY - r.top) * overlay.height / r.height]);
    if (tapping.length === 4) {
      corners = tapping; tapping = null; save('corners', corners);
      status('Corners set. Now press Calibrate with the board matching the position shown below.');
    } else status(`Tap the outer corner of ${CORNER_NAMES[tapping.length]}`);
    drawOverlay();
  });

  $('corners').onclick = () => { tapping = []; lastScan = null; drawOverlay(); status('Tap the outer corner of a8 (Black\'s queenside rook corner)'); };

  // find the board, orient it, and calibrate in one go (board must match the position shown below)
  function autoSetup() {
    if (!frameW()) return status('Camera not ready yet.'), false;
    const img = grabFrame(), gray = toGray(img);
    let max = 0; for (const v of gray) if (v > max) max = v;
    if (max < 10) return status(`The shared picture is blank (${img.width}×${img.height}). Make sure the game window isn't minimised.`), false;
    const b = detectBoard(gray, img.width, img.height);
    if (!b || b.contrast < 15) return status(`No board found in the ${img.width}×${img.height} picture (contrast ${b ? b.contrast.toFixed(0) : 0}). ` +
      'Share the game\'s own window or tab, not the entire screen, and make sure the whole board is visible. Or use Set corners.'), false;
    if (b.cover < 0.25) return status('Only part of the board is visible: some ranks or files are cut off or covered. ' +
      'Scroll or resize the game so the whole board shows, then Auto setup again.'), false;
    const START = gridOf(new Chess());
    const o = bestCalibration(img, b, [...new Set([gridOf(chess), START])]);
    if (o.grid !== gridOf(chess)) { archiveGame(); chess.reset(); } // screen shows a fresh game, drop the remembered one
    corners = o.corners; refs = o.refs; lastScan = null;
    save('corners', corners); save('refs', refs);
    if (o.err <= 2) learnModel(img);
    changed();
    if (o.err > 2) { $('sync').hidden = false; status(`Board found, but it doesn't match the position below (${o.err} squares differ).`); }
    else status('Board found and calibrated. Check the green grid sits on the board.');
    return true;
  }
  $('auto').onclick = autoSetup;

  // Joined mid-game: take the position from a pasted PGN/FEN, then learn this board's look from the screen.
  $('syncGo').onclick = () => {
    const text = $('syncIn').value.trim(), c = new Chess();
    let g = null;
    if (!c.load(text)) { // not a FEN: try PGN
      g = gameFromPgn(text);
      if (!g || !g.moves.length) return status('Couldn\'t read that. Paste the whole PGN (with the moves) or a FEN.');
      c.load_pgn(text, { sloppy: true });
    }
    archiveGame();
    if (g) { game.moves = g.moves; save('game', game); }
    chess = c; lastScan = null; prev = null;
    $('syncIn').value = ''; $('sync').hidden = true;
    // learn this board from the screen; autoSetup re-shows the banner if the screen doesn't show this position
    const found = frameW() && autoSetup();
    if (!found || $('sync').hidden) changed(true);
  };
  $('syncEdit').onclick = () => { $('sync').hidden = true; if (!editing) $('edit').click(); };
  $('syncClose').onclick = () => { $('sync').hidden = true; };

  $('calibrate').onclick = () => {
    if (!corners) return status('Set corners first.');
    const img = grabFrame();
    refs = buildRefs(squareFeatures(img, H()), gridOf(chess));
    save('refs', refs); learnModel(img);
    status('Calibrated. Make a move on the board, then press Scan.');
  };

  function scan(live) {
    if (!corners || !refs) return status('Set corners and calibrate first.');
    const img = grabFrame();
    const feats = squareFeatures(img, H());
    lastScan = classify(feats, refs);
    drawOverlay();

    // new game on screen: start position seen, either way round (rematch often swaps colours)
    const START = gridOf(new Chess());
    const flipped = diff([...lastScan].reverse().join(''), START) <= 1;
    if (gridOf(chess) !== START && (diff(lastScan, START) <= 1 || flipped)) {
      if (flipped) {
        corners = [corners[2], corners[3], corners[0], corners[1]]; // a8 <-> h1
        save('corners', corners);
        if ($('side').value !== 'both') $('side').value = $('side').value === 'w' ? 'b' : 'w';
      }
      archiveGame(); chess.reset();
      refs = buildRefs(squareFeatures(img, H()), START); save('refs', refs); learnModel(img);
      lastScan = START; drawOverlay();
      status(flipped ? 'New game detected — board flipped, you now play ' + ($('side').value === 'w' ? 'White.' : $('side').value === 'b' ? 'Black.' : 'both sides.') : 'New game detected.');
      return changed();
    }

    const r = findMoves(chess, lastScan);
    if (r.diff > 0) { // takeback on the screen fits better than any forward move?
      const tb = findTakeback(chess, lastScan);
      if (tb.diff < r.diff && tb.diff <= (live ? 1 : 2)) {
        for (let k = 0; k < tb.n; k++) chess.undo();
        game.moves.splice(-tb.n); save('game', game);
        status(`Takeback seen: ${tb.n} move${tb.n > 1 ? 's' : ''} undone.`);
        return changed(true);
      }
    }
    if (!r.moves.length && r.diff === 0) return live || status('No change seen.');
    if (!r.moves.length || r.diff > (live ? 1 : 2)) {
      const res = resync(img);
      if (res === 'moved') { status('Resynced: read the whole board from the screen.'); return changed(true); }
      if (res === 'same') return live || status('No change seen.');
      if (res === 'wait') return status('Board changed unexpectedly. Checking it again…');
      $('sync').hidden = false;
      return status(`Out of sync: the screen doesn't match the position below (${r.diff} squares off).`);
    }
    const shot = boardShot(img);
    r.moves.forEach(m => { const fen = chess.fen(); chess.move(m); logMove(fen, m, shot); });
    // re-learn from this frame: adapts to lighting drift and to sites that colour the last-move squares
    if (r.diff === 0) { refs = buildRefs(feats, gridOf(chess)); save('refs', refs); }
    status(`Seen: ${r.moves.map(m => m.san).join(' ')}` + (r.diff ? ` (${r.diff} squares unsure — check board below)` : ''));
    changed(true);
  }
  $('scan').onclick = () => scan(false);

  // piece shapes, learned whenever the position on screen is known
  function learnModel(img) { model = buildPieceModel(squarePatches(img, H()), piecesOf(chess)); save('model', model); }

  // Out of sync: read every piece from the screen and continue from that position.
  // Strict on purpose: a wrong resync silently corrupts the game (e.g. your queen turning into an enemy rook).
  let pendingRead = null;
  function resync(img) {
    if (!model) return null;
    const old = piecesOf(chess), patches = squarePatches(img, H()), read = readPieces(patches, model);
    const pieces = mergeRead(old, read);
    const colours = pieces.map(p => p === '.' ? '.' : p === p.toUpperCase() ? 'w' : 'b').join('');
    if (diff(colours, lastScan) > 0) return null; // the two readers must agree on every square
    if (pieces.some((p, i) => p !== old[i] && p !== '.' && read.unsure.has(i))) return null; // a changed piece we can't clearly recognise
    if (pieces.filter(p => p === 'K').length !== 1 || pieces.filter(p => p === 'k').length !== 1) return null;
    const key = pieces.join('');
    if (key === old.join('')) return 'same';
    if (pendingRead !== key) { pendingRead = key; return 'wait'; } // must read the same position twice in a row
    pendingRead = null;
    const mover = lastMover(patches, pieces, model), side = $('side').value;
    const turn = mover ? (mover === 'w' ? 'b' : 'w') : side === 'both' ? chess.turn() : side; // no highlight: assume it's your turn
    const c = new Chess();
    if (!plausible(pieces, turn) || !c.load(fenFromGrid(pieces, turn))) return null; // misread: stay out of sync rather than guess
    chess = c;
    return 'moved';
  }

  // live: look 4x a second, act once the board has been still for ~0.5s (3 equal frames: animation / hand gone)
  let liveTimer = null, prev = null, still = 0, searchTick = 0, lost = 0;
  $('live').onclick = () => liveTimer ? stopLive() : startLive();
  function stopLive() { clearInterval(liveTimer); liveTimer = null; $('live').textContent = '▶ Live'; $('live').classList.remove('live'); }
  function startLive() {
    if (liveTimer) return;
    $('live').textContent = '■ Stop'; $('live').classList.add('live');
    status('Live: watching the board…');
    liveTimer = setInterval(() => { if (!shot) tick(); }, 250); // phone screen: each new picture calls tick() instead
  }
  function tick() {
    if (editing || !frameW()) return;
    if (!corners || !refs) { // keep looking until the board shows up; search takes ~1s, so only every 3s
      if (++searchTick % 12 === 1) autoSetup();
      return;
    }
    const img = grabFrame();
    // every 2s: does the grid still sit on a whole board? (cheap check; the full search is ~1s, too slow to run nonstop)
    // two misses in a row (~4s, so a passing hand or popup doesn't count) = board moved, scrolled or got covered: search again
    if (++searchTick % 8 === 0) {
      lost = coverage(toGray(img), img.width, img.height, corners) < 0.25 ? lost + 1 : 0;
      if (lost >= 2) { lost = 0; corners = null; lastScan = null; drawOverlay(); status('Live: lost the board, searching again…'); return; }
    }
    const g = classify(squareFeatures(img, H()), refs);
    still = g === prev ? still + 1 : 0;
    prev = g;
    if (still >= 2 && g !== gridOf(chess)) scan(true); // steady and different: keep retrying until in sync
  }

  $('undo').onclick = () => { if (chess.undo()) { game.moves.pop(); save('game', game); } changed(); };
  $('reset').onclick = () => { archiveGame(); chess.reset(); changed(); status('New game. Set up the start position.'); };
  $('side').onchange = () => changed();

  // edit mode
  $('edit').onclick = () => {
    if (editing) {
      const c = new Chess();
      if (!plausible(editing, $('turn').value) || !c.load(fenFromGrid(editing, $('turn').value)))
        return status('Impossible position: check kings, pawns on the back rank, piece counts, and who is to move.');
      archiveGame(); chess = c; editing = null;
      $('palette').hidden = true; $('edit').textContent = 'Edit';
      status('Position set. Press Calibrate if the board matches it.');
      changed();
    } else {
      editing = piecesOf(chess);
      $('turn').value = chess.turn();
      $('palette').hidden = false; $('edit').textContent = 'Done';
      render();
    }
  };
  $('palette').addEventListener('click', e => {
    const p = e.target.dataset.p; if (!p) return;
    brush = p;
    [...$('palette').children].forEach(b => b.classList.toggle('on', b.dataset.p === p));
  });

  // Draw a board: pieces = 64 FEN letters, hl = squares to outline, arrows = [{from, to, color, opacity, width}] (last on top).
  function drawBoard(el, pieces, { flip = false, hl = [], arrows = [], onSquare = null, badge = null } = {}) {
    el.innerHTML = '';
    for (let k = 0; k < 64; k++) {
      const i = flip ? 63 - k : k, p = pieces[i], d = document.createElement('div');
      d.className = 'sq ' + (isLight(i) ? 'l' : 'd') + (hl.includes(sqName(i)) ? ' hl' : '');
      if (p !== '.') { d.textContent = GLYPH[p.toLowerCase()]; d.classList.add(p === p.toUpperCase() ? 'w' : 'b'); }
      if (onSquare) d.onclick = () => onSquare(i);
      el.append(d);
    }
    if (!arrows.length && !badge) return;
    const at = s => { const i = FILES.indexOf(s[0]) + (8 - +s[1]) * 8, k = flip ? 63 - i : i; return [k % 8 + 0.5, Math.floor(k / 8) + 0.5]; };
    let svg = arrows.map(({ from, to, color, opacity = 0.85, width = 0.18 }) => {
      const [[x1, y1], [x2, y2]] = [from, to].map(at), hw = width * 1.4;
      const a = Math.atan2(y2 - y1, x2 - x1), ex = x2 - 0.3 * Math.cos(a), ey = y2 - 0.3 * Math.sin(a);
      return `<g opacity="${opacity}"><line x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}" stroke="${color}" stroke-width="${width}" />
        <polygon fill="${color}" points="${x2},${y2} ${ex + hw * Math.sin(a)},${ey - hw * Math.cos(a)} ${ex - hw * Math.sin(a)},${ey + hw * Math.cos(a)}" /></g>`;
    }).join('');
    if (badge) { // move verdict in the corner of the destination square
      const [cx, cy] = at(badge.sq), bx = Math.min(7.78, cx + 0.32), by = Math.max(0.22, cy - 0.32);
      svg += `<circle cx="${bx}" cy="${by}" r="0.21" fill="${badge.color}" stroke="#111" stroke-width="0.02" />
        <text x="${bx}" y="${by}" font-size="0.22" font-weight="800" text-anchor="middle" dominant-baseline="central" fill="#111">${badge.text}</text>`;
    }
    el.insertAdjacentHTML('beforeend', `<svg class="arrow" viewBox="0 0 8 8" aria-hidden="true">${svg}</svg>`);
  }

  function render() {
    const blue = '#3b82f6';
    drawBoard(boardEl, editing || piecesOf(chess), {
      flip: $('side').value === 'b', hl: highlight,
      arrows: editing ? [] : [...lines].reverse().map((ln, j, all) => j === all.length - 1
        ? { ...ln, color: blue } : { ...ln, color: blue, opacity: 0.4, width: 0.12 }), // best drawn last, on top
      onSquare: editing ? i => { editing[i] = editing[i] === brush ? '.' : brush; render(); } : null,
    });
  }

  // engine
  const SF = 'https://cdnjs.cloudflare.com/ajax/libs/stockfish.js/10.0.2/stockfish.js';
  const engine = new Worker(URL.createObjectURL(new Blob([`importScripts('${SF}')`], { type: 'application/javascript' })));
  let searching = false, queued = false, pvs = [];
  engine.onmessage = e => {
    const line = String(e.data);
    if (line.startsWith('info')) { const p = parseInfo(line); if (p) pvs[p.k - 1] = p; return; }
    if (!line.startsWith('bestmove')) return;
    searching = false;
    if (queued) { queued = false; return analyse(); }
    const fen = chess.fen();
    lines = pvs.filter(Boolean).map(p => {
      const m = new Chess(fen).move({ from: p.uci.slice(0, 2), to: p.uci.slice(2, 4), promotion: p.uci[4] || 'q' });
      return m && { from: m.from, to: m.to, san: m.san, color: m.color, score: formatScore(p.score, chess.turn()) };
    }).filter(Boolean);
    if (!lines.length) return;
    const [b, ...alts] = lines;
    highlight = [b.from, b.to];
    bestEl.textContent = `Best for ${b.color === 'w' ? 'White' : 'Black'}: ${b.san}  (${b.score})`;
    $('lines').textContent = alts.map((l, j) => `${j + 2}. ${l.san} ${l.score}`).join('   ');
    if ($('speak').checked && window.speechSynthesis) {
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(sanToSpeech(b.san)));
    }
    render(); drawOverlay();
  };
  engine.postMessage('uci');
  engine.postMessage('setoption name MultiPV value 3');

  function analyse() {
    if (searching) { queued = true; return engine.postMessage('stop'); }
    searching = true; pvs = [];
    engine.postMessage('position fen ' + chess.fen());
    engine.postMessage('go movetime 1000');
  }

  // suggest only right after a move was seen on the board
  function changed(moved = false) {
    save('fen', chess.fen());
    if (moved) $('sync').hidden = true; // following moves again
    highlight = []; lines = []; bestEl.textContent = ''; $('lines').textContent = '';
    const side = $('side').value;
    if (chess.game_over()) bestEl.textContent = chess.in_checkmate() ? 'Checkmate.' : 'Draw.';
    else if (!moved) bestEl.textContent = 'Waiting for a move…';
    else if (side === 'both' || side === chess.turn()) { bestEl.textContent = 'Thinking…'; analyse(); }
    else bestEl.textContent = 'Waiting for opponent\'s move…';
    render(); drawOverlay();
  }

  // ---------- game recording & review ----------
  const freshGame = () => ({ date: Date.now(), moves: [] });
  let game = load('game') || freshGame();
  // Straightened picture of the board as the screen showed it (160px JPEG, ~8 KB), a8 at top-left.
  function boardShot(img) {
    const N = 160, c = document.createElement('canvas'); c.width = c.height = N;
    const g = c.getContext('2d'), out = g.createImageData(N, N), h = H();
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const [sx, sy] = project(h, (x + 0.5) * 8 / N, (y + 0.5) * 8 / N);
      const xi = Math.min(img.width - 1, Math.max(0, Math.round(sx))), yi = Math.min(img.height - 1, Math.max(0, Math.round(sy)));
      const p = (yi * img.width + xi) * 4, q = (y * N + x) * 4;
      out.data[q] = img.data[p]; out.data[q + 1] = img.data[p + 1]; out.data[q + 2] = img.data[p + 2]; out.data[q + 3] = 255;
    }
    g.putImageData(out, 0, 0);
    return c.toDataURL('image/jpeg', 0.7);
  }
  const noShots = g => ({ ...g, moves: g.moves.map(({ shot, ...m }) => m) });

  function logMove(fen, m, shot) {
    game.moves.push({ fen, san: m.san, uci: m.from + m.to + (m.promotion || ''), color: m.color, shot });
    if (!save('game', game)) { delete game.moves.at(-1).shot; save('game', game); } // storage full: keep the move, drop the picture
  }
  // keep the finished game (last 10, screenshots for the latest 3) and start recording a new one
  function archiveGame() {
    if (game.moves.length >= 2) {
      const result = chess.in_checkmate() ? (chess.turn() === 'w' ? '0-1' : '1-0') : chess.in_draw() || chess.in_stalemate() ? '½-½' : '*';
      const games = [{ ...game, side: $('side').value, result }, ...(load('games') || [])].slice(0, 10).map((g, i) => i < 3 ? g : noShots(g));
      if (!save('games', games)) save('games', games.map(noShots)); // storage full: keep the games, drop the pictures
    }
    game = freshGame(); save('game', game);
    listGames();
  }
  function listGames() {
    const sel = $('games'), keep = sel.value;
    sel.innerHTML = '';
    const opt = (v, t) => { const o = document.createElement('option'); o.value = v; o.textContent = t; sel.append(o); };
    opt('current', `Current game (${Math.ceil(game.moves.length / 2)} moves)`);
    (load('games') || []).forEach((g, i) => opt(i, `${g.title ? g.title + ' · ' : ''}${new Date(g.date).toLocaleString()} · ${Math.ceil(g.moves.length / 2)} moves · ${g.result}`));
    if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
  }
  $('review').addEventListener('toggle', listGames);

  // separate engine so reviewing never disturbs the live guide; top 2 moves so "only move" (Great) can be spotted
  let reviewer = null;
  const evalPos = fen => new Promise(resolve => {
    const c = new Chess(fen);
    if (c.in_checkmate()) return resolve({ cp: -10000, best: null, pv: [], second: null });
    if (c.game_over()) return resolve({ cp: 0, best: null, pv: [], second: null });
    if (!reviewer) {
      reviewer = new Worker(URL.createObjectURL(new Blob([`importScripts('${SF}')`], { type: 'application/javascript' })));
      reviewer.postMessage('uci');
      reviewer.postMessage('setoption name MultiPV value 2');
    }
    const lines = [];
    reviewer.onmessage = e => {
      const line = String(e.data);
      if (line.startsWith('info')) { const p = parseInfo(line); if (p) lines[p.k - 1] = p; }
      else if (line.startsWith('bestmove')) {
        const [a, b] = lines;
        resolve({ cp: a ? toCp(a.score) : 0, best: line.split(' ')[1], pv: a ? a.pv : [], second: b ? toCp(b.score) : null });
      }
    };
    reviewer.postMessage('position fen ' + fen);
    reviewer.postMessage('go movetime 1500'); // ~2 min for a 40-move game; shorter = wobblier verdicts
  });

  // verdict -> look (symbols and colours are EyeChess's own)
  // opening names: fetched once, built in ~1s; no file (e.g. opened as file://) = review without names
  let bookP = null;
  const loadBook = () => bookP ||= fetch('openings.tsv').then(r => r.ok ? r.text() : '').then(openingBook).catch(() => new Map());

  const VERDICT = {
    Book: { sym: '📖', col: '#c4a07a', say: 'a book move' },
    Brilliant: { sym: '!!', col: '#2dd4bf', say: 'brilliant' },
    Great: { sym: '!', col: '#60a5fa', say: 'a great move' },
    Best: { sym: '★', col: '#4ade80', say: 'the best move' },
    Excellent: { sym: '✓', col: '#86efac', say: 'excellent' },
    Good: { sym: '•', col: '#bef264', say: 'good' },
    Inaccuracy: { sym: '?!', col: '#facc15', say: 'an inaccuracy' },
    Mistake: { sym: '?', col: '#fb923c', say: 'a mistake' },
    Miss: { sym: '✗', col: '#f472b6', say: 'a miss' },
    Blunder: { sym: '??', col: '#f87171', say: 'a blunder' },
  };
  const BAD = ['Inaccuracy', 'Mistake', 'Miss', 'Blunder'];
  const moveOf = (fen, uci) => new Chess(fen).move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] || 'q' });
  const showEval = v => Math.abs(v) >= 10000 ? (v > 0 ? '+M' : '-M') : formatScore({ type: 'cp', v }, 'w');

  let reviewing = false, rv = null; // rv = { g, out, idx } while a review is open
  $('runReview').onclick = async () => {
    if (reviewing) return;
    const g = selectedGame();
    if (!g || !g.moves.length) { $('rstatus').textContent = 'No moves recorded in that game yet.'; return; }
    reviewing = true;
    $('rstatus').textContent = 'Loading opening names…';
    const book = await loadBook();
    const cache = new Map(), ev = async fen => { if (!cache.has(fen)) cache.set(fen, await evalPos(fen)); return cache.get(fen); };
    const out = [];
    let inBook = true, opening = null;
    for (let i = 0; i < g.moves.length; i++) {
      const m = g.moves[i];
      $('rstatus').textContent = `Analysing move ${i + 1} of ${g.moves.length}…`;
      const next = new Chess(m.fen); next.move(m.san);
      const fenAfter = next.fen(), before = await ev(m.fen), after = await ev(fenAfter);
      const prev = out[i - 1], prevBad = !!prev && prev.color !== m.color && BAD.includes(prev.label);
      const sacrifice = isSacrifice(m.fen, m.uci, after.pv[0], after.pv[1]);
      const verdict = reviewMove(before, after, m.uci, { prevBad, sacrifice });
      // book = every move so far is known opening theory (once out of book, stay out)
      inBook = inBook && book.has(posKey(fenAfter));
      if (inBook) { Object.assign(verdict, { label: 'Book', loss: 0, accuracy: 100 }); opening = book.get(posKey(fenAfter)) || opening; }
      out.push({
        ...m, ...verdict, fenAfter, opening: inBook ? opening : null,
        best: before.best && before.best !== '(none)' ? moveOf(m.fen, before.best) : null,
        bestLine: pvSan(m.fen, before.pv, 4), threat: pvSan(fenAfter, after.pv, 3),
        evalBefore: m.color === 'w' ? before.cp : -before.cp, evalW: m.color === 'w' ? -after.cp : after.cp,
      });
    }
    reviewing = false;
    $('rstatus').textContent = `${opening ? `Opening: ${opening}. ` : ''}Review ready. Step through with ◀ ▶ (or the arrow keys).`;
    rv = { g, out, idx: -1 };
    $('rview').hidden = false;
    summary();
    show(0);
  };

  function summary() {
    const { g, out } = rv;
    const acc = col => { const a = out.filter(m => m.color === col); return a.length ? Math.round(a.reduce((s, m) => s + m.accuracy, 0) / a.length) : '–'; };
    const who = col => (col === 'w' ? 'White' : 'Black') + (g.side === col ? ' (you)' : '');
    const rows = Object.entries(VERDICT).map(([k, v]) =>
      `<tr><td><span class="ic" style="color:${v.col}">${v.sym}</span>${k}</td>` +
      ['w', 'b'].map(c => `<td>${out.filter(m => m.color === c && m.label === k).length}</td>`).join('') + '</tr>').join('');
    $('rsum').innerHTML = `<tr><th></th><th>${who('w')}</th><th>${who('b')}</th></tr>` +
      `<tr class="acc"><td>Accuracy</td><td>${acc('w')}</td><td>${acc('b')}</td></tr>` + rows;
    // move list: number | white | black
    $('moves').innerHTML = '';
    out.forEach((m, i) => {
      if (m.color === 'w' || i === 0) {
        const num = document.createElement('span'); num.className = 'num'; num.textContent = m.fen.split(' ')[5] + '.';
        $('moves').append(num);
        if (m.color === 'b') $('moves').append(document.createElement('span')); // game started with Black to move
      }
      const b = document.createElement('button'), v = VERDICT[m.label];
      b.innerHTML = `<span class="ic" style="color:${v.col}">${v.sym}</span>${m.san}`;
      b.title = m.label;
      b.onclick = () => show(i);
      $('moves').append(b);
    });
  }

  function coachText(m) {
    const v = VERDICT[m.label], bad = BAD.includes(m.label);
    let t = `<b>${m.san}</b> is ${v.say}.`;
    if (m.san.includes('#')) t = `<b>${m.san}</b> — checkmate!`;
    else if (m.label === 'Brilliant') t += ' A real sacrifice: the material you give up comes back with interest.';
    else if (m.label === 'Great') t += ' The only move that holds your position — everything else was clearly worse.';
    else if (m.label === 'Miss') t += ' Your opponent had just slipped, and this lets them off the hook.';
    if (m.label === 'Book') return `${t}${m.opening ? ` <b>${m.opening}</b>.` : ''} <span class="alt">Evaluation ${showEval(m.evalW)}.</span>`;
    if (m.best && m.best.san !== m.san && !m.san.includes('#')) {
      if (bad) t += ` Better was <b>${m.best.san}</b>${m.bestLine.length > 1 ? ` (${m.bestLine.join(' ')})` : ''}.`;
      else if (m.label !== 'Brilliant') t += ` The engine slightly preferred <b>${m.best.san}</b>.`;
    }
    if (bad && m.threat.length) t += ` Now the opponent has <b>${m.threat[0]}</b>${m.threat.length > 1 ? ` (${m.threat.join(' ')})` : ''}.`;
    return `${t} <span class="alt">Evaluation ${showEval(m.evalBefore)} → ${showEval(m.evalW)}.</span>`;
  }

  function drawGraph() {
    const { out, idx } = rv, n = out.length;
    const pts = out.map((m, i) => `${((i + 1) / n * 100).toFixed(2)},${(40 - winPct(m.evalW) * 0.4).toFixed(2)}`);
    const dots = out.map((m, i) => BAD.includes(m.label) && m.label !== 'Inaccuracy'
      ? `<circle class="dot" cx="${((i + 1) / n * 100).toFixed(2)}" cy="${(40 - winPct(m.evalW) * 0.4).toFixed(2)}" r="1.2" fill="${VERDICT[m.label].col}" />` : '').join('');
    const x = ((idx + 1) / n * 100).toFixed(2);
    $('graph').innerHTML = `<polygon class="w" points="0,40 0,20 ${pts.join(' ')} 100,40" /><line class="mid" x1="0" y1="20" x2="100" y2="20" />` +
      `<line class="cur" x1="${x}" y1="0" x2="${x}" y2="40" />${dots}`;
  }

  function show(i) {
    if (!rv) return;
    const { g, out } = rv;
    rv.idx = i = Math.max(0, Math.min(out.length - 1, i));
    const m = out[i], v = VERDICT[m.label], flip = g.side === 'b';
    const played = moveOf(m.fen, m.uci);
    const arrows = [];
    if (m.best && m.best.san !== m.san && (BAD.includes(m.label) || m.label === 'Good')) arrows.push({ from: m.best.from, to: m.best.to, color: '#4ade80' });
    drawBoard($('rboard'), piecesOf(new Chess(m.fenAfter)), { flip, hl: [played.from, played.to], arrows, badge: { sq: played.to, color: v.col, text: v.sym } });
    // eval bar: light part = White's winning chance
    $('evalbar').classList.toggle('flip', flip);
    $('evalfill').style.height = winPct(m.evalW).toFixed(1) + '%';
    $('evaltxt').textContent = showEval(m.evalW);
    $('coach').innerHTML = `<span class="badge" style="background:${v.col}">${v.sym}</span><p>${coachText(m)}</p>`;
    [...$('moves').querySelectorAll('button')].forEach((b, j) => b.classList.toggle('on', j === i));
    $('moves').querySelectorAll('button')[i]?.scrollIntoView({ block: 'nearest' });
    drawGraph();
    $('rshotbox').hidden = !m.shot;
    if (m.shot) { $('rshot').src = m.shot; $('rshot').classList.toggle('flip', flip); }
  }

  $('rfirst').onclick = () => show(0);
  $('rprev').onclick = () => show(rv.idx - 1);
  $('rnext').onclick = () => show(rv.idx + 1);
  $('rlast').onclick = () => show(rv.out.length - 1);
  $('graph').addEventListener('click', e => {
    if (!rv) return;
    const r = $('graph').getBoundingClientRect();
    show(Math.round((e.clientX - r.left) / r.width * rv.out.length) - 1);
  });
  document.addEventListener('keydown', e => {
    if (!rv || !$('review').open || e.target.matches('input, select, textarea')) return;
    const go = { ArrowLeft: rv.idx - 1, ArrowRight: rv.idx + 1, Home: 0, End: rv.out.length - 1 }[e.key];
    if (go != null) { e.preventDefault(); show(go); }
  });

  // ---------- PGN in / out ----------
  const selectedGame = () => { const sel = $('games').value; return sel === 'current' ? { ...game, side: $('side').value } : (load('games') || [])[+sel]; };
  function exportPgn() {
    const g = selectedGame();
    if (!g || !g.moves.length) { $('rstatus').textContent = 'No moves recorded in that game yet.'; return null; }
    const r = gameToPgn(g);
    if (!r.complete) $('rstatus').textContent = 'Note: EyeChess lost track partway through this game, so the PGN stops there.';
    return r;
  }
  $('pgnCopy').onclick = async () => {
    const r = exportPgn(); if (!r) return;
    try { await navigator.clipboard.writeText(r.pgn); if (r.complete) $('rstatus').textContent = 'PGN copied to the clipboard.'; }
    catch { $('pgnBox').open = true; $('pgnIn').value = r.pgn; $('rstatus').textContent = 'Couldn\'t reach the clipboard: the PGN is in the box below, copy it from there.'; }
  };
  $('pgnSave').onclick = () => {
    const r = exportPgn(); if (!r) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([r.pgn + '\n'], { type: 'application/x-chess-pgn' }));
    a.download = `eyechess-${new Date(selectedGame().date || Date.now()).toISOString().slice(0, 10)}.pgn`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $('pgnFile').onchange = async () => { const f = $('pgnFile').files[0]; if (f) $('pgnIn').value = await f.text(); };
  $('pgnLoad').onclick = () => {
    const g = gameFromPgn($('pgnIn').value);
    if (!g || !g.moves.length) { $('rstatus').textContent = 'Couldn\'t read that PGN. Paste the whole game, including the moves.'; return; }
    const h = g.headers, title = h.White || h.Black ? `${h.White || '?'} vs ${h.Black || '?'}` : 'Imported game';
    save('games', [{ date: Date.now(), title, moves: g.moves, side: $('side').value, result: g.result }, ...(load('games') || [])].slice(0, 10));
    listGames(); $('games').value = '0'; $('pgnBox').open = false;
    $('runReview').click();
  };

  listGames();

  changed();
  status(corners ? (refs ? 'Ready. Make a move, then Scan.' : 'Press Calibrate.') : 'Set up the start position under the camera, then press ▶ Live (or Auto setup).');
})();
