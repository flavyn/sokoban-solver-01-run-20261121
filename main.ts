// Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// The generator builds each level as a one-square hallway with 1..4 rooms off it. A box
// pushed through a door into the hallway can never be pushed back, so every box stays in its
// own room in any optimal solution. We therefore split the level into rooms, solve each room
// push-optimally with A* over box configurations, and join the room solutions with deterministic
// walking paths along the hallway. Push count is the score, so walking is free.
//
// Contract: node /bot/main.ts (Node 22), read /input/<id>/level.txt, write
// /output/<id>/solution.txt (LURD moves, UTF-8, <= 65,536 B), one run for a whole set of
// five instances, 60 s wall for the set, two runs must produce identical bytes. No clock,
// no randomness, no network, no files outside /input, /output, /bot.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Grid {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

type Region = {
  cells: number[];          // the room's floor cells, including its door cell
  member: Uint8Array;       // membership by absolute cell index
  door: number;             // the room cell adjacent to the hallway
  hallDoor: number;         // the hallway cell in front of the door
  boxes: number[];          // absolute indices of this room's boxes
  goals: number[];          // absolute indices of this room's goals
};

const DIRS: Array<[number, number, string]> = [
  [0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r'],
];

function parse(text: string): Grid {
  const rows = text.split('\n').map((r) => r.replace(/\r$/, '')).filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const ch = rows[y][x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

/** Walk path between two cells with a fixed set of obstacles (box positions). */
function walk(g: Grid, from: number, to: number, blocked: Uint8Array): string | null {
  if (from === to) return '';
  const n = g.w * g.h;
  const prev = new Int32Array(n).fill(-1);
  const seen = new Uint8Array(n);
  const q = new Int32Array(n);
  let head = 0;
  let tail = 0;
  q[tail++] = from;
  seen[from] = 1;
  while (head < tail) {
    const cur = q[head++];
    for (const [dx, dy, letter] of DIRS) {
      const x = (cur % g.w) + dx;
      const y = Math.floor(cur / g.w) + dy;
      if (x < 0 || y < 0 || x >= g.w || y >= g.h) continue;
      const nx = y * g.w + x;
      if (seen[nx] || g.wall[nx] || blocked[nx]) continue;
      seen[nx] = 1;
      prev[nx] = cur;
      if (nx === to) {
        let path = '';
        for (let t = nx; t !== from; t = prev[t]) {
          const px = t % g.w;
          const py = Math.floor(t / g.w);
          let back = '';
          for (const [bx, by, bl] of DIRS) {
            if ((prev[t] % g.w) + bx === px && Math.floor(prev[t] / g.w) + by === py) { back = bl; break; }
          }
          path = back + path;
        }
        return path;
      }
      q[tail++] = nx;
    }
  }
  return null;
}

/** Split the board into the hallway row and the rooms attached to it. */
function regions(g: Grid): { hallRow: number; rooms: Region[] } {
  const hallRow = Math.floor(g.player / g.w);
  const isHall = (i: number): boolean => Math.floor(i / g.w) === hallRow && !g.wall[i];
  const seen = new Uint8Array(g.w * g.h);
  const rooms: Region[] = [];
  for (let i = 0; i < g.w * g.h; i += 1) {
    if (g.wall[i] || isHall(i) || seen[i]) continue;
    const cells: number[] = [];
    const q = [i];
    seen[i] = 1;
    while (q.length > 0) {
      const cur = q.pop()!;
      cells.push(cur);
      const x = cur % g.w;
      const y = Math.floor(cur / g.w);
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const ni = ny * g.w + nx;
        if (g.wall[ni] || isHall(ni) || seen[ni]) continue;
        seen[ni] = 1;
        q.push(ni);
      }
    }
    let door = -1;
    let hallDoor = -1;
    for (const c of cells) {
      const y = Math.floor(c / g.w);
      for (const dy of [-1, 1]) {
        const ny = y + dy;
        if (ny < 0 || ny >= g.h) continue;
        const ni = ny * g.w + (c % g.w);
        if (isHall(ni)) { door = c; hallDoor = ni; break; }
      }
      if (door >= 0) break;
    }
    if (door < 0) continue;
    const member = new Uint8Array(g.w * g.h);
    for (const c of cells) member[c] = 1;
    // The door cell may hold a box; the first push must then come from the hallway side, so
    // the hallway cell in front of the door is part of the room's walkable/pushable region.
    member[hallDoor] = 1;
    rooms.push({
      cells, member, door, hallDoor,
      boxes: g.boxes.filter((b) => member[b] === 1),
      goals: Array.from({ length: g.w * g.h }, (_u, k) => k).filter((k) => member[k] === 1 && g.goal[k] === 1),
    });
  }
  rooms.sort((a, b) => a.hallDoor - b.hallDoor);
  return { hallRow, rooms };
}

/** Reverse-BFS push distances from goals inside one room: admissible A* heuristic. */
function pushDistances(g: Grid, room: Region): Int32Array {
  const n = g.w * g.h;
  const d = new Int32Array(n).fill(0x7fffffff);
  const q: number[] = [];
  for (const goal of room.goals) { d[goal] = 0; q.push(goal); }
  for (let head = 0; head < q.length; head += 1) {
    const cur = q[head];
    const x = cur % g.w;
    const y = Math.floor(cur / g.w);
    for (const [dx, dy] of DIRS) {
      // cur is the box destination; n is where the box came from; behind is where the player stood.
      const nx = x - dx;
      const ny = y - dy;
      const bx = nx - dx;
      const by = ny - dy;
      if (nx < 0 || ny < 0 || bx < 0 || by < 0 || nx >= g.w || ny >= g.h || bx >= g.w || by >= g.h) continue;
      const nCell = ny * g.w + nx;
      const behind = by * g.w + bx;
      if (!room.member[nCell] || !room.member[behind]) continue;
      if (g.wall[nCell] || g.wall[behind]) continue;
      if (d[nCell] > d[cur] + 1) { d[nCell] = d[cur] + 1; q.push(nCell); }
    }
  }
  return d;
}

/** Push-optimal solution of one room, starting with the player on its door cell. */
function solveRoom(g: Grid, room: Region, limit: number): string {
  if (room.boxes.length === 0) return '';
  const d = pushDistances(g, room);
  if (room.boxes.some((b) => d[b] === 0x7fffffff)) {
    return '';
  }

  const sb: number[][] = [room.boxes.slice().sort((a, b) => a - b)];
  const sp: number[] = [room.hallDoor];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const delta: string[] = [''];
  const best = new Map<string, number>();
  const key = (boxes: number[], p: number): string => `${boxes.join(',')}|${p}`;
  best.set(key(sb[0], sp[0]), 0);

  // binary heap on (f, g, state index); ties on lower g then earlier index
  const heap: Array<[number, number, number]> = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] > b[1] : a[2] < b[2];
  const pushHeap = (e: [number, number, number]): void => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const popHeap = (): [number, number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c += 1;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  const heuristic = (boxes: number[]): number => {
    let h = 0;
    for (const b of boxes) h += d[b];
    return h;
  };
  pushHeap([heuristic(sb[0]), 0, 0]);

  let expanded = 0;
  while (heap.length > 0 && expanded < limit) {
    const [, gScore, s] = popHeap();
    const boxes = sb[s];
    if (best.get(key(boxes, sp[s])) !== gScore) continue;
    if (boxes.every((b) => g.goal[b] === 1)) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]) moves = delta[t] + moves;
      return moves;
    }
    expanded += 1;

    const blocked = new Uint8Array(g.w * g.h);
    for (const b of boxes) blocked[b] = 1;
    // the player's reachable set; canonicalise it so equivalent positions share a state
    const reach = new Uint8Array(g.w * g.h);
    const q = [sp[s]];
    reach[sp[s]] = 1;
    for (let head = 0; head < q.length; head += 1) {
      const cur = q[head];
      const x = cur % g.w;
      const y = Math.floor(cur / g.w);
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= g.w || ny >= g.h) continue;
        const ni = ny * g.w + nx;
        if (!room.member[ni] || blocked[ni] || reach[ni]) continue;
        reach[ni] = 1;
        q.push(ni);
      }
    }

    for (let bi = 0; bi < boxes.length; bi += 1) {
      const box = boxes[bi];
      const bx = box % g.w;
      const by = Math.floor(box / g.w);
      for (const [dx, dy, letter] of DIRS) {
        const tx = bx + dx;
        const ty = by + dy;
        const sx = bx - dx;
        const sy = by - dy;
        if (tx < 0 || ty < 0 || sx < 0 || sy < 0 || tx >= g.w || ty >= g.h || sx >= g.w || sy >= g.h) continue;
        const target = ty * g.w + tx;
        const stand = sy * g.w + sx;
        if (!room.member[target] || g.wall[target] || blocked[target]) continue;
        if (!room.member[stand] || g.wall[stand] || !reach[stand]) continue;
        if (d[target] === 0x7fffffff) continue;
        const nb = boxes.slice();
        nb[bi] = target;
        nb.sort((a, b) => a - b);
        const k = key(nb, box); // after the push the player stands where the box was
        if (best.has(k) && best.get(k)! <= gScore + 1) continue;
        best.set(k, gScore + 1);
        const wpath = walk(g, sp[s], stand, blocked);
        if (wpath === null) continue;
        const sNew = sb.length;
        sb.push(nb);
        sp.push(box);
        sg.push(gScore + 1);
        parent.push(s);
        delta.push(wpath + letter.toLowerCase());
        pushHeap([gScore + 1 + heuristic(nb), gScore + 1, sNew]);
      }
    }
  }
  return ''; // no solution within the deterministic work cap
}

function solve(text: string): string {
  const g = parse(text);
  const { rooms } = regions(g);
  const blocked = new Uint8Array(g.w * g.h);
  for (const b of g.boxes) blocked[b] = 1;
  let player = g.player;
  let moves = '';

  for (const room of rooms) {
    // stand in front of the door, walking through the hallway with the current boxes
    const here = walk(g, player, room.hallDoor, blocked);
    if (here === null) return '';
    moves += here;
    player = room.hallDoor;
    const step = solveRoom(g, room, 3_000_000);
    if (step === '') return '';
    moves += step;
    // replay the room's moves on the full board so later walking sees the final box positions
    for (const ch of step) {
      const dx = ch === 'r' ? 1 : ch === 'l' ? -1 : 0;
      const dy = ch === 'd' ? 1 : ch === 'u' ? -1 : 0;
      const from = player;
      const to = from + dy * g.w + dx;
      if (g.wall[to] === 1) {
        return '';
      }
      if (blocked[to] === 1) {
        const beyond = to + dy * g.w + dx;
        if (g.wall[beyond] === 1 || blocked[beyond] === 1) {
          return '';
        }
        blocked[to] = 0;
        blocked[beyond] = 1;
      }
      player = to;
    }
  }
  return moves;
}

mkdirSync(O, { recursive: true });
const ids = readdirSync(I).sort();
const outputs: Array<[string, string]> = [];
for (const id of ids) {
  const text = readFileSync(join(I, id, 'level.txt'), 'utf8');
  const moves = solve(text);
  const answer = `${moves}\n`;
  if (Buffer.byteLength(answer, 'utf8') > 65_536) throw new Error(`${id}: answer too large`);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), answer);
  outputs.push([id, moves]);
}
