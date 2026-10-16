"""Unified Multi-Target BBox-A* Shortest Path Tree Search.

Combines:
1. Multi-target SPT (Single search from driver finding all sinks together)
2. BBox restriction (XY bounding box + margin)
3. 3D layer-aware A* heuristic directed towards remaining sinks
"""
import time
import heapq
from m3d.model import Instance
from m3d.grid import Grid, edge_key
from m3d.checker import check
from experiments.staged_router import StagedRouter, Tree

inst = Instance.load("benchmarks_hard/case_01.json")
sr = StagedRouter(inst, initial_route="negotiated", polish_passes=2,
                  layer_opt=False, via_opt=False, final_opt=False)
sub, _ = sr.run()
init_delay = check(inst, sub).total_delay
print(f"Initial delay: {init_delay}")

g = sr.regions.grid
r = sr.regions
min_ld = int(min(inst.layer_delay))
via_del = int(inst.via_delay)

def fast_spt_bbox_astar(sr, nid, margin=3):
    """Multi-target goal-directed SPT inside BBox in a single search pass."""
    source, *sinks = r.net_pins[nid]
    if not sinks:
        return Tree({source}, set(), 0)

    all_pins = [source] + sinks
    coords = [g.coord(v) for v in all_pins]
    xmin = max(0, min(c[0] for c in coords) - margin)
    xmax = min(inst.width - 1, max(c[0] for c in coords) + margin)
    ymin = max(0, min(c[1] for c in coords) - margin)
    ymax = min(inst.height - 1, max(c[1] for c in coords) + margin)

    targets = set(sinks)
    target_coords = [g.coord(v) for v in targets]

    # Precalculate heuristic towards ANY remaining sink
    def h(vid):
        x, y, z = g.coord(vid)
        return min((abs(x-sx)+abs(y-sy))*min_ld + abs(z-sz)*via_del
                   for sx, sy, sz in target_coords)

    dist = {source: 0}
    parent = {}
    heap = [(h(source), 0, source)]
    found = []

    while heap and len(found) < len(sinks):
        prio, d, u = heapq.heappop(heap)
        if d != dist.get(u):
            continue
        if u in targets:
            found.append(u)
            if len(found) == len(sinks):
                break
        for v, delay in g.neighbors(u):
            if r.pin_owner.get(v, nid) != nid:
                continue
            if r.owner[v] not in (None, nid):
                continue
            vx, vy, _ = g.coord(v)
            if vx < xmin or vx > xmax or vy < ymin or vy > ymax:
                continue
            nd = d + delay
            if nd < dist.get(v, 1000000000):
                dist[v], parent[v] = nd, u
                heapq.heappush(heap, (nd + h(v), nd, v))

    if len(found) < len(sinks):
        # Margin was too tight; expand margin
        if margin < 8:
            return fast_spt_bbox_astar(sr, nid, margin=margin + 4)
        return None

    # Construct the Shortest Path Tree from the driver to all sinks
    vertices = {source}
    edges = set()
    for sink in found:
        cur = sink
        while cur not in vertices:
            p = parent[cur]
            edges.add(edge_key(p, cur))
            vertices.add(cur)
            cur = p

    tree = Tree(vertices, edges, 0)
    tree.delay = sr.tree_delay(nid, tree)
    return tree

# -------------------------------------------------------------
# BENCHMARK: Run Pairwise EDA Subsetting with fast_spt_bbox_astar
# -------------------------------------------------------------
t0 = time.time()
rounds = 3
for rnd in range(1, rounds + 1):
    gaps = []
    for nid, tree in sr.trees.items():
        lb = sr.lower_bound(nid)
        gap = tree.delay - lb
        if gap > 0:
            gaps.append((gap, nid))
    gaps.sort(reverse=True)

    net_verts = {nid: set(sr.trees[nid].vertices) for nid in sr.trees}
    def net_bbox(nid):
        coords = [g.coord(v) for v in net_verts[nid]]
        return (min(c[0] for c in coords), max(c[0] for c in coords),
                min(c[1] for c in coords), max(c[1] for c in coords))

    bboxes = {nid: net_bbox(nid) for nid in sr.trees}
    seen_pairs = set()
    candidate_pairs = []

    for gap, n1 in gaps:
        b1 = bboxes[n1]
        for n2 in sr.trees:
            if n2 == n1:
                continue
            pair_key = (min(n1, n2), max(n1, n2))
            if pair_key in seen_pairs:
                continue
            b2 = bboxes[n2]
            if not (b1[1] < b2[0] or b1[0] > b2[1] or b1[3] < b2[2] or b1[2] > b2[3]):
                seen_pairs.add(pair_key)
                pair_crit = (sr.trees[n1].delay - sr.lower_bound(n1)) + (sr.trees[n2].delay - sr.lower_bound(n2))
                candidate_pairs.append((pair_crit, n1, n2))

    candidate_pairs.sort(reverse=True)

    accepted = 0
    for crit, n1, n2 in candidate_pairs:
        old1 = sr.remove(n1)
        old2 = sr.remove(n2)
        old_cost = old1.delay + old2.delay

        best = None
        best_cost = old_cost

        # Order 1: n1 then n2
        c1 = fast_spt_bbox_astar(sr, n1)
        if c1:
            sr.commit(n1, c1)
            c2 = fast_spt_bbox_astar(sr, n2)
            if c2 and (c1.delay + c2.delay) < best_cost:
                best = (c1, c2)
                best_cost = c1.delay + c2.delay
            sr.remove(n1)

        # Order 2: n2 then n1
        c2 = fast_spt_bbox_astar(sr, n2)
        if c2:
            sr.commit(n2, c2)
            c1 = fast_spt_bbox_astar(sr, n1)
            if c1 and (c1.delay + c2.delay) < best_cost:
                best = (c1, c2)
                best_cost = c1.delay + c2.delay
            sr.remove(n2)

        if best:
            sr.commit(n1, best[0])
            sr.commit(n2, best[1])
            accepted += 1
            net_verts[n1] = set(best[0].vertices)
            net_verts[n2] = set(best[1].vertices)
            bboxes[n1] = net_bbox(n1)
            bboxes[n2] = net_bbox(n2)
        else:
            sr.commit(n1, old1)
            sr.commit(n2, old2)

    # Interleaved single-net polish
    sr.polish()
    total_d = sum(t.delay for t in sr.trees.values())
    print(f"Round {rnd}: tested={len(candidate_pairs)} accepted={accepted} -> total_delay={total_d}")

dt = time.time() - t0
sub = sr.current_submission()
res = check(inst, sub)
print(f"\nFinal Result: legal={res.legal} total_delay={res.total_delay} baseline=11938 ratio={11938/res.total_delay:.4f} time={dt:.2f}s")

