"""Build whole-owner LNS groups from sink-weighted trunk shortcut corridors.

Corridors propose neighborhoods, not routes. Exact unrestricted SPT bounds and
joint repair decide whether a complete replacement is useful and legal.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time

from m3d.checker import check


@dataclass(frozen=True)
class Neighborhood:
    nets: tuple[int, ...]
    seed: int
    endpoint: int
    sinks_downstream: int
    proposed_trunk_saving: int
    layer: int
    operator: str = "trunk_corridor"


def rooted_tree(sr, nid):
    """Return rooted distances and the exact downstream sink multiplicities."""
    tree = sr.trees[nid]
    source, *sinks = sr.regions.net_pins[nid]
    adj = {v: [] for v in tree.vertices}
    for a, b in tree.edges:
        adj[a].append(b)
        adj[b].append(a)
    parent, distance, order = {source: None}, {source: 0}, [source]
    queue = deque([source])
    g = sr.regions.grid
    while queue:
        u = queue.popleft()
        for v in sorted(adj[u]):
            if v in parent:
                continue
            parent[v] = u
            distance[v] = distance[u] + sr.inst.edge_delay(g.coord(u), g.coord(v))
            order.append(v)
            queue.append(v)
    sink_counts = Counter(sinks)
    weights = {v: sink_counts[v] for v in order}
    for v in reversed(order[1:]):
        weights[parent[v]] += weights[v]
    return distance, weights


def shortcut_corridor(g, source, endpoint, layer, xy_order):
    """Two vias plus an XY elbow: yield vertices and exact path delay."""
    current = list(g.coord(source))
    target = g.coord(endpoint)
    vertices = [source]
    delay = 0
    for axis, value in [(2, layer), (xy_order[0], target[xy_order[0]]),
                        (xy_order[1], target[xy_order[1]]), (2, target[2])]:
        while current[axis] != value:
            current[axis] += 1 if value > current[axis] else -1
            vertices.append(g.vid(tuple(current)))
            delay += g.via_delay if axis == 2 else g.layer_delay[current[2]]
    return vertices, delay


def corridor_neighborhoods(sr, max_group=6, endpoints_per_net=12):
    """Release every wire owner on a promising shortcut; foreign pins forbid it.

    This heuristic score is not a lower bound on group improvement: blockers
    may lose more than the trunk gains. repair_group computes the safe bound.
    """
    r, g = sr.regions, sr.regions.grid
    proposals = {}
    for nid in sorted(sr.trees):
        source = r.net_pins[nid][0]
        distance, weights = rooted_tree(sr, nid)
        source_xyz = g.coord(source)
        endpoints = []
        for v, d in distance.items():
            if v == source or not weights[v]:
                continue
            xyz = g.coord(v)
            xy = abs(xyz[0] - source_xyz[0]) + abs(xyz[1] - source_xyz[1])
            ideal = min(xy * sr.inst.layer_delay[z]
                        + (abs(source_xyz[2] - z) + abs(xyz[2] - z)) * sr.inst.via_delay
                        for z in range(sr.inst.layers))
            gap = weights[v] * (d - ideal)
            if gap > 0:
                endpoints.append((gap, weights[v], v))
        endpoints.sort(reverse=True)
        for _, weight, endpoint in endpoints[:endpoints_per_net]:
            for layer in range(sr.inst.layers):
                for order in [(0, 1), (1, 0)]:
                    corridor, cost = shortcut_corridor(g, source, endpoint, layer, order)
                    saving = weight * (distance[endpoint] - cost)
                    if saving <= 0:
                        continue
                    if any(r.pin_owner.get(v, nid) != nid for v in corridor):
                        continue
                    owners = {r.owner[v] for v in corridor if r.owner[v] is not None}
                    owners.add(nid)
                    if not 2 <= len(owners) <= max_group:
                        continue
                    group = tuple(sorted(owners))
                    proposal = Neighborhood(group, nid, endpoint, weight, saving, layer)
                    # Prefer large potential savings, but spread the budget across group sizes.
                    score = saving / (len(group) ** 0.5)
                    if group not in proposals or score > proposals[group][0]:
                        proposals[group] = (score, proposal)
    return [proposal for _, proposal in sorted(proposals.values(),
            key=lambda item: (-item[0], item[1].nets))]


def install_group(sr, candidates):
    """Commit a complete group atomically and independently check its result."""
    group = sorted(candidates)
    original = {nid: sr.trees[nid] for nid in group}
    old_delay = sum(tree.delay for tree in original.values())
    try:
        for nid in group:
            sr.remove(nid)
        for nid in group:
            candidate = candidates[nid]
            if candidate.delay != sr.tree_delay(nid, candidate):
                raise RuntimeError("candidate delay differs from its actual rooted delay")
            sr.commit(nid, candidate)
        result = check(sr.inst, sr.current_submission())
        gain = old_delay - sum(sr.trees[nid].delay for nid in group)
        if not result.legal or gain <= 0:
            raise RuntimeError("group replacement was illegal or did not improve delay")
        return gain
    except BaseException:
        for nid in group:
            if nid in sr.trees:
                sr.remove(nid)
        for nid in group:
            sr.commit(nid, original[nid])
        raise


def improve(sr, budget_s=60.0, group_budget_s=2.0, max_nodes=200,
            max_group=6, max_groups=100, on_accept=None, conflict_avoidance=False):
    """Run bounded exact joint repairs; retain the incumbent at every failure."""
    from experiments.lns_repair import repair_group

    started = time.perf_counter()
    initial = sum(tree.delay for tree in sr.trees.values())
    attempts, accepted, tried = [], 0, set()
    proposals = corridor_neighborhoods(sr, max_group=max_group)
    proposed_count = len(proposals)
    while proposals and len(attempts) < max_groups:
        remaining = budget_s - (time.perf_counter() - started)
        if remaining <= 0:
            break
        proposal = proposals.pop(0)
        if proposal.nets in tried:
            continue
        tried.add(proposal.nets)
        repair_options = dict(max_nodes=max_nodes,
                              time_budget_s=min(group_budget_s, remaining))
        if conflict_avoidance:
            repair_options['conflict_avoidance'] = True
        repair = repair_group(sr, proposal.nets, **repair_options)
        # Keep the experiment runner decoupled from the engine's search internals.
        candidates = repair['candidates']
        record = dict(repair['stats'])
        record.update(nets=list(proposal.nets), seed=proposal.seed,
                      sinks_downstream=proposal.sinks_downstream,
                      proposed_trunk_saving=proposal.proposed_trunk_saving,
                      operator=proposal.operator, accepted_gain=0)
        if candidates is not None:
            gain = install_group(sr, candidates)
            record['accepted_gain'] = gain
            accepted += 1
            if on_accept:
                on_accept(sr, record)
            # Outside occupancy has changed; reevaluate even previously failed groups.
            tried.clear()
            proposals = corridor_neighborhoods(sr, max_group=max_group)
        attempts.append(record)
        print(f"LNS group={proposal.nets} bound={record.get('relaxed_bound')} "
              f"gain={record['accepted_gain']} reason={record.get('reason')}", flush=True)
    final = sum(tree.delay for tree in sr.trees.values())
    return dict(initial_delay=initial, final_delay=final, saving=initial-final,
                runtime_s=time.perf_counter()-started, accepted_groups=accepted,
                proposed_groups=proposed_count, attempted_groups=len(attempts),
                options=dict(budget_s=budget_s, group_budget_s=group_budget_s,
                             max_nodes=max_nodes, max_group=max_group, max_groups=max_groups,
                             conflict_avoidance=conflict_avoidance),
                attempts=attempts)
