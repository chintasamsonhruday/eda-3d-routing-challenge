"""Bounded, exact-cost conflict-based repair for small static net groups.

This is an experiment engine, not a replacement for the staged router. The
caller provides a legal incumbent and commits a returned improvement as a
transaction after independent checking. No router state is mutated here.

Outside wires and every foreign pin stay fixed. Inside a CBS node each net's
driver-root shortest-path tree minimizes the *sum of sink delays*, not wire
length. Conflicting vertices split into two constraints; this is static
vertex-disjoint routing, without the time dimension of multi-agent planning.
An exhaustive search proves optimality only inside this frozen neighborhood.
Under a time/node limit the original legal incumbent is preserved.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time

from m3d.grid import edge_key

try:  # Support both direct experiment CLIs and package imports in tests.
    from .staged_router import Tree
except ImportError:
    from staged_router import Tree


class _TimeBudget(Exception):
    pass


def _shortest_tree(grid, pins, forbidden, deadline, stats, avoidance=frozenset()):
    """One root Dijkstra, pruned to the union of settled terminal paths."""
    if time.perf_counter() >= deadline:
        raise _TimeBudget
    stats['searches'] += 1
    source, *sinks = pins
    targets = set(sinks) - {source}
    if source in forbidden or any(v in forbidden for v in targets):
        return None
    # Secondary overlap cost changes only equal-delay path choices. The primary
    # objective is always the exact delay, so the CBS bound remains admissible.
    distance, parent = {source: (0, 0)}, {}
    heap = [(0, 0, source)]
    expanded = 0
    while heap and targets:
        if expanded % 64 == 0 and time.perf_counter() >= deadline:
            raise _TimeBudget
        d, overlap, u = heapq.heappop(heap)
        if (d, overlap) != distance.get(u):
            continue
        expanded += 1
        stats['expanded_vertices'] += 1
        targets.discard(u)
        if not targets:
            break
        for v, weight in grid.neighbors(u):
            if v in forbidden:
                continue
            nd = d + weight
            next_overlap = overlap + int(v in avoidance)
            candidate = nd, next_overlap
            if candidate < distance.get(v, (math.inf, math.inf)):
                distance[v], parent[v] = candidate, u
                heapq.heappush(heap, (nd, next_overlap, v))
    if targets:
        return None
    vertices, edges = {source}, set()
    for sink in sinks:
        v = sink
        while v not in vertices:
            vertices.add(v)
            u = parent[v]
            edges.add(edge_key(u, v))
            v = u
    return Tree(vertices, edges, sum(distance[v][0] for v in sinks))


def _conflicts(group, trees):
    """Count overlaps and select one; vertex exclusion implies edge exclusion."""
    owner = {}
    count, first = 0, None
    for nid in group:
        for vertex in sorted(trees[nid].vertices):
            if vertex in owner:
                count += 1
                if first is None:
                    first = vertex, owner[vertex], nid
            owner[vertex] = nid
    return count, first


def repair_group(sr, group, max_nodes=200, time_budget_s=1.0,
                 conflict_avoidance=False):
    """Return ``{'candidates': mapping-or-None, 'stats': ...}`` without edits.

    ``group`` must contain 2--6 distinct IDs with existing legal ``sr.trees``.
    ``max_nodes`` counts popped CBS nodes, including the relaxed root.
    ``relaxed_bound`` is the sum of independently optimal root-tree delays with
    the group released; it is a lower bound, not an achievable saving. Search
    may strengthen it into ``search_lower_bound`` as queue nodes are exhausted.
    explores only nodes below the original incumbent cost. A conflict-free
    minimum-queue-cost node is an optimal neighborhood repair; reaching a budget
    returns no candidates and makes no optimality claim. ``result_delay`` is
    the incumbent delay unless a complete strict improvement is returned.
    Optional ``conflict_avoidance`` favors equal-delay root paths with fewer
    vertices on already computed group trees. It never prohibits those vertices
    or sacrifices primary delay, and its avoidance set is part of the cache key.
    """
    started = time.perf_counter()
    group = tuple(group)
    if not 2 <= len(group) <= 6 or len(set(group)) != len(group):
        raise ValueError('repair group must contain 2--6 distinct net IDs')
    if not isinstance(max_nodes, int) or max_nodes < 0:
        raise ValueError('max_nodes must be a nonnegative integer')
    if not math.isfinite(time_budget_s) or time_budget_s < 0:
        raise ValueError('time_budget_s must be finite and nonnegative')
    if any(nid not in sr.trees or nid not in sr.regions.net_pins for nid in group):
        raise ValueError('every group net must have an incumbent tree and pins')
    r = sr.regions
    grid = r.grid
    if min(grid.layer_delay) <= 0 or grid.via_delay <= 0:
        raise ValueError('shortest-path repair requires positive edge delays')
    incumbent = sum(sr.trees[nid].delay for nid in group)
    stats = dict(group=list(group), incumbent_delay=incumbent,
                 relaxed_bound=None, relaxed_net_delays=None,
                 search_lower_bound=None, max_possible_saving=None,
                 remaining_possible_saving=None, nodes=0, searches=0,
                 expanded_vertices=0, cache_hits=0, generated_nodes=0,
                 result_delay=incumbent, elapsed_s=0.0, time_s=0.0,
                 reason=None, optimal=False, exact=False, root_conflicts=None,
                 conflict_avoidance=bool(conflict_avoidance),
                 max_nodes=max_nodes, time_budget_s=time_budget_s)

    heap, active_bound = [], None

    def result(reason, candidates=None, exact=False):
        elapsed = time.perf_counter() - started
        stats.update(reason=reason, elapsed_s=elapsed, time_s=elapsed,
                     exact=exact, optimal=exact)
        if candidates is not None:
            stats['result_delay'] = sum(t.delay for t in candidates.values())
        if stats['relaxed_bound'] is not None:
            if exact:
                bound = stats['result_delay']
            else:
                # A node partly expanded when the clock expires still represents
                # unsearched solutions. Its popped cost must stay in the bound.
                frontier = [incumbent]
                if heap:
                    frontier.append(heap[0][0])
                if active_bound is not None:
                    frontier.append(active_bound)
                bound = min(frontier)
            stats['search_lower_bound'] = bound
            stats['remaining_possible_saving'] = max(0, incumbent - bound)
        return dict(candidates=candidates, stats=stats)

    if max_nodes == 0:
        return result('node_budget')
    if time_budget_s == 0:
        return result('time_budget')
    deadline = started + time_budget_s
    members = set(group)
    # Group wires are relaxed, but group members may never occupy other pins.
    outside = {v for v, nid in enumerate(r.owner)
               if nid is not None and nid not in members}
    base_forbidden = {
        nid: outside | {v for v, pin_nid in r.pin_owner.items() if pin_nid != nid}
        for nid in group}
    cache = {}

    def low_level(nid, constraints, other_trees=()):
        avoidance = (frozenset(v for tree in other_trees for v in tree.vertices)
                     if conflict_avoidance else frozenset())
        key = nid, constraints, avoidance
        if key in cache:
            stats['cache_hits'] += 1
            return cache[key]
        tree = _shortest_tree(grid, r.net_pins[nid],
                              base_forbidden[nid] | constraints, deadline, stats,
                              avoidance)
        cache[key] = tree
        return tree

    try:
        root = {}
        empty = frozenset()
        for nid in group:
            tree = low_level(nid, empty, root.values())
            if tree is None:
                # A legal incumbent cannot be blocked under a relaxation. This
                # flags caller state inconsistency instead of claiming infeasible.
                return result('relaxed_disconnected')
            root[nid] = tree
        lower = sum(t.delay for t in root.values())
        stats['relaxed_bound'] = lower
        stats['relaxed_net_delays'] = {nid: root[nid].delay for nid in group}
        stats['max_possible_saving'] = max(0, incumbent - lower)
        conflict_count, first_conflict = _conflicts(group, root)
        stats['root_conflicts'] = conflict_count
        if lower >= incumbent:
            return result('no_bound_gain', exact=True)

        constraints = tuple(empty for _ in group)
        serial = itertools.count()
        heap = [(lower, conflict_count, next(serial), constraints, root, first_conflict)]
        seen = {constraints}
        stats['generated_nodes'] = 1
        index = {nid: i for i, nid in enumerate(group)}
        while heap:
            if time.perf_counter() >= deadline:
                return result('time_budget')
            if stats['nodes'] >= max_nodes:
                return result('node_budget')
            cost, _, _, bans, trees, conflict = heapq.heappop(heap)
            active_bound = cost
            stats['nodes'] += 1
            if conflict is None:
                return result('improved', dict(trees), exact=True)
            vertex, a, b = conflict
            for nid in (a, b):
                i = index[nid]
                updated = bans[i] | {vertex}
                child_bans = bans[:i] + (updated,) + bans[i+1:]
                if child_bans in seen:
                    continue
                seen.add(child_bans)
                replacement = low_level(nid, updated,
                                        (t for other, t in trees.items() if other != nid))
                if replacement is None:
                    continue
                child_cost = cost - trees[nid].delay + replacement.delay
                if child_cost >= incumbent:
                    continue
                child_trees = dict(trees)
                child_trees[nid] = replacement
                count, first = _conflicts(group, child_trees)
                heapq.heappush(heap, (child_cost, count, next(serial), child_bans,
                                      child_trees, first))
                stats['generated_nodes'] += 1
            active_bound = None
        return result('no_improvement', exact=True)
    except _TimeBudget:
        return result('time_budget')
