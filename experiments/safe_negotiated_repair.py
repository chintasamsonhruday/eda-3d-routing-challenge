"""Guarded version of the experiments' edge-and-vertex mini PathFinder.

Search happens in private occupancy maps: failure, limits, or an exception
cannot remove a net from the live router. Only checked strict improvements
are committed transactionally. Guidance and physical delay stay separate.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import heapq
import json
import math
import os
from pathlib import Path
import time

from m3d.checker import check
from m3d.grid import edge_key
from m3d.model import NetRoute, Submission
from .staged_router import Tree
from .trunk_lns import install_group


class _Timeout(Exception):
    pass


def validate_nonnegative(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')


def build_pin_chimneys(inst):
    """Keep all claimants when different pins share an escape vertex."""
    from m3d.grid import Grid
    grid, owner = Grid(inst), inst.pin_to_net()
    result = defaultdict(set)
    for pin in inst.pins:
        z = pin.z + (1 if pin.die == 0 else -1)
        if 0 <= z < inst.layers:
            result[grid.vid((pin.x, pin.y, z))].add(owner[pin.id])
    return dict(result)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.2, .01 * 2 ** attempt))


@contextmanager
def _save_lock(path):
    """Serialize cooperative writers to the same case, including compare/save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + '.lock')
    deadline = time.perf_counter() + 5
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.perf_counter() >= deadline:
                raise TimeoutError(f'saved-route writer lock is busy: {lock}')
            time.sleep(.02)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def save_checked_best(inst, submission, path):
    """An illegal or worse experiment may never overwrite a legal saved route.

    Returns the actual stored result and whether this candidate was written.
    Callers must report that result rather than reporting a discarded candidate.
    """
    result = check(inst, submission)
    if not result.legal:
        raise ValueError(f'refusing to save illegal route: {result.reasons}')
    path = Path(path)
    with _save_lock(path):
        if path.exists():
            old = Submission.load(str(path))
            old_result = check(inst, old)
            if old_result.legal and old_result.total_delay <= result.total_delay:
                return old_result, False
        atomic_json(path, submission.to_dict())
        reloaded = check(inst, Submission.load(str(path)))
        if not reloaded.legal or reloaded.total_delay != result.total_delay:
            raise RuntimeError('saved route differs from checked candidate')
        return reloaded, True


def _priced_tree(sr, nid, members, ov, oe, hv, he, present, strict,
                 deadline, stats):
    r, grid = sr.regions, sr.regions.grid
    source, *sinks = r.net_pins[nid]
    targets = set(sinks) - {source}
    distance, parent, queue = {source: 0.0}, {}, [(0.0, source)]
    stats['searches'] += 1
    expanded = 0
    while queue and targets:
        if expanded % 128 == 0 and time.perf_counter() >= deadline:
            raise _Timeout
        cost, u = heapq.heappop(queue)
        if cost != distance.get(u):
            continue
        expanded += 1
        targets.discard(u)
        if not targets:
            break
        for v, delay in grid.neighbors(u):
            if r.pin_owner.get(v, nid) != nid:
                continue
            owner = r.owner[v]
            if owner is not None and owner not in members:
                continue
            edge = edge_key(u, v)
            if strict:
                if ov.get(v) or oe.get(edge):
                    continue
                step = delay
            else:
                # get() avoids creating millions of zero-price/empty entries
                # merely by exploring free resources on a large instance.
                step = delay * (1 + he.get(edge, 0) + present * len(oe.get(edge, ())))
                step += hv.get(v, 0) + present * len(ov.get(v, ()))
            candidate = cost + step
            if candidate < distance.get(v, math.inf):
                distance[v], parent[v] = candidate, u
                heapq.heappush(queue, (candidate, v))
    stats['expanded_vertices'] += expanded
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
    tree = Tree(vertices, edges, 0)
    tree.delay = sr.tree_delay(nid, tree)
    return tree


def _checked_group(sr, group, routes, full=False):
    if set(routes) != set(group):
        return None
    occupied = set()
    members = set(group)
    for nid in group:
        tree = routes[nid]
        if (not set(sr.regions.net_pins[nid]) <= tree.vertices
                or tree.delay != sr.tree_delay(nid, tree)
                or occupied & tree.vertices
                or any(sr.regions.pin_owner.get(v, nid) != nid
                       or sr.regions.owner[v] not in (None, nid)
                       and sr.regions.owner[v] not in members for v in tree.vertices)):
            return None
        occupied.update(tree.vertices)
    if not full:
        return True
    grid = sr.regions.grid
    combined = dict(sr.trees)
    combined.update(routes)
    submission = Submission(sr.inst.name, [NetRoute(nid,
        [(grid.coord(a), grid.coord(b)) for a, b in sorted(tree.edges)])
        for nid, tree in sorted(combined.items())])
    result = check(sr.inst, submission)
    return result if result.legal else None


def mini_pathfinder(sr, group, max_iters=14, *, time_budget_s=None, stats=None):
    """Preserve the existing experimental cost; return accepted gain or None."""
    started = time.perf_counter()
    group = tuple(group)
    if len(set(group)) != len(group) or not group or any(n not in sr.trees for n in group):
        raise ValueError('group must contain distinct existing routed net IDs')
    if not isinstance(max_iters, int) or max_iters < 0:
        raise ValueError('max_iters must be a nonnegative integer')
    if time_budget_s is not None:
        validate_nonnegative(time_budget_s=time_budget_s)
    if min(sr.inst.layer_delay) <= 0 or sr.inst.via_delay <= 0:
        raise ValueError('routing requires strictly positive physical delays')
    deadline = math.inf if time_budget_s is None else started + time_budget_s
    info = {} if stats is None else stats
    original_delay = sum(sr.trees[nid].delay for nid in group)
    info.update(reason=None, iterations=0, searches=0, expanded_vertices=0,
                incumbent_delay=original_delay, result_delay=original_delay,
                gain=0, overuse_history=[])
    members = set(group)
    ov, oe, hv, he, routes = {}, {}, defaultdict(float), defaultdict(float), {}

    def finish(reason, gain=None):
        info.update(reason=reason, seconds=time.perf_counter()-started)
        return gain

    def add(nid, tree):
        routes[nid] = tree
        for v in tree.vertices:
            ov.setdefault(v, set()).add(nid)
        for edge in tree.edges:
            oe.setdefault(edge, set()).add(nid)

    def remove(nid):
        tree = routes.pop(nid)
        for mapping, resources in ((ov, tree.vertices), (oe, tree.edges)):
            for resource in resources:
                mapping[resource].remove(nid)
                if not mapping[resource]:
                    del mapping[resource]
        return tree

    def route(nid, present, strict=False):
        if time.perf_counter() >= deadline:
            raise _Timeout
        return _priced_tree(sr, nid, members, ov, oe, hv, he,
                            present, strict, deadline, info)

    try:
        for nid in group:
            tree = route(nid, .2)
            if tree is None:
                return finish('initial_route_failed')
            add(nid, tree)
        present = .3
        # Inspect the final rerouting round as well; the old loop dropped legal
        # solutions that emerged exactly at its iteration limit.
        for iteration in range(max_iters + 1):
            info['iterations'] = iteration
            if set(routes) != members:
                return finish('incomplete_group')
            conflicts_v = [v for v, owners in ov.items() if len(owners) > 1]
            conflicts_e = [e for e, owners in oe.items() if len(owners) > 1]
            info['overuse_history'].append(len(conflicts_v))
            if not conflicts_v and not conflicts_e:
                break
            if iteration == max_iters:
                return finish('iteration_limit')
            for v in conflicts_v:
                hv[v] += .4 * (len(ov[v])-1)
            for edge in conflicts_e:
                he[edge] += .4 * (len(oe[edge])-1)
            present = min(present * 1.6, 1e4)
            for nid in group:
                remove(nid)
                tree = route(nid, present)
                if tree is None:
                    # Immediately abandon the private trial, never continue
                    # with a missing net or commit any of its partial trees.
                    return finish('negotiation_route_failed')
                add(nid, tree)
        if _checked_group(sr, group, routes) is None:
            return finish('candidate_invalid')
        for nid in sorted(group, key=lambda n: (-len(sr.regions.net_pins[n]), n)):
            previous = remove(nid)
            try:
                replacement = route(nid, 0, strict=True)
            except _Timeout:
                add(nid, previous)
                break
            if replacement is None or (replacement.delay, len(replacement.vertices)) > (previous.delay, len(previous.vertices)):
                replacement = previous
            add(nid, replacement)
        checked = _checked_group(sr, group, routes)
        if checked is None:
            return finish('polish_invalid')
        result_delay = sum(tree.delay for tree in routes.values())
        if result_delay >= original_delay:
            return finish('no_improvement')
        # Outside routes are unchanged, so check them once only for a useful
        # candidate. Rechecking the whole large design for every failed group
        # would make the guards themselves a congestion-runtime bottleneck.
        if _checked_group(sr, group, routes, full=True) is None:
            return finish('submission_invalid')
        gain = install_group(sr, routes)
        info.update(result_delay=result_delay, gain=gain)
        return finish('improved', gain)
    except _Timeout:
        return finish('time_budget')
    except MemoryError:
        return finish('memory_error')
