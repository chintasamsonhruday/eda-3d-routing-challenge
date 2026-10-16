"""Anytime, checked-best LNS with independent, measurable routing operators.

Guide costs generate candidates; integer driver-to-sink delay decides quality.
Private reconstruction cannot remove live wires. Search may cross plateaus or
accept an uphill legal state, but the returned/checkpointed best never worsens.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace, asdict
import hashlib
import heapq
import math
import random
import time
from types import SimpleNamespace

from m3d.checker import check
from m3d.grid import Grid, edge_key
from m3d.model import NetRoute, Submission
from .safe_negotiated_repair import build_pin_chimneys, validate_nonnegative
from .staged_router import Tree
from .trunk_lns import corridor_neighborhoods


class Deadline(Exception):
    pass


@dataclass(frozen=True)
class Options:
    operators: tuple = ('corridor', 'branch', 'recursive', 'chimney', 'window')
    max_group: int = 12
    max_iters: int = 60
    group_seconds: float = 1.5
    bounds: bool = True
    neutral: bool = True
    temperature: float = 0.0
    adaptive: bool = True
    shuffle: bool = True
    sequential_probability: float = .35
    tree_mode: str = 'spt'
    root_weight: float = 1.0
    geometry_weight: float = 0.0
    chimney_weight: float = 0.0
    direction_weight: float = 0.0
    via_scale: float = 1.0
    fanout_exponent: float = .5
    global_every: int = 0
    global_seconds: float = 6.0

    def __post_init__(self):
        if not self.operators or set(self.operators) - {'corridor', 'branch', 'recursive', 'chimney', 'window'}:
            raise ValueError('unknown or empty operator set')
        if self.max_group < 2 or self.max_iters < 0 or self.global_every < 0:
            raise ValueError('invalid group/iteration limits')
        if self.tree_mode not in ('spt', 'attachment'):
            raise ValueError('invalid tree mode')
        for name in ('group_seconds', 'temperature', 'root_weight', 'geometry_weight',
                     'chimney_weight', 'direction_weight', 'via_scale', 'fanout_exponent', 'global_seconds'):
            validate_nonnegative(**{name: getattr(self, name)})
        if self.via_scale == 0 or not 0 <= self.sequential_probability <= 1:
            raise ValueError('invalid via scale or reconstruction probability')


def profile(name):
    """One-factor ablations relative to the wider-neighborhood configuration."""
    base = Options(neutral=False, adaptive=False, bounds=False, max_group=6,
                   max_iters=14, shuffle=False, sequential_probability=0,
                   operators=('corridor',))
    groups = replace(base, operators=Options().operators, max_group=12, max_iters=60)
    choices = {
        'control': base, 'groups': groups,
        'bounds': replace(groups, bounds=True),
        'diversity': replace(groups, shuffle=True, sequential_probability=.35),
        'shuffle': replace(groups, shuffle=True),
        'sequential': replace(groups, sequential_probability=.35),
        'neutral': replace(groups, neutral=True),
        'neutral_shuffle': replace(groups, neutral=True, shuffle=True),
        'anneal': replace(groups, neutral=True, temperature=12),
        'root': replace(groups, tree_mode='attachment', root_weight=1),
        'branch': replace(groups, tree_mode='attachment', root_weight=1, geometry_weight=.3),
        'chimney': replace(groups, chimney_weight=.5),
        'direction': replace(groups, direction_weight=.3),
        'via': replace(groups, via_scale=.8),
        'via_reduce': replace(groups, via_scale=1.2),
        'scheduler': replace(groups, adaptive=True),
        'global': replace(groups, global_every=12, shuffle=True),
        'balanced': replace(groups, bounds=True, neutral=True, sequential_probability=.35),
        'combined': Options(temperature=8, tree_mode='attachment', geometry_weight=.15,
                            chimney_weight=.25, direction_weight=.15, global_every=35),
    }
    return choices[name]


class Search:
    def __init__(self, inst, initial, seed=0):
        verified = check(inst, initial)
        if not verified.legal:
            raise ValueError(f'illegal initial route: {verified.reasons}')
        if min(inst.layer_delay) <= 0 or inst.via_delay <= 0:
            raise ValueError('physical delays must be positive')
        self.inst, self.grid, self.rng = inst, Grid(inst), random.Random(seed)
        self.seed = seed
        self.size = inst.width * inst.height * inst.layers
        # Hard instances fit comfortably; avoid Python adjacency expansion on
        # very large tiers. Both paths use the same legal Grid neighbors.
        self.graph = ([tuple(self.grid.neighbors(v)) for v in range(self.size)]
                      if self.size <= 150000 else None)
        pv = {p.id: self.grid.vid(p.vertex()) for p in inst.pins}
        self.pins = {n.id: [pv[p] for p in n.pins()] for n in inst.nets}
        self.pin_owner = {v: nid for nid, pins in self.pins.items() for v in pins}
        self.chimneys = build_pin_chimneys(inst)
        self.access = None
        self.current = {r.net: self.make_tree(r.net,
                        {edge_key(self.grid.vid(a), self.grid.vid(b)) for a, b in r.edges})
                        for r in initial.routes}
        self.best = dict(self.current)
        self.initial_delay = self.best_delay = verified.total_delay
        self.owner = self.ownership(self.current)
        self.ideal = {}
        self.records = []
        self.opstats = {}
        self.visited = {self.fingerprint(self.current)}
        self.searches = self.expansions = 0
        self.corridors = None
        self.version = 0

    def neighbors(self, v):
        return self.graph[v] if self.graph is not None else self.grid.neighbors(v)

    def make_tree(self, nid, edges):
        adjacency = {}
        for a, b in edges:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        source = self.pins[nid][0]
        dist, order = {source: 0}, [source]
        for u in order:
            for v in adjacency.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + self.inst.edge_delay(self.grid.coord(u), self.grid.coord(v))
                    order.append(v)
        if len(edges) != len(dist)-1 or any(v not in dist for v in self.pins[nid]):
            raise ValueError('candidate is not a connected pin-spanning tree')
        return Tree(set(dist), set(edges), sum(dist[v] for v in self.pins[nid][1:]))

    def ownership(self, trees):
        owner = {}
        for nid, tree in trees.items():
            for v in tree.vertices:
                if v in owner and owner[v] != nid:
                    raise ValueError('overlapping candidate')
                owner[v] = nid
        return owner

    def submission(self, trees=None):
        trees = self.best if trees is None else trees
        return Submission(self.inst.name, [NetRoute(nid,
            [(self.grid.coord(a), self.grid.coord(b)) for a, b in sorted(trees[nid].edges)])
            for nid in sorted(trees)])

    @staticmethod
    def fingerprint(trees):
        h = hashlib.sha256()
        for nid in sorted(trees):
            h.update(repr((nid, sorted(trees[nid].edges))).encode())
        return h.hexdigest()

    def free_distance(self, a, b):
        ax, ay, az = self.grid.coord(a)
        bx, by, bz = self.grid.coord(b)
        xy = abs(ax-bx)+abs(ay-by)
        return min(xy*d + self.inst.via_delay*(abs(az-z)+abs(bz-z))
                   for z, d in enumerate(self.inst.layer_delay))

    def guide(self, nid, target, options, salt):
        source = self.pins[nid][0]
        central = ((self.inst.layers-1)//2, self.inst.layers//2)
        span = max(1, self.free_distance(source, target)) if target is not None else 1
        regrets = {}
        def extra(u, v, delay):
            z = v // self.grid.wh
            uz = u // self.grid.wh
            value = delay * (options.via_scale-1) if uz != z else 0.0
            if options.geometry_weight and target is not None:
                if v not in regrets:
                    regrets[v] = max(0, self.free_distance(source, v)+self.free_distance(v, target)-span) / span
                value += options.geometry_weight * delay * regrets[v]
            if options.chimney_weight and self.access:
                value += options.chimney_weight * sum(cost for owner, cost in self.access.get(v, {}).items() if owner != nid)
            if options.direction_weight and central[0] != central[1] and z == uz and z in central:
                dx = (v % self.grid.w) != (u % self.grid.w)
                prefer_x = (z == central[0]) != bool(salt & 1)
                if dx != prefer_x:
                    value += options.direction_weight * delay
            # Tiny, positive per-vertex noise breaks shortest-path ties without
            # changing the physical objective or creating negative guide edges.
            if options.shuffle:
                value += ((v*73856093 ^ nid*19349663 ^ salt*83492791) & 1023) * 1e-8
            return value
        return extra

    def route(self, nid, blocked, deadline, options=None, occupancy=None,
              history=None, present=0., salt=0):
        if time.perf_counter() >= deadline:
            raise Deadline
        options = options or Options(shuffle=False, tree_mode='spt', fanout_exponent=0)
        occupancy, history = occupancy or {}, history or {}
        source, *sinks = self.pins[nid]
        scale = max(1, len(sinks)) ** -options.fanout_exponent
        self.searches += 1
        tree_vertices, tree_edges, roots = {source}, set(), {source: 0}
        pending = sorted(set(sinks)-{source})
        if options.shuffle:
            random.Random(salt).shuffle(pending)
        goals = [None] if options.tree_mode == 'spt' else pending
        for target in goals:
            if target is not None and target in tree_vertices:
                continue
            targets = set(sinks)-{source} if target is None else {target}
            distance = ({source: 0.} if target is None else
                        {v: options.root_weight*roots[v] for v in tree_vertices})
            queue, parent = [(d, v) for v, d in distance.items()], {}
            heapq.heapify(queue)
            extra = self.guide(nid, target, options, salt)
            expanded = 0
            while queue and targets:
                if expanded % 128 == 0 and time.perf_counter() >= deadline:
                    raise Deadline
                d, u = heapq.heappop(queue)
                if d != distance.get(u):
                    continue
                expanded += 1
                targets.discard(u)
                if not targets:
                    break
                for v, delay in self.neighbors(u):
                    if v in blocked or self.pin_owner.get(v, nid) != nid:
                        continue
                    if target is not None and v in tree_vertices:
                        continue
                    price = scale * (history.get(v, 0.) + present*occupancy.get(v, 0))
                    nd = d + delay + extra(u, v, delay) + price
                    if nd < distance.get(v, math.inf):
                        distance[v], parent[v] = nd, u
                        heapq.heappush(queue, (nd, v))
            self.expansions += expanded
            if targets:
                return None
            for sink in sinks if target is None else [target]:
                v = sink
                chain = []
                while v not in tree_vertices:
                    u = parent[v]
                    chain.append((u, v))
                    v = u
                for u, v in reversed(chain):
                    tree_vertices.add(v)
                    tree_edges.add(edge_key(u, v))
                    roots[v] = roots[u]+self.inst.edge_delay(self.grid.coord(u), self.grid.coord(v))
        return Tree(tree_vertices, tree_edges, sum(roots[s] for s in sinks))

    def prepare(self, options, deadline):
        for nid in sorted(self.pins):
            if nid not in self.ideal:
                self.ideal[nid] = self.route(nid, set(), deadline)
        if options.chimney_weight and self.access is None:
            access = {}
            for v, claimants in sorted(self.chimneys.items()):
                for nid in sorted(claimants):
                    if v in self.pins[nid]:
                        continue
                    alternative = self.route(nid, {v}, deadline)
                    if alternative is not None:
                        penalty = max(0, alternative.delay-self.ideal[nid].delay)
                        if penalty:
                            access.setdefault(v, {})[nid] = penalty
            self.access = access

    def group_bound(self, group, deadline):
        members = set(group)
        blocked = {v for v, owner in self.owner.items() if owner not in members}
        lower = 0
        for nid in group:
            tree = self.route(nid, blocked, deadline)
            if tree is None:
                return math.inf
            lower += tree.delay
        return lower

    def repair(self, group, options, deadline, salt):
        """Build a complete private candidate; no live state changes here."""
        members = set(group)
        if not members or len(members) != len(group) or not members <= self.current.keys():
            raise ValueError('invalid group')
        blocked = {v for v, owner in self.owner.items() if owner not in members}
        order = list(group)
        if options.shuffle:
            self.rng.shuffle(order)
        stats = dict(iterations=0, conflicts=[], method='negotiated')
        if self.rng.random() < options.sequential_probability:
            trial, occupied = {}, set(blocked)
            for nid in order:
                tree = self.route(nid, occupied, deadline, options, salt=salt)
                if tree is None:
                    break
                trial[nid] = tree
                occupied.update(tree.vertices)
            if len(trial) == len(group):
                stats['method'] = 'sequential'
                return self.polish_group(trial, blocked, deadline), stats
        routes, occupancy, history = {}, Counter(), Counter()
        present = .5
        for iteration in range(options.max_iters+1):
            for nid in order:
                if nid in routes:
                    occupancy.subtract(routes[nid].vertices)
                tree = self.route(nid, blocked, deadline, options, occupancy, history, present, salt)
                if tree is None:
                    return None, dict(stats, reason='route_failed')
                routes[nid] = tree
                occupancy.update(tree.vertices)
            over = {v: count-1 for v, count in occupancy.items() if count > 1}
            stats['iterations'] = iteration
            stats['conflicts'].append(len(over))
            if not over:
                return self.polish_group(routes, blocked, deadline), stats
            history.update(over)
            present = min(8., present*1.4)
            if options.shuffle:
                self.rng.shuffle(order)
        return None, dict(stats, reason='iteration_limit')

    def polish_group(self, routes, blocked, deadline):
        occupied = set(blocked)
        for tree in routes.values():
            occupied.update(tree.vertices)
        for nid in sorted(routes, key=lambda n: -routes[n].delay):
            previous = routes[nid]
            occupied.difference_update(previous.vertices)
            try:
                replacement = self.route(nid, occupied, deadline)
            except Deadline:
                break
            if replacement is not None and (replacement.delay, len(replacement.vertices)) < (previous.delay, len(previous.vertices)):
                routes[nid] = replacement
            occupied.update(routes[nid].vertices)
        return routes

    def corridor_groups(self, maximum):
        proxy = SimpleNamespace(trees=self.current, inst=self.inst,
            regions=SimpleNamespace(grid=self.grid, net_pins=self.pins,
                                    pin_owner=self.pin_owner,
                                    owner=[self.owner.get(v) for v in range(self.size)]))
        return corridor_neighborhoods(proxy, max_group=maximum)

    def choose_group(self, operator, maximum):
        ids = sorted(self.current)
        weights = [max(0, self.current[n].delay-self.ideal[n].delay)+1 for n in ids]
        seed = self.rng.choices(ids, weights=weights)[0]
        maximum = min(maximum, len(ids))
        if operator == 'corridor':
            if self.corridors is None or self.corridors[0] != maximum:
                self.corridors = (maximum, self.corridor_groups(maximum))
            proposals = self.corridors[1]
            if not proposals:
                return None
            p = proposals.pop(self.rng.randrange(min(100, len(proposals))))
            return p.nets
        counts = Counter()
        if operator in ('branch', 'recursive'):
            ideal = self.ideal[seed]
            # Pick one sink branch rather than requiring the whole ideal tree's
            # potentially huge blocker union to fit a small neighborhood.
            adjacency = {}
            for a, b in ideal.edges:
                adjacency.setdefault(a, []).append(b); adjacency.setdefault(b, []).append(a)
            parent = {self.pins[seed][0]: None}
            queue = list(parent)
            for u in queue:
                for v in adjacency.get(u, ()):
                    if v not in parent:
                        parent[v] = u; queue.append(v)
            v = self.rng.choice(self.pins[seed][1:])
            while v is not None:
                owner = self.owner.get(v)
                if owner is not None and owner != seed:
                    counts[owner] += 1
                v = parent[v]
        elif operator == 'chimney':
            for v, claimants in self.chimneys.items():
                if seed in claimants:
                    owner = self.owner.get(v)
                    if owner is not None and owner != seed:
                        counts[owner] += 10
        elif operator == 'window':
            center = self.rng.choice(sorted(self.current[seed].vertices))
            x, y, _ = self.grid.coord(center)
            radius = self.rng.randint(1, 4)
            for z in range(self.inst.layers):
                for yy in range(max(0, y-radius), min(self.inst.height, y+radius+1)):
                    for xx in range(max(0, x-radius), min(self.inst.width, x+radius+1)):
                        owner = self.owner.get(self.grid.vid((xx, yy, z)))
                        if owner is not None and owner != seed:
                            counts[owner] += 1
        ranked = sorted(counts, key=lambda n: (-counts[n], self.rng.random()))
        group = [seed]+ranked[:maximum-1]
        if operator in ('recursive', 'chimney'):
            for nid in list(group[1:]):
                blockers = Counter(self.owner[v] for v in self.ideal[nid].vertices
                                   if v in self.owner and self.owner[v] not in group)
                for owner, _ in blockers.most_common():
                    if len(group) >= maximum:
                        break
                    if owner not in group:
                        group.append(owner)
        return tuple(group) if len(group) >= 2 else None

    def accept(self, routes, options, temperature, on_best=None):
        if not routes or not set(routes) <= self.current.keys():
            raise ValueError('empty/unknown candidate group')
        proposed = dict(self.current)
        proposed.update(routes)
        result = check(self.inst, self.submission(proposed))
        if not result.legal:
            return dict(reason='candidate_illegal', accepted=False, gain=0)
        delta = result.total_delay-sum(t.delay for t in self.current.values())
        fingerprint = self.fingerprint(proposed)
        if fingerprint in self.visited:
            return dict(reason='repeated_state', accepted=False, gain=0, delta=delta)
        allowed = delta < 0 or (delta == 0 and options.neutral) or (
            delta > 0 and temperature > 0 and self.rng.random() < math.exp(-delta/temperature))
        if not allowed:
            return dict(reason='no_improvement', accepted=False, gain=0, delta=delta)
        previous_best = self.best_delay
        self.current, self.owner = proposed, self.ownership(proposed)
        self.version += 1
        self.visited.add(fingerprint)
        self.corridors = None
        if result.total_delay < self.best_delay:
            self.best, self.best_delay = dict(proposed), result.total_delay
            if on_best:
                on_best(self.submission(), self.best_delay)
        return dict(reason='improved' if delta < 0 else 'neutral' if delta == 0 else 'uphill',
                    accepted=True, gain=previous_best-self.best_delay, delta=delta)

    def run(self, seconds, options=None, max_attempts=10000, on_best=None):
        options = options or Options()
        validate_nonnegative(seconds=seconds)
        started, first_record = time.perf_counter(), len(self.records)
        deadline = started+seconds
        initial = self.best_delay
        prepared = False
        try:
            self.prepare(options, deadline)
            prepared = True
            attempts, stagnant, seen_groups = 0, 0, set()
            while attempts < max_attempts and time.perf_counter() < deadline:
                attempts += 1
                t0 = time.perf_counter()
                stats_before = (self.searches, self.expansions)
                untried = [op for op in options.operators if not self.opstats.get(op, {}).get('attempts')]
                if untried:
                    operator = untried[0]
                elif options.adaptive and self.rng.random() > .2:
                    operator = max(options.operators, key=lambda op:
                        self.opstats[op]['gain']/max(.01, self.opstats[op]['seconds']) +
                        math.sqrt(2*math.log(attempts+1)/self.opstats[op]['attempts']))
                else:
                    operator = options.operators[(attempts-1) % len(options.operators)]
                maximum = min(options.max_group, (4, 6, 8, 12)[min(3, stagnant//15)])
                if options.operators == ('corridor',):
                    maximum = options.max_group
                global_trial = bool(options.global_every and attempts % options.global_every == 0)
                group = tuple(sorted(self.current)) if global_trial else self.choose_group(operator, maximum)
                if group is None and options.operators == ('corridor',):
                    break
                operator = 'global' if global_trial else operator
                record = dict(operator=operator, group=list(group or ()), maximum=maximum,
                              reason='no_group', accepted=False, gain=0)
                group_deadline = min(deadline, t0+(options.global_seconds if global_trial else options.group_seconds))
                try:
                    if group:
                        # Order matters for deterministic sequential negotiation.
                        # A cheap state generation avoids hashing every wire on
                        # every rejected proposal. Full hashes guard acceptance.
                        state_group = (self.version, tuple(group), operator)
                        if not options.shuffle and state_group in seen_groups:
                            record['reason'] = 'repeated_group'
                        else:
                            seen_groups.add(state_group)
                            bound = self.group_bound(group, group_deadline) if options.bounds else None
                            old = sum(self.current[n].delay for n in group)
                            record.update(group_delay=old, lower_bound=bound)
                            if bound is not None and old <= bound and not options.neutral and not options.temperature:
                                record['reason'] = 'zero_bound'
                            else:
                                candidate, repair_stats = self.repair(group, options, group_deadline,
                                            self.rng.randrange(1, 2**30) if options.shuffle else 0)
                                record['repair'] = repair_stats
                                if candidate is None:
                                    record['reason'] = repair_stats.get('reason', 'repair_failed')
                                else:
                                    cooling = max(0., 1-(time.perf_counter()-started)/max(seconds, 1e-9))
                                    record.update(self.accept(candidate, options, options.temperature*cooling, on_best))
                except Deadline:
                    record['reason'] = 'time_budget'
                except MemoryError:
                    record['reason'] = 'memory_error'
                elapsed = time.perf_counter()-t0
                record.update(seconds=elapsed, best_delay=self.best_delay,
                    current_delay=sum(t.delay for t in self.current.values()),
                    searches=self.searches-stats_before[0], expanded=self.expansions-stats_before[1])
                self.records.append(record)
                account = self.opstats.setdefault(operator, dict(attempts=0, gain=0, seconds=0., accepted=0))
                account['attempts'] += 1; account['gain'] += record['gain']; account['seconds'] += elapsed
                account['accepted'] += int(record['accepted'])
                stagnant = 0 if record['gain'] else stagnant+1
        except Deadline:
            pass
        # Always return to the independently verified best, including at a
        # timeout after a legal uphill working move.
        self.current, self.owner = dict(self.best), self.ownership(self.best)
        self.corridors = None
        checked = check(self.inst, self.submission())
        if not checked.legal or checked.total_delay != self.best_delay or self.best_delay > initial:
            raise RuntimeError('best-incumbent invariant failed')
        return dict(initial_delay=initial, best_delay=self.best_delay, legal=True,
                    gain=initial-self.best_delay, seconds=time.perf_counter()-started,
                    prepared=prepared, options=asdict(options), records=self.records[first_record:],
                    operator_statistics={op: dict(values) for op, values in self.opstats.items()})
