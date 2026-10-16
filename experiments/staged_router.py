"""Coarse planning, legal detailed routing, and staged route optimization.

Regions are tile_size by tile_size XY vertices on EACH layer. Coarse capacity
and demand are estimates; only the exact vertex reservations prove legality.
All returned routes are independently checked. No reference routes are read.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
import time

from m3d.checker import check
from m3d.grid import Grid, edge_key
from m3d.model import NetRoute, Submission
from m3d.negotiated import route_negotiated


@dataclass
class Tree:
    vertices: set
    edges: set
    delay: int


class Regions:
    """Track exact occupied vertices and unavailable boundary crossing edges."""

    def __init__(self, inst, tile_size=4):
        if tile_size < 1:
            raise ValueError("tile_size must be positive")
        self.inst, self.grid, self.side = inst, Grid(inst), tile_size
        self.size = inst.width * inst.height * inst.layers
        self.nx = (inst.width + tile_size - 1) // tile_size
        self.ny = (inst.height + tile_size - 1) // tile_size
        self.tiles = []
        for z in range(inst.layers):
            for ty in range(self.ny):
                for tx in range(self.nx):
                    x0, y0 = tx * tile_size, ty * tile_size
                    x1, y1 = min(x0 + tile_size, inst.width), min(y0 + tile_size, inst.height)
                    self.tiles.append(dict(id=len(self.tiles), x0=x0, y0=y0, x1=x1,
                                           y1=y1, z=z, capacity=(x1-x0)*(y1-y0)))
        self.region = [self.tile_id(*self.grid.coord(v)) for v in range(self.size)]
        self.owner = [None] * self.size
        self.used = [0] * len(self.tiles)
        self.boundaries, self.boundary_id = [], {}
        self.links = [[] for _ in self.tiles]
        self.portals_at = [[] for _ in range(self.size)]
        for u in range(self.size):
            a = self.region[u]
            for v, delay in self.grid.neighbors(u):
                b = self.region[v]
                if v <= u or a == b:
                    continue
                key = edge_key(a, b)
                if key not in self.boundary_id:
                    bid = len(self.boundaries)
                    self.boundary_id[key] = bid
                    # Coarse traversal approximates a tile-width movement.
                    cost = delay if self.tiles[a]['z'] != self.tiles[b]['z'] else delay * tile_size
                    self.boundaries.append(dict(a=a, b=b, pairs=[], blocked=0))
                    self.links[a].append((b, bid, cost))
                    self.links[b].append((a, bid, cost))
                bid = self.boundary_id[key]
                self.boundaries[bid]['pairs'].append((u, v))
                self.portals_at[u].append((bid, v))
                self.portals_at[v].append((bid, u))
        self.pin_owner = {}
        self.net_pins = {}
        pin_vid = {p.id: self.grid.vid(p.vertex()) for p in inst.pins}
        for net in inst.nets:
            self.net_pins[net.id] = [pin_vid[p] for p in net.pins()]
            for v in self.net_pins[net.id]:
                self.pin_owner[v] = net.id
                self.claim(v, net.id)

    def tile_id(self, x, y, z):
        return (z * self.ny + y // self.side) * self.nx + x // self.side

    def claim(self, v, nid):
        if self.owner[v] == nid:
            return
        if self.owner[v] is not None:
            raise RuntimeError(f"vertex {v} already owned by {self.owner[v]}")
        for bid, other in self.portals_at[v]:
            if self.owner[other] is None:
                self.boundaries[bid]['blocked'] += 1
        self.owner[v] = nid
        self.used[self.region[v]] += 1

    def release(self, v, nid):
        if self.owner[v] != nid:
            raise RuntimeError("cannot release another net's vertex")
        if v in self.pin_owner:
            return
        self.owner[v] = None
        self.used[self.region[v]] -= 1
        for bid, other in self.portals_at[v]:
            if self.owner[other] is None:
                self.boundaries[bid]['blocked'] -= 1

    def halo(self, tiles, radius=1):
        result = set(tiles)
        for _ in range(radius):
            result |= {b for a in tuple(result) for b, _, _ in self.links[a]}
        return result


class StagedRouter:
    def __init__(self, inst, tile_size=4, utilization_weight=0.7,
                 guide_weight=0.05, max_repairs=128, polish=True, polish_passes=2,
                 middle_bias=2.0, via_bias=4.0, stage_delay_slack=0.02,
                 layer_fraction=0.5, via_fraction=0.5, final_fraction=0.25,
                 layer_opt=True, via_opt=True, final_opt=True,
                 negotiated_fallback=True, fallback_iterations=50,
                 initial_route='staged'):
        if (utilization_weight < 0 or guide_weight < 0 or max_repairs < 0 or polish_passes < 1
                or middle_bias < 0 or via_bias < 0 or stage_delay_slack < 0
                or fallback_iterations < 1):
            raise ValueError("weights, budgets, and slack must be nonnegative")
        if any(not 0 <= fraction <= 1 for fraction in (layer_fraction, via_fraction, final_fraction)):
            raise ValueError("stage fractions must be between zero and one")
        if initial_route not in ('staged', 'negotiated'):
            raise ValueError("initial_route must be staged or negotiated")
        started = time.perf_counter()
        self.inst, self.regions = inst, Regions(inst, tile_size)
        self.utilization_weight, self.guide_weight = utilization_weight, guide_weight
        self.max_repairs, self.polish_enabled = max_repairs, polish
        self.polish_passes = polish_passes
        self.middle_bias, self.via_bias = middle_bias, via_bias
        self.stage_delay_slack = stage_delay_slack
        self.layer_fraction, self.via_fraction = layer_fraction, via_fraction
        self.final_fraction = final_fraction
        self.layer_opt_enabled, self.via_opt_enabled, self.final_opt_enabled = layer_opt, via_opt, final_opt
        self.negotiated_fallback = negotiated_fallback
        self.fallback_iterations = fallback_iterations
        self.initial_route = initial_route
        self.best_submission = None
        self.trees, self.guides, self.guide_edges = {}, {}, {}
        self.forecast = [0] * len(self.regions.boundaries)
        self.history = [0] * self.regions.size
        self.attempts = {n.id: 0 for n in inst.nets}
        self.events = []
        self.stats = dict(tile_size=tile_size, tiles=len(self.regions.tiles),
                          boundaries=len(self.regions.boundaries), searches=0,
                          expanded_vertices=0, corridor_expansions=0, repairs=0,
                          polished_nets=0, fallback_used=False, initial_route=initial_route,
                          stage_seconds={'partition': time.perf_counter()-started}, stages=[])
        self.initial_snapshot = None

    def order(self):
        r = self.regions
        def priority(n):
            points = [r.grid.coord(v) for v in r.net_pins[n.id]]
            span = sum(max(p[i] for p in points)-min(p[i] for p in points) for i in range(3))
            return (-span, n.id)
        return [n.id for n in sorted(self.inst.nets, key=priority)]

    def coarse_plan(self, nid):
        r = self.regions
        source, *sinks = [r.region[v] for v in r.net_pins[nid]]
        targets = set(sinks) - {source}
        distance, parent, capacity = {source: 0.0}, {}, {}
        heap = [(0.0, source)]
        while heap and targets:
            d, a = heapq.heappop(heap)
            if d != distance[a]:
                continue
            targets.discard(a)
            for b, bid, delay in r.links[a]:
                if bid not in capacity:
                    capacity[bid] = sum(
                        r.pin_owner.get(u, nid) == nid and r.pin_owner.get(v, nid) == nid
                        for u, v in r.boundaries[bid]['pairs'])
                if not capacity[bid]:
                    continue
                predicted_util = (self.forecast[bid] + 1) / capacity[bid]
                cost = delay * (1 + 2 * predicted_util ** 2)
                nd = d + cost
                if nd < distance.get(b, float('inf')):
                    distance[b] = nd
                    parent[b] = (a, bid)
                    heapq.heappush(heap, (nd, b))
        if targets:
            return None
        tiles, borders = {source}, set()
        for sink in sinks:
            cur = sink
            while cur not in tiles:
                tiles.add(cur)
                cur, bid = parent[cur]
                borders.add(bid)
        return tiles, borders

    def plan_all(self):
        # Initial estimate followed by a global demand-aware replanning pass.
        for pass_no in range(2):
            for nid in self.order():
                for bid in self.guide_edges.get(nid, ()):
                    self.forecast[bid] -= 1
                plan = self.coarse_plan(nid)
                if plan is None:
                    self.stats['failure'] = f"no coarse connection for net {nid}"
                    return False
                self.guides[nid], self.guide_edges[nid] = plan
                for bid in plan[1]:
                    self.forecast[bid] += 1
            self.events.append(dict(stage='coarse_estimate' if pass_no == 0 else 'global_plan',
                                    planned_nets=len(self.guides), connected_nets=0,
                                    pending_nets=self.order()))
        self.initial_snapshot = self.snapshot()
        return True

    def fine_tree(self, nid, guided=True, allowed=None, relaxed=False, objective='delay'):
        """Relaxed searches diagnose blockers only; their routes are never committed."""
        r = self.regions
        owner, regions = r.owner, r.region
        source, *sinks = r.net_pins[nid]
        remaining = set(sinks) - {source}
        middle_penalty = None
        if objective in ('middle', 'via'):
            centers = self.middle_layers()
            middle_penalty = [self.middle_bias * min(abs(z-center) for center in centers)
                              for z in range(self.inst.layers)]
        if guided:
            projected = [min(1.0, (n+1)/t['capacity']) for n, t in zip(r.used, r.tiles)]
            tile_penalty = [self.utilization_weight * u**4 for u in projected]
            border_penalty = {}
            for border in r.boundaries:
                a, b = border['a'], border['b']
                utilization = max(projected[a], projected[b],
                                  min(1.0, (border['blocked']+1)/len(border['pairs'])))
                border_penalty[edge_key(a,b)] = self.utilization_weight * utilization**4
            guide_penalty = [0 if t in self.guides[nid] else self.guide_weight
                             for t in range(len(r.tiles))]
        vertices, edges, real_distance = {source}, set(), {source: 0}
        while remaining:
            # Connectivity stage: attach the nearest sink to the existing tree
            # to conserve resources. Cleanup: root all shortest paths at driver.
            sources = vertices if guided else {source}
            distance, parent = [None] * r.size, [-1] * r.size
            heap = [(0.0, v) for v in sorted(sources)]
            for v in sources:
                distance[v] = 0.0
            heapq.heapify(heap)
            targets, found = set(remaining), []
            self.stats['searches'] += 1
            while heap and targets:
                d, u = heapq.heappop(heap)
                if d != distance[u]:
                    continue
                self.stats['expanded_vertices'] += 1
                if u in targets:
                    targets.remove(u)
                    found.append(u)
                    if guided or not targets:
                        break
                a = regions[u]
                for v, delay in r.grid.neighbors(u):
                    b = regions[v]
                    if v in sources or (allowed is not None and b not in allowed):
                        continue
                    if r.pin_owner.get(v, nid) != nid:
                        continue
                    occupied = owner[v] is not None and owner[v] != nid
                    if occupied and not relaxed:
                        continue
                    if guided:
                        penalty = tile_penalty[b] if a == b else border_penalty[edge_key(a,b)]
                        cost = delay * (1 + penalty + guide_penalty[b]) + self.history[v]
                    else:
                        cost = delay
                        if objective in ('middle', 'via') and r.tiles[a]['z'] == r.tiles[b]['z']:
                            cost += middle_penalty[r.tiles[b]['z']] * (1 if objective == 'middle' else 0.25)
                        elif objective == 'via' and r.tiles[a]['z'] != r.tiles[b]['z']:
                            cost += self.via_bias
                    if occupied:
                        cost += 10000  # diagnostic path favors few occupied vertices
                    nd = d + cost
                    if distance[v] is None or nd < distance[v]:
                        distance[v], parent[v] = nd, u
                        heapq.heappush(heap, (nd, v))
            if not found or (not guided and targets):
                return None
            for sink in found:
                path, cur = [], sink
                while cur not in vertices:
                    path.append(cur)
                    cur = parent[cur]
                for v in reversed(path):
                    edges.add(edge_key(cur, v))
                    vertices.add(v)
                    real_distance[v] = real_distance[cur] + self.inst.edge_delay(r.grid.coord(cur), r.grid.coord(v))
                    cur = v
                remaining.discard(sink)
        tree = Tree(vertices, edges, 0)
        tree.delay = self.tree_delay(nid, tree)
        return tree

    def tree_delay(self, nid, tree):
        """Measure the submitted tree from the driver, including shared branches."""
        adjacency = {v: [] for v in tree.vertices}
        for a, b in tree.edges:
            delay = self.inst.edge_delay(self.regions.grid.coord(a), self.regions.grid.coord(b))
            adjacency[a].append((b, delay))
            adjacency[b].append((a, delay))
        source, *sinks = self.regions.net_pins[nid]
        distance = {source: 0}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v, delay in adjacency[u]:
                if v not in distance:
                    distance[v] = distance[u] + delay
                    queue.append(v)
        if len(distance) != len(tree.vertices) or len(tree.edges) != len(tree.vertices)-1:
            raise RuntimeError(f'net {nid} route is not a connected tree')
        return sum(distance[v] for v in sinks)

    def commit(self, nid, tree):
        # Check all ownership before mutating anything, including tile borders.
        if any(self.regions.owner[v] not in (None, nid) for v in tree.vertices):
            raise RuntimeError("attempt to commit conflicting detailed route")
        for v in sorted(tree.vertices):
            self.regions.claim(v, nid)
        self.trees[nid] = tree

    def remove(self, nid):
        tree = self.trees.pop(nid)
        for v in sorted(tree.vertices):
            self.regions.release(v, nid)
        return tree

    def connect(self):
        queue = deque(self.order())
        while queue:
            nid = queue.popleft()
            if nid in self.trees:
                continue
            self.attempts[nid] += 1
            tree = self.fine_tree(nid, allowed=self.regions.halo(self.guides[nid]))
            if tree is None:
                self.stats['corridor_expansions'] += 1
                tree = self.fine_tree(nid)
            if tree is not None:
                self.commit(nid, tree)
                self.events.append(dict(stage='detailed', net=nid, action='connected',
                                        connected_nets=len(self.trees), pending_count=len(self.inst.nets)-len(self.trees)))
                continue
            if self.stats['repairs'] >= self.max_repairs:
                self.stats['failure'] = f"repair budget exhausted at net {nid}"
                return False
            proposal = self.fine_tree(nid, allowed=None, relaxed=True)
            if proposal is None:
                self.stats['failure'] = f"net {nid} blocked even without other routes"
                return False
            blocked_vertices = [v for v in proposal.vertices if self.regions.owner[v] not in (None, nid)]
            blockers = sorted({self.regions.owner[v] for v in blocked_vertices})
            if not blockers:
                raise RuntimeError("legal search failed but diagnostic found no blockers")
            for v in blocked_vertices:
                self.history[v] += 1
            for blocker in blockers:
                self.remove(blocker)
            self.stats['repairs'] += 1
            self.events.append(dict(stage='repair', blocked_net=nid, removed_nets=blockers,
                                    connected_nets=len(self.trees), diagnostic_only=True))
            # Only these blockers are removed. All other legal routes stay fixed.
            queue = deque([nid] + blockers + [n for n in queue if n != nid and n not in blockers])
        return True

    def polish(self):
        before = sum(t.delay for t in self.trees.values())
        changed_nets = set()
        self.stats['polish_progress'] = []
        for pass_no in range(self.polish_passes):
            changes = 0
            for nid in sorted(self.trees, key=lambda n: (-self.trees[n].delay, n)):
                old = self.remove(nid)
                candidate = self.fine_tree(nid, guided=False)
                improved = candidate is not None and (candidate.delay, len(candidate.vertices)) < (old.delay, len(old.vertices))
                self.commit(nid, candidate if improved else old)
                if improved:
                    changes += 1
                    changed_nets.add(nid)
            self.stats['polish_progress'].append(dict(pass_number=pass_no+1, accepted=changes,
                                                  total_delay=sum(t.delay for t in self.trees.values())))
            if not changes:
                break
        self.stats['polished_nets'] = len(changed_nets)
        self.stats['delay_before_polish'] = before
        self.stats['delay_after_polish'] = sum(t.delay for t in self.trees.values())

    def middle_layers(self):
        count = self.inst.layers
        return ((count-1)//2, count//2) if count % 2 == 0 else (count//2,)

    def tree_counts(self, tree):
        via = middle = outer = 0
        centers = set(self.middle_layers())
        for a, b in tree.edges:
            z_a = self.regions.grid.coord(a)[2]
            z_b = self.regions.grid.coord(b)[2]
            if z_a != z_b:
                via += 1
            elif z_a in centers:
                middle += 1
            else:
                outer += 1
        return dict(vias=via, middle_edges=middle, outer_edges=outer)

    def current_submission(self):
        g = self.regions.grid
        return Submission(self.inst.name, [NetRoute(n.id, [(g.coord(a), g.coord(b))
            for a, b in sorted(self.trees[n.id].edges)]) for n in self.inst.nets])

    def record_stage(self, name, started, enabled=True, accepted=0):
        if len(self.trees) != len(self.inst.nets):
            raise RuntimeError(f'{name} left nets disconnected')
        submission = self.current_submission()
        result = check(self.inst, submission)
        if not result.legal:
            raise RuntimeError(f'{name} failed independent legality check: {result.reasons}')
        via = middle = outer = 0
        for tree in self.trees.values():
            counts = self.tree_counts(tree)
            via += counts['vias']
            middle += counts['middle_edges']
            outer += counts['outer_edges']
        row = dict(stage=name, enabled=enabled, legal=True,
                   connected_nets=len(self.trees), pending_nets=0,
                   total_delay=result.total_delay, via_edges=via,
                   middle_edges=middle, outer_edges=outer,
                   middle_fraction=middle/(middle+outer) if middle+outer else 0,
                   accepted_changes=accepted, seconds=time.perf_counter()-started)
        self.stats['stages'].append(row)
        self.stats['stage_seconds'][name] = row['seconds']
        if (self.initial_route == 'negotiated'
                and result.total_delay < self.stats.get('best_delay', float('inf'))):
            self.best_submission = submission
            self.stats['best_delay'] = result.total_delay
            self.stats['best_stage'] = name
        return row

    def optimize_resource(self, objective, fraction):
        if fraction == 0:
            return 0
        key = 'outer_edges' if objective == 'middle' else 'vias'
        candidates = [(self.tree_counts(tree)[key], tree.delay, nid)
                      for nid, tree in self.trees.items()]
        candidates = sorted((item for item in candidates if item[0] > 0), reverse=True)
        limit = math.ceil(len(self.trees) * fraction)
        changed = 0
        for _, _, nid in candidates[:limit]:
            old = self.remove(nid)
            replacement = self.fine_tree(nid, guided=False, objective=objective)
            accept = False
            if replacement is not None and replacement.delay <= old.delay * (1+self.stage_delay_slack):
                old_count, new_count = self.tree_counts(old), self.tree_counts(replacement)
                if objective == 'middle':
                    accept = ((new_count['middle_edges'] > old_count['middle_edges']
                               and new_count['outer_edges'] <= old_count['outer_edges'])
                              or (new_count['outer_edges'] < old_count['outer_edges']
                                  and new_count['middle_edges'] >= old_count['middle_edges']))
                else:
                    old_wire = old_count['middle_edges'] + old_count['outer_edges']
                    new_wire = new_count['middle_edges'] + new_count['outer_edges']
                    keeps_middle_share = (new_count['middle_edges'] * old_wire
                                          >= old_count['middle_edges'] * new_wire)
                    accept = new_count['vias'] < old_count['vias'] and keeps_middle_share
            self.commit(nid, replacement if accept else old)
            changed += int(accept)
        return changed

    def lower_bound(self, nid):
        g = self.regions.grid
        source, *sinks = [g.coord(v) for v in self.regions.net_pins[nid]]
        return sum(min(
            (abs(source[0]-sink[0])+abs(source[1]-sink[1])) * self.inst.layer_delay[z]
            + (abs(source[2]-z)+abs(sink[2]-z))*self.inst.via_delay
            for z in range(self.inst.layers)) for sink in sinks)

    def fast_driver_tree(self, nid):
        """A* grows from the existing driver tree, seeding exact driver distances.

        Unlike the broad full-root delay pass, it searches only until the next
        sink is reached and never revisits existing tree vertices as new wire.
        """
        r = self.regions
        g = r.grid
        source, *sinks = r.net_pins[nid]
        pending = set(sinks) - {source}
        vertices, edges, real = {source}, set(), {source: 0}
        sink_positions = {v:g.coord(v) for v in pending}
        minimum_layer_delay = min(self.inst.layer_delay)
        while pending:
            targets = [sink_positions[v] for v in pending]
            heuristic_cache = {}
            def heuristic(v):
                if v not in heuristic_cache:
                    x,y,z = g.coord(v)
                    heuristic_cache[v] = min((abs(x-sx)+abs(y-sy))*minimum_layer_delay
                                              + abs(z-sz)*self.inst.via_delay
                                              for sx,sy,sz in targets)
                return heuristic_cache[v]
            distance = {v:real[v] for v in vertices}
            parent = {}
            heap = [(distance[v]+heuristic(v), distance[v], v) for v in vertices]
            heapq.heapify(heap)
            self.stats['searches'] += 1
            found = None
            while heap:
                _, d, u = heapq.heappop(heap)
                if d != distance.get(u):
                    continue
                self.stats['expanded_vertices'] += 1
                if u in pending:
                    found = u
                    break
                for v, delay in g.neighbors(u):
                    if v in vertices or r.pin_owner.get(v, nid) != nid:
                        continue
                    if r.owner[v] not in (None, nid):
                        continue
                    candidate = d + delay
                    if candidate < distance.get(v, float('inf')):
                        distance[v], parent[v] = candidate, u
                        heapq.heappush(heap, (candidate+heuristic(v), candidate, v))
            if found is None:
                return None
            path, cursor = [], found
            while cursor not in vertices:
                path.append(cursor)
                cursor = parent[cursor]
            for v in reversed(path):
                edges.add(edge_key(cursor, v))
                vertices.add(v)
                real[v] = real[cursor] + self.inst.edge_delay(g.coord(cursor), g.coord(v))
                cursor = v
            pending.remove(found)
        tree = Tree(vertices, edges, 0)
        tree.delay = self.tree_delay(nid, tree)
        return tree

    def fast_final_delay(self):
        if self.final_fraction == 0:
            return 0
        ranked = sorted(((tree.delay-self.lower_bound(nid), nid) for nid, tree in self.trees.items()),
                        key=lambda item: (-item[0], item[1]))
        limit = math.ceil(len(self.trees)*self.final_fraction)
        changed = 0
        for gap, nid in ranked[:limit]:
            if gap <= 0:
                break
            old = self.remove(nid)
            replacement = self.fast_driver_tree(nid)
            if replacement is not None:
                old_count, new_count = self.tree_counts(old), self.tree_counts(replacement)
                accept = ((replacement.delay, new_count['vias'], len(replacement.vertices))
                          < (old.delay, old_count['vias'], len(old.vertices)))
            else:
                accept = False
            self.commit(nid, replacement if accept else old)
            changed += int(accept)
        return changed

    def install_checked_submission(self, submission):
        """Install a complete checked route into the exact ownership map."""
        result = check(self.inst, submission)
        if not result.legal:
            raise RuntimeError(f'cannot install illegal route: {result.reasons}')
        for nid in list(self.trees):
            self.remove(nid)
        g = self.regions.grid
        for net_route in submission.routes:
            edges = {edge_key(g.vid(a), g.vid(b)) for a, b in net_route.edges}
            vertices = set(self.regions.net_pins[net_route.net])
            for a, b in edges:
                vertices.add(a)
                vertices.add(b)
            tree = Tree(vertices, edges, 0)
            tree.delay = self.tree_delay(net_route.net, tree)
            self.commit(net_route.net, tree)
        return result.total_delay

    def negotiated_start(self, fallback=False):
        """Initialize from negotiated congestion, optionally after a failed repair."""
        submission, info = route_negotiated(self.inst, max_iters=self.fallback_iterations)
        self.stats['negotiated_iterations_used'] = info.iterations
        if submission is None or not check(self.inst, submission).legal:
            return False
        self.stats['negotiated_seed_delay'] = self.install_checked_submission(submission)
        if fallback:
            self.stats['fallback_used'] = True
            self.stats['initial_repair_failure'] = self.stats.pop('failure', None)
        self.events.append(dict(stage='negotiated_fallback' if fallback else 'negotiated_seed',
                                connected_nets=len(self.trees), pending_count=0,
                                iterations=info.iterations))
        return True

    def try_negotiated_fallback(self):
        """Recover a legal starting route when targeted repair cannot converge."""
        success = self.negotiated_start(fallback=True)
        self.stats['fallback_iterations_used'] = self.stats.get('negotiated_iterations_used', 0)
        return success

    def snapshot(self):
        r = self.regions
        routed, pending = [set() for _ in r.tiles], [set() for _ in r.tiles]
        for nid, tree in self.trees.items():
            for v in tree.vertices:
                routed[r.region[v]].add(nid)
        for nid, tiles in self.guides.items():
            if nid not in self.trees:
                for tile in tiles:
                    pending[tile].add(nid)
        tiles = []
        for t in r.tiles:
            i = t['id']
            tiles.append(dict(t, occupied_vertices=r.used[i], free_vertices=t['capacity']-r.used[i],
                              utilization=r.used[i]/t['capacity'], routed_nets=sorted(routed[i]),
                              planned_pending_nets=sorted(pending[i])))
        borders = [dict(id=i, a=b['a'], b=b['b'], capacity=len(b['pairs']),
                        available_crossings=len(b['pairs'])-b['blocked'],
                        blocked_fraction=b['blocked']/len(b['pairs']),
                        estimated_net_demand=self.forecast[i]) for i,b in enumerate(r.boundaries)]
        return dict(tiles=tiles, boundaries=borders,
                    max_tile_utilization=max((t['utilization'] for t in tiles), default=0),
                    max_boundary_blocked_fraction=max((b['blocked_fraction'] for b in borders), default=0),
                    note='Occupancy includes all reserved pins. Free capacity alone does not guarantee connectivity.')

    def run(self):
        if self.initial_route == 'negotiated':
            start = time.perf_counter()
            if not self.negotiated_start():
                self.stats['failure'] = 'negotiated initialization did not find a legal route'
                self.stats['stage_seconds']['negotiated_seed'] = time.perf_counter()-start
                return None, self.report()
            self.initial_snapshot = self.snapshot()
            self.record_stage('negotiated_seed', start)
        else:
            start = time.perf_counter()
            planned = self.plan_all()
            self.stats['stage_seconds']['coarse_plan'] = time.perf_counter()-start
            if not planned:
                return None, self.report()
            start = time.perf_counter()
            connected = self.connect()
            if not connected and self.negotiated_fallback:
                connected = self.try_negotiated_fallback()
            if not connected:
                self.stats['stage_seconds']['detailed_and_repair'] = time.perf_counter()-start
                return None, self.report()
            self.record_stage('global_connectivity', start)
            self.stats['stage_seconds']['detailed_and_repair'] = self.stats['stage_seconds']['global_connectivity']
        start = time.perf_counter()
        if self.polish_enabled:
            self.polish()
        accepted = sum(p['accepted'] for p in self.stats.get('polish_progress', []))
        self.record_stage('delay_opt', start, enabled=self.polish_enabled, accepted=accepted)
        self.stats['stage_seconds']['polish'] = self.stats['stage_seconds']['delay_opt']
        start = time.perf_counter()
        accepted = self.optimize_resource('middle', self.layer_fraction) if self.layer_opt_enabled else 0
        self.record_stage('layer_opt', start, enabled=self.layer_opt_enabled, accepted=accepted)
        start = time.perf_counter()
        accepted = self.optimize_resource('via', self.via_fraction) if self.via_opt_enabled else 0
        self.record_stage('via_opt', start, enabled=self.via_opt_enabled, accepted=accepted)
        start = time.perf_counter()
        accepted = self.fast_final_delay() if self.final_opt_enabled else 0
        self.record_stage('final_delay_connectivity', start, enabled=self.final_opt_enabled, accepted=accepted)
        if (self.initial_route == 'negotiated'
                and self.stats['stages'][-1]['total_delay'] > self.stats['best_delay']):
            start = time.perf_counter()
            self.install_checked_submission(self.best_submission)
            self.record_stage('best_delay_selection', start)
        submission = self.current_submission()
        result = check(self.inst, submission)
        if not result.legal:
            raise RuntimeError(f"independent checker rejected route: {result.reasons}")
        self.stats.update(legal=True, total_delay=result.total_delay)
        return submission, self.report()

    def report(self):
        return dict(stats=self.stats, events=self.events,
                    connected_nets=sorted(self.trees),
                    pending_nets=[n.id for n in self.inst.nets if n.id not in self.trees],
                    net_attempts=self.attempts, planned=self.initial_snapshot,
                    final=self.snapshot())


def route(inst, **kwargs):
    return StagedRouter(inst, **kwargs).run()
