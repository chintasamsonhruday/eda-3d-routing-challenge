"""Run Optimized Multidimensional Pipeline on Hard Suite (case_09 down to case_01) in Parallel.

LPT (Longest Processing Time first) scheduling: case_09 -> case_01.
Workers: 6 parallel processes.
Pipeline per case:
  1. Seed Router with Knobs (cross_bias=5.5, mid_bias=0.5)
  2. Fast 1-Net Slack Polish
  3. Corridor LNS Pass 1 (max_group=5) + Interleaved Breathing Polish
  4. Corridor LNS Pass 2 (max_group=5) + Interleaved Breathing Polish
  5. Corridor LNS Pass 3 (max_group=6) + Interleaved Breathing Polish
  6. Chimney-Blocker LNS Cleanup (resolve remaining capped pins)
  7. Final Verification & Save
"""
import os, sys, time, json, math, heapq, multiprocessing
from collections import defaultdict
from pathlib import Path

from m3d.model import Instance, Submission, NetRoute
from m3d.grid import Grid, edge_key
from m3d.checker import check
from experiments.staged_router import StagedRouter, Tree
from experiments.trunk_lns import corridor_neighborhoods
from experiments.safe_negotiated_repair import mini_pathfinder as guarded_mini_pathfinder, save_checked_best

def _solve_case_impl(args):
    case_path, out_dir = args
    t0 = time.time()
    inst = Instance.load(case_path)
    g = Grid(inst)
    case_name = inst.name

    cross_bias = 5.5
    mid_bias = 0.5

    # Net classification & Pin maps
    pin_z = {p.id: p.z for p in inst.pins}
    pin_vid_map = {p.id: g.vid(p.vertex()) for p in inst.pins}
    pin_owner_map = {pin_vid_map[p.id]: inst.pin_to_net()[p.id] for p in inst.pins}

    net_types = {}
    for n in inst.nets:
        zs = {pin_z[p] for p in [n.driver] + n.sinks}
        if max(zs) <= 2: net_types[n.id] = 'bottom'
        elif min(zs) >= 3: net_types[n.id] = 'top'
        else: net_types[n.id] = 'cross'

    net_pins = {n.id: [pin_vid_map[p] for p in n.pins()] for n in inst.nets}

    # 1. Seed Router with Knobs
    def route_seed(max_iters=40):
        h_v = defaultdict(float)
        h_e = defaultdict(float)
        owners_v = defaultdict(set)
        owners_e = defaultdict(set)
        routes = {}

        def route_net(nid, pres_fac):
            pins = net_pins[nid]
            forbidden_pins = {v for v, o in pin_owner_map.items() if o != nid}
            ntype = net_types[nid]
            net_obj = [n for n in inst.nets if n.id == nid][0]
            n_sinks = len(net_obj.sinks)

            tree_v = {pins[0]}
            tree_e = set()
            remaining = set(pins[1:])

            while remaining:
                dist = {s: 0.0 for s in tree_v}
                prev = {}
                heap = [(0.0, s) for s in tree_v]
                heapq.heapify(heap)
                found = None

                while heap:
                    d, u = heapq.heappop(heap)
                    if d > dist.get(u, d): continue
                    if u in remaining:
                        found = u
                        break
                    for v, w in g.neighbors(u):
                        if v in forbidden_pins or v in tree_v: continue
                        ek = edge_key(u, v)
                        v_z = v // (g.w * g.h)

                        extra_cost = 0.0
                        if ntype == 'bottom':
                            if v_z >= 3: extra_cost += cross_bias
                            elif v_z == 2 and n_sinks == 1: extra_cost += mid_bias
                        elif ntype == 'top':
                            if v_z <= 2: extra_cost += cross_bias
                            elif v_z == 3 and n_sinks == 1: extra_cost += mid_bias

                        ec = (w + extra_cost) * (1.0 + h_e.get(ek, 0.0) + pres_fac * len(owners_e.get(ek, ())))
                        vc = 1.0 * (h_v.get(v, 0.0) + pres_fac * len(owners_v.get(v, ())))
                        nd = d + ec + vc
                        if nd < dist.get(v, 1e30):
                            dist[v] = nd
                            prev[v] = u
                            heapq.heappush(heap, (nd, v))

                if found is None: return None
                cur = found
                while cur in prev:
                    p = prev[cur]
                    tree_v.add(cur); tree_v.add(p)
                    tree_e.add(edge_key(p, cur))
                    cur = p
                remaining.discard(found)
            return tree_v, tree_e

        pres_fac = 0.5
        for nid in sorted(net_pins):
            r = route_net(nid, pres_fac)
            if r is None: return None
            for v in r[0]: owners_v[v].add(nid)
            for e in r[1]: owners_e[e].add(nid)
            routes[nid] = r

        for it in range(1, max_iters + 1):
            ov = [v for v, o in owners_v.items() if len(o) > 1]
            oe = [e for e, o in owners_e.items() if len(o) > 1]
            if not ov and not oe: break
            affected = set()
            for v in ov:
                h_v[v] += 0.5 * (len(owners_v[v]) - 1)
                affected |= owners_v[v]
            for e in oe:
                h_e[e] += 0.5 * (len(owners_e[e]) - 1)
                affected |= owners_e[e]
            pres_fac = min(pres_fac * 1.7, 1e5)
            for nid in sorted(affected):
                tv, te = routes.pop(nid)
                for v in tv:
                    owners_v[v].discard(nid)
                    if not owners_v[v]: del owners_v[v]
                for e in te:
                    owners_e[e].discard(nid)
                    if not owners_e[e]: del owners_e[e]
                r = route_net(nid, pres_fac)
                if r is None: return None
                for v in r[0]: owners_v[v].add(nid)
                for e in r[1]: owners_e[e].add(nid)
                routes[nid] = r

        sub_routes = []
        for nid, (tv, te) in sorted(routes.items()):
            edge_list = [[tuple(g.coord(a)), tuple(g.coord(b))] for (a, b) in te]
            sub_routes.append(NetRoute(nid, edge_list))
        return Submission(case_name, sub_routes)

    seed_sub = route_seed()
    if seed_sub is None:
        return {"case": case_name, "legal": False, "error": "Seed route failed"}
    seed_chk = check(inst, seed_sub)
    if not seed_chk.legal:
        return {"case": case_name, "legal": False, "error": f"Seed illegal: {seed_chk.reasons}"}

    sr = StagedRouter(inst, initial_route='negotiated', polish_passes=1, polish=False,
                      layer_opt=False, via_opt=False, final_opt=False)
    sr.install_checked_submission(seed_sub)

    # 2. Fast 1-Net Slack Polish
    sr.polish()

    # Mini-PathFinder Engine
    def mini_pathfinder(sr, group, max_iters=14):
        return guarded_mini_pathfinder(sr, group, max_iters=max_iters)

    # 3. Corridor LNS Pass 1
    props1 = corridor_neighborhoods(sr, max_group=5)
    for p in props1: mini_pathfinder(sr, p.nets)
    sr.polish()

    # 4. Corridor LNS Pass 2
    props2 = corridor_neighborhoods(sr, max_group=5)
    for p in props2: mini_pathfinder(sr, p.nets)
    sr.polish()

    # 5. Corridor LNS Pass 3 (max_group=6)
    props3 = corridor_neighborhoods(sr, max_group=6)
    for p in props3: mini_pathfinder(sr, p.nets)
    sr.polish()

    # 6. Chimney-Blocker LNS Cleanup
    vert_owner = {}
    for nid, tree in sr.trees.items():
        for v in tree.vertices: vert_owner[v] = nid
    knot_pairs = set()
    for p in inst.pins:
        cv = g.vid((p.x, p.y, 1 if p.z == 0 else 4))
        owner = vert_owner.get(cv)
        pn = inst.pin_to_net()[p.id]
        if owner is not None and owner != pn:
            knot_pairs.add(tuple(sorted((pn, owner))))
    for pair in sorted(knot_pairs):
        mini_pathfinder(sr, pair)

    # 7. Final Polish
    sr.polish()

    final_sub = sr.current_submission()
    res = check(inst, final_sub)
    dt = time.time() - t0

    sol_path = os.path.join(out_dir, f"{case_name}.sol.json")
    candidate_delay = res.total_delay
    res, saved_candidate = save_checked_best(inst, final_sub, sol_path)

    return {
        "case": case_name,
        "legal": res.legal,
        "delay": res.total_delay,
        "time": dt,
        "reasons": res.reasons,
        "candidate_delay": candidate_delay,
        "saved_candidate": saved_candidate
    }

def solve_case(args):
    try:
        return _solve_case_impl(args)
    except (ValueError, RuntimeError, KeyError, TypeError, OSError, MemoryError) as error:
        return dict(case=Path(args[0]).stem, legal=False,
                    error=str(error) or type(error).__name__, error_type=type(error).__name__)


def main():
    bench_dir = Path("benchmarks_hard")
    # Reverse order: case_09 down to case_01 (LPT scheduling)
    cases = sorted(bench_dir.glob("case_*.json"), reverse=True)
    out_dir = Path("experiments/results/hard_optimized_suite_20261011_1955")
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = 6
    print(f"============================================================")
    print(f"LAUNCHING HARD SUITE PARALLEL EVALUATION: case_09 -> case_01")
    print(f"Parallel Workers: {workers}")
    print(f"Cases ({len(cases)}): {[c.name for c in cases]}")
    print(f"Output Directory: {out_dir}")
    print(f"============================================================")

    t_start = time.time()
    task_args = [(str(c), str(out_dir)) for c in cases]

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(solve_case, task_args)

    total_time = time.time() - t_start

    # Sort results ascending for presentation
    results_sorted = sorted(results, key=lambda x: x["case"])

    baseline_delays = {
        "case_01": 11938, "case_02": 18910, "case_03": 14724,
        "case_04": 18077, "case_05": 21051, "case_06": 27134,
        "case_07": 31341, "case_08": 30886, "case_09": 27802
    }
    baseline_total = sum(baseline_delays.values())

    print("\n--- RESULTS PER CASE ---")
    total_delay = 0
    all_legal = True
    ratios = []
    for r in results_sorted:
        c_name = r["case"]
        status = "LEGAL" if r.get("legal") else "FAILED"
        if not r.get("legal"): all_legal = False
        del_val = r.get("delay", 0)
        total_delay += del_val
        b = baseline_delays.get(c_name, 0)
        ratio = b / del_val if del_val > 0 else 0.0
        ratios.append(ratio)
        print(f"  {c_name}: {status} | delay={del_val:5d} | ratio={ratio:.5f} | time={r.get('time', 0):.1f}s")

    geo_mean = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if all_legal else 0.0

    print(f"\n============================================================")
    print(f"HARD SUITE PARALLEL EVALUATION SUMMARY:")
    print(f"Legal: {all_legal} ({sum(1 for r in results if r.get('legal'))}/{len(cases)})")
    print(f"Total Delay: {total_delay} (Baseline: {baseline_total})")
    print(f"Total Delay Saved: {baseline_total - total_delay} points")
    print(f"GEOMETRIC MEAN AGGREGATE: {geo_mean:.5f}")
    print(f"Total Suite Runtime: {total_time:.2f}s")
    print(f"============================================================")

    summary = {
        "aggregate": geo_mean,
        "total_delay": total_delay,
        "total_time": total_time,
        "results": results_sorted
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

