"""Warm-start a local legal route and test sink-weighted corridor group LNS."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3d.checker import check
from m3d.model import Instance, Submission
from experiments.staged_router import StagedRouter
from experiments.trunk_lns import improve


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--initial-sol', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=60)
    parser.add_argument('--group-budget', type=float, default=2)
    parser.add_argument('--max-nodes', type=int, default=200)
    parser.add_argument('--max-group', type=int, default=6)
    parser.add_argument('--max-groups', type=int, default=100)
    parser.add_argument('--conflict-avoidance', action='store_true')
    parser.add_argument('--baseline-delay', type=int)
    parser.add_argument('--seed-summary', type=Path,
                        help='measured seed summary, to include initialization runtime')
    args = parser.parse_args()
    if (args.budget <= 0 or args.group_budget <= 0 or args.max_nodes < 1
            or not 2 <= args.max_group <= 6 or args.max_groups < 1):
        parser.error('budgets and counts must be positive; group size must be 2--6')
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    instance = Instance.load(str(args.case))
    seed = Submission.load(str(args.initial_sol))
    initial_result = check(instance, seed)
    if not initial_result.legal:
        raise ValueError(f'illegal initial route: {initial_result.reasons}')
    seed_runtime = None
    if args.seed_summary:
        measured_seed = json.loads(args.seed_summary.read_text(encoding='utf-8'))
        if (measured_seed['case'] != instance.name
                or measured_seed['total_delay'] != initial_result.total_delay):
            raise ValueError('seed summary does not match the initial solution')
        seed_runtime = measured_seed['runtime_s']
    # No negotiated seed rerun and no public submitted routes: use our own saved result.
    sr = StagedRouter(instance, layer_opt=False, via_opt=False, final_opt=False)
    sr.install_checked_submission(seed)
    hashes = {}
    for filename in ['experiments/staged_router.py', 'experiments/trunk_lns.py',
                     'experiments/lns_repair.py', 'experiments/run_lns_experiment.py']:
        contents = (ROOT / filename).read_bytes()
        hashes[filename] = hashlib.sha256(contents).hexdigest()
        (output / ('source_' + Path(filename).name)).write_bytes(contents)
    seed.save(str(output / 'initial.sol.json'))
    write_json(output / 'run.json', dict(case=str(args.case.resolve()),
            initial_solution=str(args.initial_sol.resolve()),
            initial_delay=initial_result.total_delay, source_hashes=hashes,
            python=platform.python_version(), status='running'))
    print(f'Initial: legal=True delay={initial_result.total_delay}', flush=True)

    def checkpoint(router, record):
        router.current_submission().save(str(output / f'{instance.name}.sol.json'))
        print(f"ACCEPTED saving={record['accepted_gain']} "
              f"total_delay={sum(t.delay for t in router.trees.values())}", flush=True)

    report = improve(sr, budget_s=args.budget, group_budget_s=args.group_budget,
                     max_nodes=args.max_nodes, max_group=args.max_group,
                     max_groups=args.max_groups, on_accept=checkpoint,
                     conflict_avoidance=args.conflict_avoidance)
    solution_path = output / f'{instance.name}.sol.json'
    sr.current_submission().save(str(solution_path))
    result = check(instance, Submission.load(str(solution_path)))
    if not result.legal or result.total_delay > initial_result.total_delay:
        raise RuntimeError('final independent legality or incumbent check failed')
    runtime = time.perf_counter() - started
    report.update(case=instance.name, legal=result.legal, total_delay=result.total_delay,
                  full_experiment_runtime_s=runtime, source_hashes=hashes,
                  initial_solution=str(args.initial_sol.resolve()),
                  seed_runtime_s=seed_runtime,
                  end_to_end_runtime_s=seed_runtime + runtime if seed_runtime is not None else None,
                  baseline_delay=args.baseline_delay,
                  ratio=args.baseline_delay / result.total_delay if args.baseline_delay else None)
    write_json(output / f'{instance.name}.check.json', result.to_dict())
    write_json(output / 'summary.json', report)
    write_json(output / 'runtime.json', {
        instance.name: seed_runtime + runtime if seed_runtime is not None else runtime,
    })
    write_json(output / 'meta.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'hardware': platform.processor(), 'threads': 1,
        'measurement': ('seed generation plus warm-start experiment' if seed_runtime is not None
                        else 'warm-start experiment only, excludes seed generation'),
    })
    manifest = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    manifest.update(status='complete', legal=result.legal, runtime_s=runtime)
    write_json(output / 'run.json', manifest)
    print(f'Final: legal={result.legal} delay={result.total_delay} '
          f'saving={report["saving"]} experiment_runtime_s={runtime:.3f}', flush=True)


if __name__ == '__main__':
    main()
