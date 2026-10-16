"""Run the guarded optimized hard pipeline with incremental progress reports."""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_hard_suite_parallel import solve_case
from experiments.safe_negotiated_repair import atomic_json


def worker(args):
    case, output, stages, seconds, seed = args
    started = time.perf_counter()
    name = Path(case).stem
    atomic_json(Path(output) / (name + '.status.json'),
                dict(case=name, status='running', pid=os.getpid(),
                     started_utc=datetime.now(timezone.utc).isoformat()))
    result = solve_case((case, output))
    if result.get('legal') and stages:
        from experiments.run_multidimensional import run_case
        from experiments.safe_negotiated_repair import save_checked_best
        from m3d.model import Instance, Submission
        refined = Path(output) / 'refinement' / name
        try:
            detail = run_case(case, Path(output)/(name+'.sol.json'), refined,
                              stages.split(','), seconds, seed)
            inst = Instance.load(case)
            sub = Submission.load(str(refined/(name+'.sol.json')))
            stored, _ = save_checked_best(inst, sub, Path(output)/(name+'.sol.json'))
            result.update(delay=stored.total_delay, legal=stored.legal,
                          refinement=detail, time=time.perf_counter()-started)
        except Exception as error:
            # The complete seed is already safely stored; retain it and expose
            # refinement failure instead of losing the entire case result.
            result['refinement_error'] = f'{type(error).__name__}: {error}'
    atomic_json(Path(output) / (name + '.status.json'),
                dict(result, status='completed', finished_utc=datetime.now(timezone.utc).isoformat()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--refine-stages', default='', help='optional comma-separated multidimensional stages')
    parser.add_argument('--refine-seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.refine_seconds <= 0:
        parser.error('positive worker count and refinement budget required')
    if args.refine_stages:
        from experiments.multidimensional_lns import profile
        for stage in args.refine_stages.split(','):
            if stage != 'portfolio':profile(stage)
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'launch.json').exists():
        raise ValueError('Use a fresh output directory for this benchmark')
    manifest = json.loads((ROOT / 'benchmarks_hard/suite.json').read_text())
    sources = ['run_hard_suite_parallel.py', 'experiments/safe_negotiated_repair.py',
               'experiments/staged_router.py', 'experiments/trunk_lns.py',
               'experiments/run_guarded_hard_suite.py',
               'experiments/multidimensional_lns.py', 'experiments/run_multidimensional.py']
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        (snapshot / Path(name).name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    atomic_json(output / 'launch.json', dict(pid=os.getpid(), workers=args.workers,
        started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        cases=[c['name'] for c in manifest['cases']], refinement_stages=args.refine_stages,
        refinement_seconds=args.refine_seconds, seed=args.seed))
    tasks = [(str(ROOT / 'benchmarks_hard' / c['instance_file']), str(output),
              args.refine_stages, args.refine_seconds, args.seed)
             for c in reversed(manifest['cases'])]
    results = []
    started = time.perf_counter()
    print(f'Started {len(tasks)} hard cases, {args.workers} workers; output={output}', flush=True)
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(result)
            atomic_json(output / 'progress.json', dict(completed=len(results),
                total=len(tasks), elapsed_s=time.perf_counter()-started, results=results))
            print(f"{len(results)}/{len(tasks)} {result['case']}: legal={result.get('legal')} "
                  f"delay={result.get('delay')} time={result.get('time')} error={result.get('error')}", flush=True)
    results.sort(key=lambda r: r['case'])
    baselines = {c['name']: c['baseline_total'] for c in manifest['cases']}
    complete = len(results) == len(tasks) and all(r.get('legal') for r in results)
    score = math.exp(sum(math.log(baselines[r['case']]/r['delay']) for r in results)/len(tasks)) if complete else 0
    summary = dict(aggregate=score, total_delay=sum(r.get('delay', 0) or 0 for r in results),
        total_time=time.perf_counter()-started, n_legal=sum(bool(r.get('legal')) for r in results),
        n_cases=len(tasks), results=results, finished_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
