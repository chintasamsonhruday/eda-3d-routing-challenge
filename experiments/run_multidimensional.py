"""Run selected stages on one instance, preserving its best verified solution."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3d.model import Instance, Submission
from m3d.checker import check
from experiments.multidimensional_lns import Search, profile
from experiments.safe_negotiated_repair import atomic_json, save_checked_best
from experiments.staged_router import StagedRouter


def run_case(case, initial, output, stages, seconds=60, seed=0):
    """Each stage receives an equal share of the total search-time budget."""
    if 'portfolio' in stages and stages != ['portfolio']:
        raise ValueError('portfolio is a complete schedule; use it alone')
    if not stages or seconds <= 0:
        raise ValueError('positive budget and at least one stage required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise ValueError('use a new output directory')
    started = time.perf_counter()
    inst = Instance.load(str(case))
    if initial:
        sub = Submission.load(str(initial))
    else:
        sr = StagedRouter(inst, initial_route='negotiated', polish_passes=2,
                          layer_opt=False, via_opt=False, final_opt=False)
        sub, _ = sr.run()
        if sub is None:
            raise RuntimeError('initial negotiation failed')
    original = check(inst, sub)
    if not original.legal:
        raise ValueError('initial submission is illegal')
    sub.save(str(output/'initial.sol.json'))
    solution = output/(inst.name+'.sol.json')
    save_checked_best(inst, sub, solution)
    engine = Search(inst, sub, seed)
    hashes = {}
    for relative in ['experiments/multidimensional_lns.py', 'experiments/run_multidimensional.py',
                     'experiments/safe_negotiated_repair.py', 'experiments/trunk_lns.py',
                     'experiments/staged_router.py']:
        data = (ROOT/relative).read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        (output/('source_'+Path(relative).name)).write_bytes(data)
    report = dict(status='running', case=inst.name, seed=seed, stages=stages,
                  initial_delay=original.total_delay, initial_source=str(initial),
                  input_sha256=hashlib.sha256(Path(case).read_bytes()).hexdigest(),
                  initial_sha256=hashlib.sha256((output/'initial.sol.json').read_bytes()).hexdigest(),
                  source_sha256=hashes, stage_results=[], python=platform.python_version(),
                  runtime_scope='incremental_from_saved_incumbent' if initial else 'including_seed',
                  search_budget_s=seconds)
    atomic_json(output/'summary.json', report)
    def retain(candidate, delay):
        save_checked_best(inst, candidate, solution)
    schedule = ([('bounds', .4, seed, 'original'),
                 ('sequential', .4, seed+1, 'original'),
                 ('neutral', .2, seed+2, 'best')]
                if stages == ['portfolio'] else
                [(stage, 1/len(stages), seed, 'continue') for stage in stages])
    for stage, fraction, search_seed, source in schedule:
        if source == 'original':
            engine = Search(inst, sub, search_seed)
        elif source == 'best':
            engine = Search(inst, Submission.load(str(solution)), search_seed)
        result = engine.run(seconds*fraction, profile(stage), on_best=retain)
        result['stage'] = stage
        result['search_seed'] = search_seed
        result['start_from'] = source
        report['stage_results'].append(result)
        report['best_delay'] = check(inst, Submission.load(str(solution))).total_delay
        atomic_json(output/'summary.json', report)
    stored, _ = save_checked_best(inst, engine.submission(), solution)
    baseline = None
    manifest_path = Path(case).resolve().parent/'suite.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        baseline = next((c['baseline_total'] for c in manifest['cases'] if c['name']==inst.name), None)
    report.update(status='complete', legal=stored.legal, total_delay=stored.total_delay,
                  saving=original.total_delay-stored.total_delay,
                  runtime_s=time.perf_counter()-started, baseline=baseline,
                  ratio=baseline/stored.total_delay if baseline else None)
    atomic_json(output/'summary.json', report)
    atomic_json(output/'check.json', stored.to_dict())
    return {k:report[k] for k in ['case','seed','stages','initial_delay','legal','total_delay','saving','runtime_s']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--initial-sol', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stages', default='bounds,neutral')
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    stages = args.stages.split(',')
    for stage in stages:
        if stage != 'portfolio':profile(stage)
    print(json.dumps(run_case(args.case,args.initial_sol,args.out,stages,args.seconds,args.seed),indent=2))


if __name__ == '__main__':main()
