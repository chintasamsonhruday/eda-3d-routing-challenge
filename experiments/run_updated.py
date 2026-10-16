"""Run a router prototype, optionally substituting only its case and report baseline."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3d.checker import check
from m3d.model import Submission


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_source(source_bytes, case=None, baseline_delay=None):
    """Adapt a source snapshot's input/report without changing routing logic.

    The original prototype remains untouched. There must be exactly one literal
    Instance.load call for the case override; refusing ambiguity prevents the
    benchmark silently testing an unintended input.
    """
    if case is None and baseline_delay is None:
        return source_bytes, []
    tree = ast.parse(source_bytes.decode('utf-8-sig'))
    adaptations = []
    if case is not None:
        loads = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'load'
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == 'Instance'
                 and node.args and isinstance(node.args[0], ast.Constant)
                 and isinstance(node.args[0].value, str)]
        if len(loads) != 1:
            raise ValueError('case override requires exactly one literal Instance.load call')
        previous = loads[0].args[0].value
        loads[0].args[0] = ast.Constant(str(Path(case).resolve()))
        adaptations.append(dict(kind='instance_path', previous=previous,
                                value=str(Path(case).resolve())))
    if baseline_delay is not None:
        # This prototype embeds the case01 baseline in its final print only.
        reports = [node for node in tree.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
                   and any(isinstance(part, ast.Constant) and isinstance(part.value, str)
                           and 'Final Result:' in part.value for part in ast.walk(node))]
        if len(reports) != 1:
            raise ValueError('baseline override requires one Final Result print')
        for node in ast.walk(reports[0]):
            if isinstance(node, ast.Constant):
                if node.value == 11938:
                    node.value = baseline_delay
                elif isinstance(node.value, str):
                    node.value = node.value.replace('baseline=11938', f'baseline={baseline_delay}')
        adaptations.append(dict(kind='report_baseline', value=baseline_delay))
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree) + '\n').encode('utf-8'), adaptations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case", type=Path,
                        help="override the snapshot's literal Instance.load input only")
    parser.add_argument("--baseline-delay", type=int)
    args = parser.parse_args()
    source = args.script.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = source.read_bytes()
    executed_bytes, adaptations = prepare_source(source_bytes, args.case, args.baseline_delay)
    snapshot = output / ("source_" + source.name)
    snapshot.write_bytes(executed_bytes)
    if adaptations:
        (output / ('original_' + source.name)).write_bytes(source_bytes)
    write_json(output / "run.json", {
        "script": str(source), "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "executed_sha256": hashlib.sha256(executed_bytes).hexdigest(),
        "adaptations": adaptations,
        "python": platform.python_version(), "platform": platform.platform(),
        "status": "running",
    })
    started = time.perf_counter()
    # Execute the source snapshot so concurrent prototype edits cannot alter this run.
    namespace = runpy.run_path(str(snapshot), run_name="__main__")
    elapsed = time.perf_counter() - started
    sr = namespace["sr"]
    submission = namespace.get("sub") or sr.current_submission()
    instance = namespace.get("inst") or sr.inst
    solution_path = output / f"{instance.name}.sol.json"
    submission.save(str(solution_path))
    result = check(instance, Submission.load(str(solution_path)))
    write_json(output / f"{instance.name}.check.json", result.to_dict())
    write_json(output / "summary.json", {
        "case": instance.name, "legal": result.legal, "total_delay": result.total_delay,
        "runtime_s": elapsed, "script": str(source),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "executed_sha256": hashlib.sha256(executed_bytes).hexdigest(),
        "adaptations": adaptations, "baseline_delay": args.baseline_delay,
        "ratio": (args.baseline_delay / result.total_delay
                  if args.baseline_delay is not None and result.legal else None),
        "solution": str(solution_path), "reasons": result.reasons,
    })
    write_json(output / "runtime.json", {instance.name: elapsed})
    write_json(output / "meta.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "hardware": platform.processor(), "threads": 1,
        "measurement": "full prototype execution including initialization",
    })
    manifest = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    manifest.update(status='complete', legal=result.legal, runtime_s=elapsed)
    write_json(output / 'run.json', manifest)
    print(f"SAVED: legal={result.legal} delay={result.total_delay} full_runtime_s={elapsed:.3f}", flush=True)
    return 0 if result.legal else 3


if __name__ == "__main__":
    raise SystemExit(main())
