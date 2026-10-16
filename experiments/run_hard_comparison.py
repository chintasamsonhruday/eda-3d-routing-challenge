"""Serially compare an unchanged updated prototype and warm-start V1 LNS."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3d.checker import check
from m3d.model import Instance, Submission
from m3d.scorer import score_submission_set


LNS_FILES = (
    "experiments/staged_router.py", "experiments/trunk_lns.py",
    "experiments/lns_repair.py", "experiments/run_lns_experiment.py",
)
CSV_FIELDS = (
    "case", "baseline_delay", "updated_status", "updated_legal", "updated_delay",
    "updated_runtime_s", "lns_status", "lns_legal", "lns_delay", "lns_runtime_s",
    "lns_incremental_runtime_s", "lns_saving", "updated_error", "lns_error",
)


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in LNS_FILES}


def verified_result(case, suite: Path, folder: Path, variant: str,
                    script_sha: str, lns_hashes: dict, seed_delay=None):
    """Only reuse matching measured results after checking the actual route."""
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    name = case["name"]
    if summary["case"] != name or summary.get("legal") is not True:
        raise ValueError("summary is missing legality or names a different case")
    if variant == "updated":
        if summary.get("sha256") != script_sha:
            raise ValueError("updated source hash differs from this run")
        runtime = summary["runtime_s"]
    else:
        if summary.get("source_hashes") != lns_hashes:
            raise ValueError("LNS source hashes differ from this run")
        if summary.get("initial_delay") != seed_delay:
            raise ValueError("LNS seed delay differs from the updated route")
        runtime = summary["end_to_end_runtime_s"]
    if not isinstance(runtime, (int, float)) or runtime < 0:
        raise ValueError("missing or invalid measured end-to-end runtime")
    inst = Instance.load(str(suite / case["instance_file"]))
    solution = folder / f"{name}.sol.json"
    result = check(inst, Submission.load(str(solution)))
    saved_check = json.loads((folder / f"{name}.check.json").read_text(encoding="utf-8"))
    if (not result.legal or result.total_delay != summary["total_delay"]
            or not saved_check.get("legal")
            or saved_check.get("total_delay") != result.total_delay):
        raise ValueError("saved route, checker report, and summary do not match")
    if variant == "lns" and result.total_delay > seed_delay:
        raise ValueError("LNS worsened its seed")
    return summary, result


def run_child(command, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (folder / "stdout.log").open("w", encoding="utf-8") as stdout:
        with (folder / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"child exit code {completed.returncode}; see {folder / 'stderr.log'}")
    return time.perf_counter() - started


def promote_result(folder: Path, target: Path, name: str):
    for extension in ("sol.json", "check.json"):
        shutil.copy2(folder / f"{name}.{extension}", target / f"{name}.{extension}")


def refresh_reports(output: Path, suite: Path, suite_manifest: dict,
                    run_manifest: dict, rows: list, status: str):
    scores = {}
    for variant in ("updated", "lns"):
        target = output / variant
        runtimes = {row["case"]: row[f"{variant}_runtime_s"] for row in rows
                    if row[f"{variant}_status"] == "complete"}
        write_json(target / "runtime.json", runtimes)
        score = score_submission_set(suite_manifest, str(suite), str(target),
                                     name=variant, runtimes=runtimes).to_dict()
        write_json(target / "score.json", score)
        scores[variant] = score
    report = dict(status=status, suite=str(suite), script=run_manifest["script"],
                  script_sha256=run_manifest["script_sha256"],
                  source_hashes=run_manifest["source_hashes"],
                  lns_parameters=run_manifest["lns_parameters"],
                  cases=rows, updated=scores["updated"], lns=scores["lns"])
    write_json(output / "summary.json", report)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (output / "comparison.csv").write_text(buffer.getvalue(), encoding="utf-8")
    table = ["# Updated router versus V1 LNS", "", f"Status: {status}.", "",
             "| Case | Baseline | Updated delay | Updated seconds | LNS delay | LNS total seconds | Extra seconds | Saving |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def display(value):
        return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    for row in rows:
        table.append("| " + " | ".join(display(row.get(key)) for key in (
            "case", "baseline_delay", "updated_delay", "updated_runtime_s",
            "lns_delay", "lns_runtime_s", "lns_incremental_runtime_s", "lns_saving")) + " |")
    table.extend(["", "Runtime includes initialization; LNS total adds its measured experiment runtime to the updated seed.",
                  "Aggregate is zero until all suite cases have checked legal solutions.", ""])
    for variant, score in scores.items():
        table.append(f"{variant}: {score['n_legal']}/{score['n_cases']} legal; aggregate {score['aggregate']:.6f}.")
    (output / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    run_manifest["status"] = status
    write_json(output / "run.json", run_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks_hard")
    parser.add_argument("--script", type=Path, default=ROOT / "test_multi_target_astar.py")
    parser.add_argument("--resume", action="store_true",
                        help="reuse only source-matching, independently checked results in this output")
    args = parser.parse_args()
    suite = args.suite.resolve()
    script = args.script.resolve()
    output = args.out_dir.resolve()
    suite_manifest = json.loads((suite / "suite.json").read_text(encoding="utf-8"))
    if not suite_manifest.get("cases"):
        parser.error("suite contains no cases")
    source = script.read_bytes()
    script_sha = hashlib.sha256(source).hexdigest()
    hashes = source_hashes()
    lns_parameters = dict(budget=60, group_budget=2, max_nodes=200,
                          max_group=6, max_groups=100, conflict_avoidance=False)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_test_multi_target_astar.py"
    if (output / "run.json").exists():
        if not args.resume:
            parser.error("output already contains a run; use a fresh directory or --resume")
        previous = json.loads((output / "run.json").read_text(encoding="utf-8"))
        if (previous.get("script_sha256") != script_sha
                or previous.get("source_hashes") != hashes
                or previous.get("lns_parameters") != lns_parameters
                or previous.get("suite_sha256") != hashlib.sha256((suite / "suite.json").read_bytes()).hexdigest()
                or not snapshot.exists()
                or hashlib.sha256(snapshot.read_bytes()).hexdigest() != script_sha):
            parser.error("resume source, suite, parameters, or snapshot does not match; use a fresh output")
    elif any(output.iterdir()):
        parser.error("fresh output directory must be empty")
    else:
        snapshot.write_bytes(source)
    run_manifest = dict(status="running", script=str(script), script_sha256=script_sha,
                        suite=str(suite), suite_sha256=hashlib.sha256((suite / "suite.json").read_bytes()).hexdigest(),
                        source_hashes=hashes, lns_parameters=lns_parameters,
                        python=platform.python_version(), platform=platform.platform())
    rows = [dict(case=case["name"], baseline_delay=case["baseline_total"],
                 updated_status="pending", updated_legal=None, updated_delay=None,
                 updated_runtime_s=None, lns_status="pending", lns_legal=None,
                 lns_delay=None, lns_runtime_s=None, lns_incremental_runtime_s=None,
                 lns_saving=None, updated_error=None, lns_error=None)
            for case in suite_manifest["cases"]]
    for variant in ("updated", "lns"):
        target = output / variant
        (target / "_cases").mkdir(parents=True, exist_ok=True)
        # Keep per-case evidence for verified resume, but rebuild promoted files.
        # Rejected reuse or failed reruns must not leave a stale ranked solution.
        for row in rows:
            for extension in ("sol.json", "check.json"):
                (target / f"{row['case']}.{extension}").unlink(missing_ok=True)
        write_json(target / "meta.json", dict(
            python=platform.python_version(), platform=platform.platform(),
            hardware=platform.processor(), threads=1,
            measurement=("full updated prototype including initialization" if variant == "updated"
                         else "updated seed generation plus warm-start V1 LNS"),
            script_sha256=script_sha, source_hashes=hashes))
    refresh_reports(output, suite, suite_manifest, run_manifest, rows, "running")
    for case, row in zip(suite_manifest["cases"], rows):
        name = case["name"]
        updated = output / "updated" / "_cases" / name
        lns = output / "lns" / "_cases" / name
        print(f"START {name} updated", flush=True)
        row["updated_status"] = "running"
        refresh_reports(output, suite, suite_manifest, run_manifest, rows, "running")
        try:
            reusable = False
            if args.resume:
                try:
                    seed_summary, _ = verified_result(case, suite, updated, "updated", script_sha, hashes)
                    reusable = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if not reusable:
                run_child([sys.executable, "-u", str(ROOT / "experiments/run_updated.py"),
                           "--script", str(snapshot), "--case", str(suite / case["instance_file"]),
                           "--baseline-delay", str(case["baseline_total"]),
                           "--out-dir", str(updated)], updated)
                seed_summary, _ = verified_result(case, suite, updated, "updated", script_sha, hashes)
            promote_result(updated, output / "updated", name)
            row.update(updated_status="complete", updated_legal=True,
                       updated_delay=seed_summary["total_delay"], updated_runtime_s=seed_summary["runtime_s"])
            print(f"DONE {name} updated legal=True delay={row['updated_delay']} seconds={row['updated_runtime_s']:.3f}" +
                  (" reused=True" if reusable else ""), flush=True)
        except Exception as error:
            row.update(updated_status="failed", updated_legal=False, updated_error=str(error),
                       lns_status="skipped", lns_error="updated seed failed")
            print(f"FAILED {name} updated: {error}", flush=True)
            refresh_reports(output, suite, suite_manifest, run_manifest, rows, "running")
            continue
        print(f"START {name} LNS", flush=True)
        row["lns_status"] = "running"
        refresh_reports(output, suite, suite_manifest, run_manifest, rows, "running")
        try:
            reusable = False
            if args.resume:
                try:
                    lns_summary, _ = verified_result(case, suite, lns, "lns", script_sha, hashes,
                                                      seed_delay=row["updated_delay"])
                    reusable = True
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if not reusable:
                run_child([sys.executable, "-u", str(ROOT / "experiments/run_lns_experiment.py"),
                           "--case", str(suite / case["instance_file"]),
                           "--initial-sol", str(updated / f"{name}.sol.json"),
                           "--seed-summary", str(updated / "summary.json"),
                           "--out-dir", str(lns), "--baseline-delay", str(case["baseline_total"]),
                           "--budget", "60", "--group-budget", "2", "--max-nodes", "200",
                           "--max-group", "6", "--max-groups", "100"], lns)
                lns_summary, _ = verified_result(case, suite, lns, "lns", script_sha, hashes,
                                                  seed_delay=row["updated_delay"])
            promote_result(lns, output / "lns", name)
            row.update(lns_status="complete", lns_legal=True, lns_delay=lns_summary["total_delay"],
                       lns_runtime_s=lns_summary["end_to_end_runtime_s"],
                       lns_incremental_runtime_s=lns_summary["full_experiment_runtime_s"],
                       lns_saving=row["updated_delay"] - lns_summary["total_delay"])
            print(f"DONE {name} LNS legal=True delay={row['lns_delay']} saving={row['lns_saving']} "
                  f"total_seconds={row['lns_runtime_s']:.3f}" + (" reused=True" if reusable else ""), flush=True)
        except Exception as error:
            row.update(lns_status="failed", lns_legal=False, lns_error=str(error))
            print(f"FAILED {name} LNS: {error}", flush=True)
        refresh_reports(output, suite, suite_manifest, run_manifest, rows, "running")
    status = ("complete" if all(row[f"{variant}_status"] == "complete"
                               for row in rows for variant in ("updated", "lns"))
              else "complete_with_failures")
    refresh_reports(output, suite, suite_manifest, run_manifest, rows, status)
    print(f"FINISHED status={status} output={output}", flush=True)


if __name__ == "__main__":
    main()
