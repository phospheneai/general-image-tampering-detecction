#!/usr/bin/env python3
"""
DAG orchestrator for the dataset pipeline (see packages/mdsconverter/README.md):
validate -> remediate -> convert. Each stage is its own standalone script
with its own config (validate_dataset.py, remediate_dataset.py,
build_mds_dataset.py) and can be toggled on/off independently via
configs/mds/pipeline.yml's stages.<name>.enabled — this script just runs
whichever are enabled, in that fixed order, as subprocesses, stopping if
one fails. Each stage's own output (progress bar included) streams straight
through.

Interactive gate: if stages.remediate.enabled and stages.remediate.interactive
are both true, the pipeline pauses right after validate if (and only if) it
found any failed/corrupted files — prints how many and where the report is,
then asks y/n at the terminal before doing anything else. "yes" runs
remediate for real (--execute, regardless of the config's static execute:
value — the "yes" *is* that confirmation) and then convert; "no" stops the
pipeline immediately, before remediate or convert ever run. Zero failures
found -> no prompt, stages proceed exactly per their configured enabled:
(and remediate's static execute:, if it runs) as normal.

A prompt with no attached terminal (e.g. launched via `< /dev/null` for a
durable background run) can't be answered — that's treated as "no" rather
than hanging or crashing, so a backgrounded run never sits there forever
waiting on unanswerable input.

Usage:
    python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml

    # override configs/mds/pipeline.yml's enabled: flags for a one-off run
    python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml --only validate
    python packages/mdsconverter/run_pipeline.py --config configs/mds/pipeline.yml --skip remediate
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

import yaml

from _forensics_common import load_yaml_config

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_SCRIPT = {
    "validate": "validate_dataset.py",
    "remediate": "remediate_dataset.py",
    "convert": "build_mds_dataset.py",
}
STAGE_ORDER = ["validate", "remediate", "convert"]


def resolve_config_path(stage_cfg: dict, base_dir: str) -> str | None:
    config_path = stage_cfg.get("config")
    if not config_path:
        return None
    if not os.path.isabs(config_path):
        config_path = os.path.normpath(os.path.join(base_dir, config_path))
    return config_path


def run_stage(name: str, stage_cfg: dict, base_dir: str) -> int:
    script = os.path.join(SCRIPTS_DIR, STAGE_SCRIPT[name])
    cmd = [sys.executable, script]

    config_path = resolve_config_path(stage_cfg, base_dir)
    if config_path:
        cmd += ["--config", config_path]
    if stage_cfg.get("split"):
        cmd += ["--split", str(stage_cfg["split"])]
    if name == "remediate" and stage_cfg.get("execute"):
        cmd += ["--execute"]
    cmd += [str(a) for a in (stage_cfg.get("extra_args") or [])]

    print(f"\n{'=' * 70}\n[pipeline] stage '{name}': {' '.join(cmd)}\n{'=' * 70}")
    t0 = time.time()
    # stdin=DEVNULL: none of these stages read stdin themselves, but letting
    # them inherit it is dangerous — validate's multiprocessing.Pool (fork)
    # hands copies of the fd to its workers, and something in that chain
    # reliably drains/closes it, so a later input() in this process (the
    # interactive remediate gate below) hits EOF even on data that was
    # actually piped in. Isolating the child here keeps this process's own
    # stdin intact for its own prompts.
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[pipeline] stage '{name}' {status} in {dt:.1f}s")
    return result.returncode


def count_failed_in_report(report_path: str) -> int:
    if not report_path or not os.path.exists(report_path):
        return 0
    with open(report_path, newline="", encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f) if r.get("status") == "failed")


def confirm(question: str) -> bool:
    """y/n at the terminal. No TTY / closed stdin (e.g. a `< /dev/null`
    background launch) -> can't be answered, so it's treated as "no" rather
    than hanging (EOFError) or crashing the whole pipeline."""
    try:
        while True:
            ans = input(f"{question} [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                return True
            if ans in ("", "n", "no"):
                return False
            print("please answer y or n")
    except EOFError:
        print("\n[pipeline] no terminal attached to answer this prompt — treating as 'no'")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to configs/mds/pipeline.yml.")
    ap.add_argument("--only", nargs="+", choices=STAGE_ORDER, default=None,
                     help="Run only these stages (still in validate/remediate/convert order), "
                          "ignoring each stage's enabled: in the config.")
    ap.add_argument("--skip", nargs="+", choices=STAGE_ORDER, default=None,
                     help="Skip these stages even if enabled: true in the config.")
    ap.add_argument("--yes", action="store_true",
                     help="Auto-confirm the interactive remediate gate instead of prompting "
                          "(same effect as answering 'y') — for scripted/background runs where "
                          "you've already decided to proceed on any failures found.")
    ap.add_argument("--split", default=None,
                     help="Restrict this run to one split (e.g. train), overriding each stage's own "
                          "split: in the config. Applies to validate and convert (the stages that take "
                          "--split); remediate has no split of its own — it just acts on whatever rows "
                          "the (now split-restricted) validate report contains.")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(os.path.abspath(args.config))
    stages_cfg = cfg.get("stages", {})

    plan = []
    for name in STAGE_ORDER:
        stage_cfg = dict(stages_cfg.get(name, {}) or {})
        if args.split and name in ("validate", "convert"):
            stage_cfg["split"] = args.split
        want = (name in args.only) if args.only else bool(stage_cfg.get("enabled", False))
        if args.skip and name in args.skip:
            want = False
        plan.append((name, stage_cfg, want))

    print("[pipeline] plan:")
    for name, _, want in plan:
        print(f"  {name}: {'RUN' if want else 'skip'}")
    if not any(want for _, _, want in plan):
        print("\n[pipeline] nothing enabled — nothing to do. "
              "Set stages.<name>.enabled: true in the config, or pass --only.", file=sys.stderr)
        sys.exit(1)

    plan_by_name = {name: (stage_cfg, want) for name, stage_cfg, want in plan}

    for name in STAGE_ORDER:
        stage_cfg, want = plan_by_name[name]
        if not want:
            continue

        if name == "remediate" and stage_cfg.get("interactive"):
            validate_cfg, validate_ran = plan_by_name.get("validate", ({}, False))
            validate_config_path = resolve_config_path(validate_cfg, base_dir)
            report_path = None
            if validate_config_path and os.path.exists(validate_config_path):
                report_path = load_yaml_config(validate_config_path, path_keys=("report_out",)).get("report_out")
            n_failed = count_failed_in_report(report_path)
            if n_failed > 0:
                print(f"\n[pipeline] validate found {n_failed} corrupted/failed file(s) — see {report_path}")
                proceed = args.yes or confirm(
                    f"Proceed with remediate on these {n_failed} file(s), then convert?"
                )
                if not proceed:
                    print("[pipeline] stopping at your request — remediate and convert were not run")
                    sys.exit(0)
                stage_cfg = {**stage_cfg, "execute": True}
            # n_failed == 0: nothing to approve, proceed with stage_cfg as configured (unchanged)

        rc = run_stage(name, stage_cfg, base_dir)
        if rc != 0:
            print(f"\n[pipeline] stopping — stage '{name}' failed (exit {rc})", file=sys.stderr)
            sys.exit(rc)

    print("\n[pipeline] all enabled stages completed successfully")


if __name__ == "__main__":
    main()
