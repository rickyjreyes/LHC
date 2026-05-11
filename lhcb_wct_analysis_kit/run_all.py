#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py

Orchestration script for the LHCb / WCT analysis pipeline.

Default behavior:
    Runs the paper-grade pipeline in dependency order.

Useful commands:
    python run_all.py
    python run_all.py --fast
    python run_all.py --full
    python run_all.py --controls
    python run_all.py --from 19
    python run_all.py --only 13,17,28
    python run_all.py --continue-on-error
    python run_all.py --dry-run

Notes:
    - Some stages require prior outputs. For example, 20/21/22 require 19,
      and 12 requires 09d.
    - Control tests are skipped unless --controls is passed or data_control/
      contains ROOT files.
    - Heavy null tests can take a long time, especially on CPU.

Requires Python 3.9+.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parent
RUN_LOG_DIR = ROOT / "outputs_run_all"
RUN_LOG_DIR.mkdir(exist_ok=True, parents=True)


@dataclass(frozen=True)
class Stage:
    key: str
    script: str
    label: str
    group: str
    heavy: bool = False
    optional: bool = False
    control: bool = False
    needs_root: bool = True  # whether this stage requires ROOT files in data/
    requires: Tuple[str, ...] = ()
    outputs_hint: Tuple[str, ...] = ()


STAGES: List[Stage] = [
    # ------------------------------------------------------------------
    # 0. Intake / readiness checks
    # ------------------------------------------------------------------
    Stage(
        key="00",
        script="00_inspect_root.py",
        label="Inspect first ROOT file and tree layout",
        group="intake",
    ),
    Stage(
        key="01",
        script="01_check_branches.py",
        label="Check q2, angular, and useful branch readiness",
        group="intake",
    ),
    Stage(
        key="03",
        script="03_bootstrap_scan.py",
        label="Bootstrap diagnostic on yield-side log-FFT peaks",
        group="intake",
        heavy=True,
        outputs_hint=("bootstrap_peaks.csv", "bootstrap_summary.json"),
    ),
    Stage(
        key="04",
        script="04_angle_branch_report.py",
        label="Search for direct angular branch candidates",
        group="intake",
        optional=True,
    ),

    # ------------------------------------------------------------------
    # 1. Yield-side repaired log-cos / winding pipeline
    # ------------------------------------------------------------------
    Stage(
        key="09d",
        script="09d_two_mode_kde_baseline_polar_cupy.py",
        label="KDE-baseline bounded-Poisson two-mode scan (paper Table 1)",
        group="yield",
        heavy=True,
        outputs_hint=("outputs_logcos_poisson_twomode_kde_polar",),
    ),
    Stage(
        key="12",
        script="12_wct_integer_winding_scan.py",
        label="Discrete active-domain integer-winding scan (paper Fig. 3)",
        group="yield",
        heavy=True,
        requires=("09d",),
        outputs_hint=("outputs_wct_integer_winding",),
    ),
    Stage(
        key="13",
        script="13_wct_koide_trig_comb_scan_cupy.py",
        label="WCT Koide / trig comb scan (paper Fig. 4)",
        group="yield",
        heavy=True,
        outputs_hint=("outputs_wct_koide_comb",),
    ),
    Stage(
        key="16",
        script="16_wct_vs_smqft_likelihood_test_cupy.py",
        label="WCT comb vs smooth SM/QFT-like likelihood test",
        group="yield",
        heavy=True,
        outputs_hint=("outputs_wct_vs_smqft",),
    ),
    Stage(
        key="17",
        script="17_wct_sideband_subtracted_comb_test_cupy.py",
        label="Sideband-subtracted WCT comb test (cupy)",
        group="yield",
        heavy=True,
        outputs_hint=("outputs_wct_sideband_subtracted",),
    ),
    Stage(
        key="28",
        script="28_sideband.py",
        label="Sideband-subtracted residual test (paper Sec. 20.2)",
        group="yield",
        heavy=True,
        outputs_hint=("outputs_sideband_subtracted",),
    ),

    # ------------------------------------------------------------------
    # 2. Well-first Koide and cross-region diagnostics
    # ------------------------------------------------------------------
    Stage(
        key="19",
        script="19_koide_well.py",
        label="Well-first Koide raw-well scan",
        group="well_first",
        heavy=True,
        outputs_hint=("outputs_wct_well_first_koide/well_first_wells.csv",),
    ),
    Stage(
        key="20",
        script="20_koide_proof.py",
        label="Well-first Koide geometry proof / null test",
        group="well_first",
        heavy=True,
        needs_root=False,  # consumes outputs of stage 19
        requires=("19",),
        outputs_hint=("outputs_wct_well_proof",),
    ),
    Stage(
        key="21",
        script="21_cross_region_scaling_phase_test.py",
        label="Cross-region scaling and phase-coherence test (paper Sec. 15)",
        group="well_first",
        heavy=True,
        needs_root=False,
        requires=("19",),
        outputs_hint=("outputs_wct_cross_region_scaling",),
    ),
    Stage(
        key="22",
        script="22_cross_regional_stability_test.py",
        label="Cross-region scaling stability sweep (paper Fig. 5)",
        group="well_first",
        heavy=True,
        needs_root=False,
        requires=("19",),
        outputs_hint=("outputs_wct_cross_region_stability",),
    ),
    Stage(
        key="24",
        script="24_locked_branch_amplitude.py",
        label="Locked-branch amplitude-cap ladder (paper Sec. 15.4)",
        group="well_first",
        heavy=True,
        outputs_hint=("outputs_wct_locked_branch_amplitude_ladder",),
    ),
    Stage(
        key="lwcr",
        script="locked_winding_cross_region.py",
        label="Locked-winding cross-region test (paper Sec. 15.3)",
        group="well_first",
        heavy=True,
        outputs_hint=("outputs_wct_locked_winding_cross_region",),
    ),

    # ------------------------------------------------------------------
    # 3. Veto covariance / active-domain invariance
    # ------------------------------------------------------------------
    Stage(
        key="25",
        script="25_veto_window_covariance_test.py",
        label="Veto-window covariance / active-domain invariance test (paper Sec. 16)",
        group="covariance",
        heavy=True,
        outputs_hint=("outputs_wct_veto_covariance",),
    ),
    Stage(
        key="26",
        script="26_veto_covariance.py",
        label="Alternate/report-style veto covariance Koide test",
        group="covariance",
        heavy=True,
        optional=True,
        outputs_hint=("outputs_veto_covariance_koide",),
    ),

    # ------------------------------------------------------------------
    # 4. Controls
    # ------------------------------------------------------------------
    Stage(
        key="05",
        script="05_control_compare.py",
        label="Legacy signal/control FFT comparison",
        group="control",
        optional=True,
        control=True,
        needs_root=False,  # has its own data_signal/data_control checks
    ),
    Stage(
        key="27",
        script="27_control_channel_blind_test.py",
        label="Blind control-channel / reconstruction-control test (paper Sec. 20.1)",
        group="control",
        heavy=True,
        optional=True,
        control=True,
        needs_root=False,  # uses data_control/, not data/
        outputs_hint=("outputs_control_blind",),
    ),
    Stage(
        key="29",
        script="29_charm_tail_trimmed_control.py",
        label="Charm-tail / trimmed control test",
        group="control",
        heavy=True,
        optional=True,
        outputs_hint=("outputs_charm_trimmed_control",),
    ),
]


FAST_KEYS: Set[str] = {
    "00", "01", "04",
    "19", "20", "21", "22",
    "25", "27",
}

DEFAULT_SKIP_KEYS: Set[str] = {
    # Historical / legacy optional control, not paper-grade by default.
    "05",
    # 26 overlaps with 25. Use --full or --only 26 to run it.
    "26",
}


# Exit codes
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED_REQUIRED = 2
EXIT_BAD_USAGE = 3


class RunFailure(RuntimeError):
    pass


def has_any_root(patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if glob.glob(str(ROOT / pat)):
            return True
    return False


def stage_by_key(key: str) -> Stage:
    for stage in STAGES:
        if stage.key == key:
            return stage
    valid = ", ".join(s.key for s in STAGES)
    raise KeyError(f"Unknown stage key '{key}'. Valid keys: {valid}")


def normalize_key_list(text: Optional[str]) -> Optional[Set[str]]:
    if not text:
        return None
    out: Set[str] = set()
    for raw in text.replace(";", ",").split(","):
        key = raw.strip()
        if key:
            out.add(key)
    return out


def select_stages(args: argparse.Namespace) -> List[Stage]:
    only = normalize_key_list(args.only)
    skip = normalize_key_list(args.skip) or set()

    # Validate user-supplied keys early with a clean error.
    all_valid = {s.key for s in STAGES}
    for label, keys in (("--only", only), ("--skip", skip)):
        if keys:
            bad = sorted(keys - all_valid)
            if bad:
                raise SystemExit(
                    f"{label} contains unknown stage key(s): {', '.join(bad)}\n"
                    f"Valid keys: {', '.join(s.key for s in STAGES)}"
                )

    if only:
        # --only overrides --fast / --full / --group / --from
        if args.fast or args.full or args.group or args.from_key:
            print("Note: --only overrides --fast/--full/--group/--from.")
        stages = [s for s in STAGES if s.key in only]
    else:
        stages = list(STAGES)

        if args.fast:
            stages = [s for s in stages if s.key in FAST_KEYS]

        if not args.full:
            stages = [s for s in stages if s.key not in DEFAULT_SKIP_KEYS]

        if args.group:
            valid_groups = {s.group for s in STAGES}
            groups = {g.strip() for g in args.group.split(",") if g.strip()}
            bad_groups = sorted(groups - valid_groups)
            if bad_groups:
                raise SystemExit(
                    f"--group contains unknown group(s): {', '.join(bad_groups)}\n"
                    f"Valid groups: {', '.join(sorted(valid_groups))}"
                )
            stages = [s for s in stages if s.group in groups]

        if args.from_key:
            if args.from_key not in all_valid:
                raise SystemExit(
                    f"--from key '{args.from_key}' is not a valid stage. "
                    f"Valid keys: {', '.join(s.key for s in STAGES)}"
                )
            if args.from_key not in {s.key for s in stages}:
                raise SystemExit(
                    f"--from key '{args.from_key}' is valid but not in the current plan "
                    f"(after --fast/--full/--group filters). "
                    f"Plan contains: {', '.join(s.key for s in stages)}"
                )
            seen = False
            selected = []
            for s in stages:
                if s.key == args.from_key:
                    seen = True
                if seen:
                    selected.append(s)
            stages = selected

    stages = [s for s in stages if s.key not in skip]

    # Controls are opt-in unless control files exist.
    control_data_exists = has_any_root(
        ["data_control/*.root", "data_control/*.dvntuple.root"]
    )
    if not args.controls and not control_data_exists:
        stages = [s for s in stages if not s.control]

    return stages


def check_script_exists(stage: Stage) -> bool:
    return (ROOT / stage.script).exists()


def check_stage_inputs(
    stage: Stage, completed_keys: Set[str], args: argparse.Namespace
) -> Tuple[bool, str]:
    if not check_script_exists(stage):
        return False, f"missing script: {stage.script}"

    if not args.ignore_dependencies:
        missing_deps = [dep for dep in stage.requires if dep not in completed_keys]
        if missing_deps:
            return False, f"missing dependency stages: {', '.join(missing_deps)}"

    # ROOT data check: declarative on the Stage now.
    if stage.needs_root:
        if not has_any_root(["data/*.root", "data/*.dvntuple.root"]):
            return False, "no ROOT files found in data/"

    # Specific output dependencies that aren't captured by `requires`.
    if stage.key in {"20", "21", "22"} and not args.ignore_dependencies:
        if not (ROOT / "outputs_wct_well_first_koide/well_first_wells.csv").exists():
            return False, (
                "missing outputs_wct_well_first_koide/well_first_wells.csv; "
                "run stage 19 first"
            )

    if stage.key == "27":
        if not has_any_root(["data_control/*.root", "data_control/*.dvntuple.root"]):
            return False, "no control ROOT files found in data_control/"

    if stage.key == "05":
        if not has_any_root(["data_signal/*.root", "data_signal/*.dvntuple.root"]):
            return False, "no signal ROOT files found in data_signal/"
        if not has_any_root(["data_control/*.root", "data_control/*.dvntuple.root"]):
            return False, "no control ROOT files found in data_control/"

    return True, "ok"


def run_stage(stage: Stage, args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"
    
    script_path = ROOT / stage.script
    log_path = RUN_LOG_DIR / f"{stage.key}_{script_path.stem}.log"

    cmd = [sys.executable, str(script_path)]
    if stage.key == "27" and args.control_mode:
        cmd.extend(["--mode", args.control_mode])

    print("\n" + "=" * 96)
    print(f"RUNNING {stage.key}: {stage.script}")
    print(f"LABEL   : {stage.label}")
    print(f"GROUP   : {stage.group}")
    print(f"LOG     : {log_path.relative_to(ROOT)}")
    print("=" * 96)

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"COMMAND: {' '.join(cmd)}\n")
        log.write(f"START: {time.ctime(start)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        end = time.time()
        log.write(f"\nEND: {time.ctime(end)}\n")
        log.write(f"RETURN_CODE: {rc}\n")
        log.write(f"SECONDS: {end - start:.2f}\n")

    result = {
        "key": stage.key,
        "script": stage.script,
        "label": stage.label,
        "group": stage.group,
        "return_code": rc,
        "seconds": round(end - start, 3),
        "log": str(log_path.relative_to(ROOT)),
        "outputs_hint": list(stage.outputs_hint),
    }

    if rc != 0:
        raise RunFailure(
            f"Stage {stage.key} failed with return code {rc}. See {log_path}"
        )

    return result


def _json_safe(obj):
    """Coerce argparse Namespace values to JSON-safe forms."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def write_manifest(
    results: List[dict],
    skipped: List[dict],
    failed: List[dict],
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "created_at": time.ctime(),
        "command": " ".join(sys.argv),
        "args": _json_safe(vars(args)),
        "results": results,
        "skipped": skipped,
        "failed": failed,
        "summary": {
            "ran": len(results),
            "skipped": len(skipped),
            "failed": len(failed),
        },
    }
    path = RUN_LOG_DIR / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def print_plan(stages: List[Stage]) -> None:
    print("\nExecution plan:")
    if not stages:
        print("  No stages selected.")
        return
    for i, s in enumerate(stages, 1):
        flags = []
        if s.heavy:
            flags.append("heavy")
        if s.optional:
            flags.append("optional")
        if s.control:
            flags.append("control")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {i:02d}. {s.key:>4}  {s.script:<52} {s.label}{flag_text}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LHCb/WCT analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0  all selected stages ran successfully\n"
            "  1  one or more stages failed\n"
            "  2  one or more non-optional stages were skipped\n"
            "  3  bad command-line usage (unknown key, etc.)\n"
        ),
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run a reduced but meaningful pipeline.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include optional overlapping/legacy stages.",
    )
    parser.add_argument(
        "--controls", action="store_true",
        help="Run control stages even if they are optional.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["rare_like", "jpsi_peak"],
        default=None,
        help="Mode passed to 27_control_channel_blind_test.py.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated stage keys to run, e.g. --only 13,17,28",
    )
    parser.add_argument(
        "--skip",
        help="Comma-separated stage keys to skip.",
    )
    parser.add_argument(
        "--from", dest="from_key",
        help="Start from a stage key in the selected plan, e.g. --from 19",
    )
    parser.add_argument(
        "--group",
        help="Comma-separated groups: intake,yield,well_first,covariance,control",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan but do not run scripts.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep running later stages after a failure.",
    )
    parser.add_argument(
        "--ignore-dependencies", action="store_true",
        help="Do not enforce dependency/output checks.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        stages = select_stages(args)
    except SystemExit as exc:
        # argparse raises SystemExit(int) and prints its own message.
        # Our own raise SystemExit("...") carries a string we need to show.
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return EXIT_BAD_USAGE
        code = exc.code if isinstance(exc.code, int) else EXIT_BAD_USAGE
        return code if code != 0 else EXIT_BAD_USAGE
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE

    print_plan(stages)

    if args.dry_run:
        print("\nDry run only. No scripts executed.")
        return EXIT_OK

    completed_keys: Set[str] = set()
    results: List[dict] = []
    skipped: List[dict] = []
    failed: List[dict] = []

    for stage in stages:
        ok, reason = check_stage_inputs(stage, completed_keys, args)
        if not ok:
            item = {"key": stage.key, "script": stage.script, "reason": reason}
            skipped.append(item)
            print(f"\nSKIPPING {stage.key}: {stage.script} -- {reason}")
            if not stage.optional and not args.continue_on_error:
                manifest_path = write_manifest(results, skipped, failed, args)
                print(
                    f"\nStopped: required stage {stage.key} could not run "
                    f"({reason}).\nManifest: {manifest_path.relative_to(ROOT)}"
                )
                return EXIT_SKIPPED_REQUIRED
            continue

        try:
            result = run_stage(stage, args)
            results.append(result)
            completed_keys.add(stage.key)
        except RunFailure as exc:
            item = {"key": stage.key, "script": stage.script, "error": str(exc)}
            failed.append(item)
            print(f"\nFAILED {stage.key}: {exc}")
            if not args.continue_on_error:
                manifest_path = write_manifest(results, skipped, failed, args)
                print(f"\nStopped. Manifest: {manifest_path.relative_to(ROOT)}")
                return EXIT_FAILED

    manifest_path = write_manifest(results, skipped, failed, args)

    print("\n" + "=" * 96)
    print("RUN_ALL SUMMARY")
    print("=" * 96)
    print(f"Ran    : {len(results)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed : {len(failed)}")
    print(f"Manifest: {manifest_path.relative_to(ROOT)}")

    if failed:
        return EXIT_FAILED
    required_skipped = [
        item for item in skipped
        if not stage_by_key(item["key"]).optional
    ]
    if required_skipped:
        return EXIT_SKIPPED_REQUIRED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())