"""Run 30_run_group_holdout.py against the stable CERN EOS XRootD endpoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "30_run_group_holdout.py"

spec = importlib.util.spec_from_file_location("run_group_holdout", TARGET)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.REMOTE_BASE = (
    "root://eospublic.cern.ch//eos/opendata/lhcb/upload/"
    "opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/"
    "outputs/real-production"
)

if __name__ == "__main__":
    mod.main()
