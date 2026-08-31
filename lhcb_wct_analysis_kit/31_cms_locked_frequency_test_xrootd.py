from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "31_cms_locked_frequency_test.py"

spec = importlib.util.spec_from_file_location("cms_locked_lhcb", TARGET)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.REMOTE_BASE = (
    "root://eospublic.cern.ch//eos/opendata/lhcb/upload/"
    "opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/"
    "outputs/real-production"
)

_uproot_open = mod.uproot.open


def _open_native_xrootd(path, *args, **kwargs):
    if isinstance(path, str) and path.startswith("root://"):
        kwargs["handler"] = mod.uproot.source.xrootd.XRootDSource
    return _uproot_open(path, *args, **kwargs)


mod.uproot.open = _open_native_xrootd

if __name__ == "__main__":
    raise SystemExit(mod.main())
