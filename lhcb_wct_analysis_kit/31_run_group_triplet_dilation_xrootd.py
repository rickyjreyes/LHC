"""Run stage 31 against the stable CERN EOS XRootD endpoint.

Matches the working stage-30 transport: force uproot's native XRootDSource
instead of the fsspec root:// handler.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "31_run_group_triplet_dilation.py"
spec = importlib.util.spec_from_file_location("run_group_triplet_dilation", TARGET)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

mod.hold.REMOTE_BASE = (
    "root://eospublic.cern.ch//eos/opendata/lhcb/upload/"
    "opendata-lhcb-ntupling-service/analysis-productions/merge-requests/5826/"
    "outputs/real-production"
)

_uproot_open = mod.hold.uproot.open


def _open_native_xrootd(path, *args, **kwargs):
    kwargs["handler"] = mod.hold.uproot.source.xrootd.XRootDSource
    return _uproot_open(path, *args, **kwargs)


mod.hold.uproot.open = _open_native_xrootd

if __name__ == "__main__":
    mod.main()
