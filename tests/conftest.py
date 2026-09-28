"""Point the backend at a non-existent checkpoint before it is imported, so the
API tests always exercise the default random-init model regardless of whether a
real checkpoints/model.npz exists on disk. conftest is imported before any test
module, i.e. before backend.main runs its module-level model load."""
from __future__ import annotations

import os

os.environ["MICROGPT_CHECKPOINT"] = "__tests_force_untrained__.npz"
