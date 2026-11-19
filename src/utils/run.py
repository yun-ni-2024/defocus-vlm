"""Experiment configs and run folders: every run records what it used."""
import getpass
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def defocus_root():
    """Return $DEFOCUS_ROOT, defaulting to /w/20261/<username>/csc2529.

    Returns:
        Path of the folder that holds data/, models/ and runs/.
    """
    return Path(os.environ.get("DEFOCUS_ROOT", os.path.join("/w/20261", getpass.getuser(), "csc2529")))


def load_config(path):
    """Read a YAML experiment config.

    Args:
        path: config file.

    Returns:
        The config as a dict.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def git_state():
    """Describe the code version: commit hash and whether there were uncommitted changes.

    Returns:
        Dict with "commit" (str) and "dirty" (bool); "unknown" if git is not available.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def new_run(config_path, name):
    """Create runs/<name>_<time>/ and store the config and the run's metadata in it.

    Args:
        config_path: the config file used, copied into the run folder.
        name: short run name, usually the config's "name".

    Returns:
        Path of the new run folder.
    """
    run = defocus_root() / "runs" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True)
    # the exact config, and where/with what code the run happened
    shutil.copy(config_path, run / "config.yaml")
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), **git_state()}
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run
