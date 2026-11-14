# Code conventions

## Layout

- `src/`: all project code, including evaluation and analysis helpers.
- `scripts/`: thin command-line entry points that call into `src/`.
- `configs/`: one YAML file per experiment; an experiment is run as `script + config`.
- `slurm/`: sbatch templates.
- `docs/`: documentation, the findings log and design notes.

There is no `pyproject.toml` and nothing is installed with `pip install -e`. Each entry point adds the repository root to the import path and imports from `src`:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.restore import wiener
```

## Paths

No absolute path is written into the code. Data, weights and results are found from the environment variable `DEFOCUS_ROOT` (`cluster.md`, section 3).

## Writing code

- Every function has a docstring with its goal, `Args` and `Returns`.
- Each step inside a function body has a short comment.
- A new package goes into `requirements.txt`, with a fixed version, in the same commit as the code that first imports it.

## Runs

- Every run writes into its own folder under `$DEFOCUS_ROOT/runs/`, with the config it used, the git commit, its log and its outputs.
- Results are never committed. What is worth keeping for the report goes into `findings.md`, with a pointer to the run folder.

## Git

- Pull before editing, push soon after.
- Move code between machines through git only (Windows line endings break shell scripts).
