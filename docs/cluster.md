# Running on the CSLab cluster

All experiments run on the CSLab GPU cluster. Replace `YOUR_USERNAME` with your CSLab username. Sections marked TODO will be filled in once the corresponding step works.

## 1. Logging in

You need a CSLab account with GPU access. It is separate from your UTORid and from the CS Teaching Labs account.

The cluster has three kinds of machines:

- `apps0` (`cs.toronto.edu`): the department's login server. Only for getting in.
- `comps0`: the entry point of the compute cluster. Edit files, submit jobs and check on them here.
- `gpunodeNN`: the machines with GPUs. You never log in to them directly; Slurm gives you one when you ask for it (section 6).

Do not run heavy computation on `apps0` or `comps0`; dozens of people share them.

```bash
ssh YOUR_USERNAME@cs.toronto.edu
ssh comps0.cs
```

The hop from `apps0` to `comps0` does not ask for a password.

Optional, to log in from your own computer without typing a password: create a key on your computer and append its public half to `~/.ssh/authorized_keys` on the cluster (this step asks for your password once).

```bash
ssh-keygen -t ed25519 -f ~/.ssh/cslab_ed25519 -C "cslab"
cat ~/.ssh/cslab_ed25519.pub | ssh YOUR_USERNAME@cs.toronto.edu "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

On Windows PowerShell, use `$HOME\.ssh\cslab_ed25519` in the first command and `type $HOME\.ssh\cslab_ed25519.pub | ssh ...` in the second.

## 2. Getting the code

Each team member keeps their own clone in their own cluster home, at `~/csc2529/defocus-vlm`. Do not share one working copy; two people editing the same folder overwrite each other.

The repository is private, so the cluster needs permission to read it. Two options:

- Use the SSH key of your own GitHub account (add the cluster's public key under GitHub → Settings → SSH and GPG keys).
- Or use a deploy key, which gives access to this repository only:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/github_defocus_vlm -C "cslab deploy key for defocus-vlm"
cat ~/.ssh/github_defocus_vlm.pub
```

Add the printed key in the repository under Settings → Deploy keys, with "Allow write access" ticked. Then clone and tell git to keep using that key for this repository:

```bash
mkdir -p ~/csc2529 && cd ~/csc2529
GIT_SSH_COMMAND="ssh -i ~/.ssh/github_defocus_vlm -o IdentitiesOnly=yes" git clone git@github.com:yun-ni-2024/defocus-vlm.git
cd defocus-vlm
git config core.sshCommand "ssh -i ~/.ssh/github_defocus_vlm -o IdentitiesOnly=yes"
git config pull.ff only
```

Keeping the copies in sync:

- `git pull` before you start editing; commit and `git push` soon after.
- Do not edit the same file on your computer and on the cluster at the same time.
- A queued job reads the code when it starts, not when it is submitted. If you pull new code while a job is waiting, it runs the new code. Every run records the git commit it used.
- Move code between your computer and the cluster through git, not by copying files. Files copied from Windows can carry Windows line endings, which break shell scripts.

## 3. Folder layout on the cluster

```
~/csc2529/
  defocus-vlm/   this repository
  data/          HDC2021, downloaded once, treated as read-only
  models/        pretrained weights
  runs/          one folder per experiment run: config, git commit, logs, checkpoints, predictions
  logs/          Slurm output files
```

The environment variable `DEFOCUS_ROOT` points to `~/csc2529`. The code finds `data/`, `models/` and `runs/` from it, so no path is written into the code.

Data, weights and results are never committed to the repository.

## 4. Environment

TODO. The cluster has no conda and no Docker or Apptainer; the plan is a Miniforge installation in your home with one environment for the project, described by `environment.yml` in this repository.

## 5. Data and model weights

TODO.

## 6. Running jobs

See which GPUs are free:

```bash
slurm_report -g
```

**Interactive session**, for debugging. You get a shell on a GPU node; it ends when you `exit`, when the time runs out, or when your SSH connection drops. The GPU stays reserved for you until then, so `exit` as soon as you are done.

```bash
srun --partition=gpunodes -c 4 --mem=16G --gres=gpu:rtx_4090:1 -t 30 --pty bash --login
```

**Batch job**, for anything long. It runs on its own after you submit it; you can log out.

```bash
sbatch slurm/TEMPLATE.sbatch configs/EXPERIMENT.yaml
```

TODO: the templates in `slurm/`.

Checking and stopping jobs:

```bash
squeue -u $USER
tail -f ~/csc2529/logs/OUTPUT_FILE
scancel JOB_ID
```

## 7. Cluster facts (checked on 2026-11-05)

- Partition `gpunodes`. Longest job: 5 days; a job without `-t` gets 12 hours. Jobs are not preempted.
- Without `-c` and `--mem`, a GPU job gets only 1 core and 1 GB of memory, so always ask for what you need.
- GPUs:
  - RTX 4090, 24 GB: 4 nodes (`gpunode4`, `5`, `33`, `34`), 32 cores and about 120 GB of memory each. Our main GPU; there are few, so you may have to wait.
  - RTX A4500, 20 GB: 16 nodes, but only 4 cores and 29 GB of memory each. Good for debugging and evaluation when the 4090s are busy.
  - RTX A6000, 48 GB: 2 nodes. RTX A2000 and A4000 nodes also exist.
- GPU nodes can reach the internet (Zenodo and Hugging Face were checked), so data and weights can be downloaded inside a job.
- The home directory is a network folder, the same on every machine. It sits on a 600 GB volume shared with other users; the personal quota is unknown.
- Each GPU node also has a local disk (`/tmp` had about 170 GB free on `gpunode33`, and there is `/scratch`). It is faster than the home directory but not kept between jobs: copy data there at the start of a job if reading speed matters.
- No conda, Docker or Apptainer is installed. The system Python is 3.14.
