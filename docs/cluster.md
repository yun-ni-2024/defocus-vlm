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

> **Storage is not settled yet.** The home quota is only about 10 GB (section 7), while the data, the environment and the weights together need about 95 GB. Sections 4 and 5 assume everything lives in the home folder; the paths will change once we get more space. The commands in section 4 have not yet been run on the cluster.

## 4. Environment

Each person has their own environment in their own home. Conda only provides Python 3.11; every package comes from `requirements.txt` through pip.

Run the installation on a CPU node, not on `comps0`:

```bash
srun --partition=cpunodes -c 4 --mem=8G -t 2:00:00 --pty bash --login
```

Install Miniforge (a conda distribution) into `~/miniforge3`. This does not change `~/.bashrc`:

```bash
wget -O /tmp/miniforge_$USER.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge_$USER.sh -b -p ~/miniforge3
rm /tmp/miniforge_$USER.sh
```

Create the environment and install the packages:

```bash
~/miniforge3/bin/conda create -y -n defocus-vlm python=3.11
source ~/miniforge3/bin/activate defocus-vlm
cd ~/csc2529/defocus-vlm
pip install --no-cache-dir -r requirements.txt
conda clean -y -a
```

Activate it in every new shell and at the top of every Slurm job:

```bash
source ~/miniforge3/bin/activate defocus-vlm
```

Rules:

- Packages are added when code first needs them. Add the package with a fixed version (`name==x.y.z`) to `requirements.txt` in the same commit as the code that first imports it.
- After pulling a commit that changed `requirements.txt`, run `pip install --no-cache-dir -r requirements.txt` again.
- Never install packages inside a job: it is slow, holds the GPU while doing nothing, and can silently change versions between runs.
- Versions match the pilot runs. The environment takes about 8 to 10 GB.

## 5. Data and model weights

### HDC2021

The Helsinki Deblur Challenge 2021 data is on Zenodo (CC BY 4.0):

| Split | Zenodo record | Files | Size |
|---|---|---|---|
| training | 4916176 (version 1.1) | 20 zips, one per defocus step 0 to 19 | 49.6 GB |
| test | 5713637 | 4 zips, five steps each | 13.3 GB |

Do not use record 4772280: it is an older version of the training set with steps 0 to 9 only.

Download with a job on a CPU node (submit from the repository root; the argument is `test`, `train` or `all`):

```bash
cd ~/csc2529/defocus-vlm
sbatch -o ~/csc2529/logs/%x_%j.out slurm/download_hdc.sbatch all
```

For each zip, `scripts/download_hdc.py` downloads it (resuming a partial download), checks its MD5 against the value Zenodo publishes, extracts it, and deletes the zip. If the job stops, submit it again: zips already extracted are skipped. At about 10 MB/s, the test set takes about 25 minutes. Extracted files take about 1.3 times the zip size.

Result:

```
~/csc2529/data/hdc2021/
  test/steps_0_to_4/CAM01_focused/    sharp photos
  test/steps_0_to_4/CAM02_blurred/    the same scenes, defocused
  test/steps_5_to_9/ ...
  train/ ...
  _downloads/                         partial downloads, empty when finished
```

In the test set, each step has 40 text images (each with a `.txt` transcription), 15 natural images, 1 QR code and 3 PSF calibration targets, in both camera folders. File names start with `focusStep_N_`.

### Model weights

The reader is `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face (about 4.5 GB), stored in `~/csc2529/models/Qwen2-VL-2B-Instruct`. Download it inside the environment with the version fixed to one Hub commit, so that everyone uses identical weights:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct', revision='COMMIT', local_dir='$HOME/csc2529/models/Qwen2-VL-2B-Instruct')"
```

TODO: the commit hash, once the weights are downloaded. Weights used by deepinv (DRUNet, RAM) will be added when that code is written.

## 6. Running jobs

### Why jobs are submitted

There are about 25 GPU machines and many users. Nobody logs in to a GPU machine and starts a program directly: two programs on the same GPU would run out of memory together. Instead, the scheduler, **Slurm**, hands out machines:

- you tell Slurm what to run and how many resources it needs;
- Slurm finds a machine with those resources free and starts your program there;
- if nothing is free, your job waits in a queue and starts on its own when space opens up.

"Submitting a job" means handing Slurm that request. See which GPUs are free right now:

```bash
slurm_report -g
```

### What to ask for

| Option | Meaning | Typical value for us |
|---|---|---|
| `--partition=gpunodes` | which group of machines | always `gpunodes` |
| `--gres=gpu:TYPE:1` | which GPU and how many | `gpu:rtx_4090:1` |
| `-c N` | CPU cores | 4 to 8 |
| `--mem=SIZE` | system memory (not GPU memory) | `16G` to `32G` |
| `-t TIME` | time limit, in minutes or as `H:MM:SS` | `30` for debugging, `24:00:00` for training |
| `-J NAME` | a name for the job | e.g. `c_lora_sim` |
| `-o FILE` | where the program's output goes; `%x` becomes the job name, `%j` the job ID | `/u/YOUR_USERNAME/csc2529/logs/%x_%j.out` |

The resources are both reserved for you and a hard limit: nobody else can use them while your job runs, but a job that uses more memory than it asked for is killed, and a job that runs past its time limit is stopped. Without `-c` and `--mem`, a job gets only 1 core and 1 GB.

GPU types accepted by `--gres` (checked on 2026-11-05): `rtx_4090`, `rtx_a4500`, `rtx_a6000`, `rtx_a4000`, `rtx_a2000`. If all 4090s are busy, `rtx_a4500` (20 GB) is the usual fallback.

### Interactive session: `srun`

Asks for resources and gives you a shell on the GPU machine. Your prompt changes to something like `YOUR_USERNAME@gpunode33`, and you type commands as if you were sitting at that machine.

```bash
srun --partition=gpunodes -c 4 --mem=16G --gres=gpu:rtx_4090:1 -t 30 --pty bash --login
```

Use it for debugging and short tests. It ends when you `exit`, when the time runs out, or when your SSH connection drops, so it cannot be used for long training. The GPU stays reserved for you until it ends: `exit` as soon as you are done.

### Batch job: `sbatch`

Write the resource request and the commands into one file and hand it to Slurm. You can log out afterwards; the job runs on its own. A minimal example, `hello.sbatch`:

```bash
#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 10
#SBATCH -J hello
#SBATCH -o /u/YOUR_USERNAME/csc2529/logs/%x_%j.out

hostname
nvidia-smi -L
echo "done"
```

- Lines starting with `#SBATCH` are the resource request from the table above.
- The other lines are the commands to run. In real jobs they activate the environment and run a script from this repository.
- Write the `-o` path in full. `~` is not expanded inside `#SBATCH` lines, and the folder must already exist.

Submit it:

```bash
sbatch hello.sbatch
```

Slurm replies `Submitted batch job 229485`; the number is the job ID. This exact example ran on `gpunode5` and its output file contained the node name, the GPU and `done`.

In the project, the resource requests live in templates in `slurm/` and the experiment is chosen by a config file:

```bash
sbatch slurm/TEMPLATE.sbatch configs/EXPERIMENT.yaml
```

TODO: the templates in `slurm/`.

### After submitting

A job goes through these states:

```
PENDING (waiting) -> RUNNING (on some gpunode) -> COMPLETED
                                                  or FAILED (the program stopped with an error)
                                                  or TIMEOUT (stopped at the time limit)
```

List your jobs that have not finished yet, with their state and machine:

```bash
squeue -u $USER
```

Everything the program prints, including error messages, goes to the `-o` file. Follow it while the job runs (Ctrl-C stops watching, not the job):

```bash
tail -f /u/YOUR_USERNAME/csc2529/logs/hello_229485.out
```

Cancel a job:

```bash
scancel JOB_ID
```

### How a batch job differs from running a program on your own computer

1. It cannot ask you anything while it runs. Every setting must be fixed beforehand, which is why experiments are described by config files.
2. You do not see errors as they happen. After submitting, check that the job started and read the first lines of its output file.
3. It runs on another machine, but it reads and writes the same `~/csc2529` as every other machine, so its results appear on `comps0` straight away.

## 7. Cluster facts (checked on 2026-11-05)

- Partition `gpunodes`. Longest job: 5 days; a job without `-t` gets 12 hours. Jobs are not preempted.
- Without `-c` and `--mem`, a GPU job gets only 1 core and 1 GB of memory, so always ask for what you need.
- GPUs:
  - RTX 4090, 24 GB: 4 nodes (`gpunode4`, `5`, `33`, `34`), 32 cores and about 120 GB of memory each. Our main GPU; there are few, so you may have to wait.
  - RTX A4500, 20 GB: 16 nodes, but only 4 cores and 29 GB of memory each. Good for debugging and evaluation when the 4090s are busy.
  - RTX A6000, 48 GB: 2 nodes. RTX A2000 and A4000 nodes also exist.
- GPU nodes can reach the internet (Zenodo and Hugging Face were checked), so data and weights can be downloaded inside a job.
- The home directory is a network folder, the same on every machine. Each user can store only about 10 GB there: a download stopped with "Disk quota exceeded" at about 11 GB. The `quota` command is not installed. When the quota is full, every write fails, including the job's own output file, so the job stops without an error message.
- GPU driver 595.71 (CUDA 13.2).
- Each GPU node also has a local disk (`/tmp` had about 170 GB free on `gpunode33`, and there is `/scratch`). It is faster than the home directory but not kept between jobs: copy data there at the start of a job if reading speed matters.
- No conda, Docker or Apptainer is installed. The system Python is 3.14.
