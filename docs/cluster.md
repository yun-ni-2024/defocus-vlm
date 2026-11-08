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

The home folder holds only about 10 GB (section 7). Code and Slurm output files stay in the home folder; everything large goes to the personal storage folder `/w/20261/YOUR_USERNAME`. Every MScAC student has this folder, and every machine (`comps0`, GPU nodes, CPU nodes) sees it.

```
~/csc2529/                       home folder, small files only
  defocus-vlm/                   this repository
  logs/                          Slurm output files
/w/20261/YOUR_USERNAME/
  miniforge3/                    conda and the project environment (section 4)
  csc2529/                       DEFOCUS_ROOT
    data/hdc2021/                HDC2021, downloaded once, treated as read-only (section 5)
    models/                      pretrained weights (section 5)
    runs/                        one folder per experiment run: config, git commit, logs, checkpoints, predictions
```

The environment variable `DEFOCUS_ROOT` points to `/w/20261/YOUR_USERNAME/csc2529`. The code finds `data/`, `models/` and `runs/` from it, so no path is written into the code. When `DEFOCUS_ROOT` is not set, the scripts use this same location.

Data, weights and results are never committed to the repository.

## 4. Environment

Each person has their own environment in their own storage folder. Conda only provides Python 3.11; every package comes from `requirements.txt` through pip.

Run the installation on a CPU node, not on `comps0`:

```bash
srun --partition=cpunodes -c 4 --mem=8G -t 2:00:00 --pty bash --login
```

Install Miniforge (a conda distribution) into `/w/20261/$USER/miniforge3`. This does not change `~/.bashrc`:

```bash
wget -O /tmp/miniforge_$USER.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge_$USER.sh -b -p /w/20261/$USER/miniforge3
rm /tmp/miniforge_$USER.sh
```

Create the environment and install the packages:

```bash
/w/20261/$USER/miniforge3/bin/conda create -y -n defocus-vlm python=3.11
source /w/20261/$USER/miniforge3/bin/activate defocus-vlm
cd ~/csc2529/defocus-vlm
pip install --no-cache-dir -r requirements.txt
conda clean -y -a
```

Run these lines in every new shell and at the top of every Slurm job. The last two keep the download caches of Hugging Face and PyTorch out of the home folder:

```bash
source /w/20261/$USER/miniforge3/bin/activate defocus-vlm
export DEFOCUS_ROOT=/w/20261/$USER/csc2529
export HF_HOME=$DEFOCUS_ROOT/models/hf_home
export TORCH_HOME=$DEFOCUS_ROOT/models/torch_home
```

Rules:

- Packages are added when code first needs them. Add the package with a fixed version (`name==x.y.z`) to `requirements.txt` in the same commit as the code that first imports it.
- After pulling a commit that changed `requirements.txt`, run `pip install --no-cache-dir -r requirements.txt` again.
- Never install packages inside a job: it is slow, holds the GPU while doing nothing, and can silently change versions between runs.
- Versions match the pilot runs. Miniforge with the environment takes about 6.5 GB. PyTorch is the CUDA 13.0 build, which the driver on the GPU nodes supports.

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

For each zip, `scripts/download_hdc.py` downloads it (resuming a partial download), checks its MD5 against the value Zenodo publishes, extracts it, and deletes the zip. If the job stops, submit it again: zips already extracted are skipped. The full download took about 3.5 hours (4 to 10 MB/s) and takes 69 GB once extracted.

Result, under `/w/20261/YOUR_USERNAME/csc2529/data/hdc2021/`:

```
test/
  steps_0_to_4/CAM01_focused/    sharp photos of steps 0 to 4
  steps_0_to_4/CAM02_blurred/    the same scenes, defocused
  steps_5_to_9/  steps_10_to_14/  steps_15_to_19/
train/
  step0/Times/CAM01/             sharp photos
  step0/Times/CAM02/             the same scenes, defocused
  step0/Verdana/CAM01/  step0/Verdana/CAM02/
  step1/ ... step19/
```

- Test set: in each camera folder, every step has 40 text images (each with a `.txt` transcription), 15 natural images, 1 QR code and 3 PSF calibration targets. File names start with `focusStep_N_`.
- Training set: in each camera folder there are 100 text images (each with a `.txt` transcription) and 3 PSF calibration targets.
- The `__MACOSX` folders come from the zip files and can be ignored.

### Model weights

The reader is `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face (4.2 GB), fixed to Hub commit `895c3a49bc3fa70a340399125c650a463535e71c` so that everyone uses identical weights. Download it with the environment activated:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct', revision='895c3a49bc3fa70a340399125c650a463535e71c', local_dir='/w/20261/$USER/csc2529/models/Qwen2-VL-2B-Instruct')"
```

Weights used by deepinv (DRUNet, RAM) will be added when that code is written.

### Using the shared copy instead of downloading

The data and the Qwen2-VL weights are already downloaded in `/w/20261/yni/csc2529`, and every cluster user can read (but not change) them. Instead of downloading your own copy, link to them from your own `DEFOCUS_ROOT`, so the code finds them at the usual paths:

```bash
mkdir -p /w/20261/$USER/csc2529/models
ln -s /w/20261/yni/csc2529/data /w/20261/$USER/csc2529/data
ln -s /w/20261/yni/csc2529/models/Qwen2-VL-2B-Instruct /w/20261/$USER/csc2529/models/Qwen2-VL-2B-Instruct
```

| | Shared or your own |
|---|---|
| HDC2021 data, Qwen2-VL weights | shared, read-only; this saves 3.5 hours and 74 GB, and everyone uses identical files |
| conda environment (section 4) | your own: installing packages writes into it |
| `runs/`, and the caches `HF_HOME` and `TORCH_HOME` | your own: jobs write there |

- Link only the Qwen2-VL folder, not the whole `models/` folder. The caches live under `models/`, and writing them into someone else's folder fails.
- The shared folders must not be moved or deleted while anyone links to them.
- Files written to `/w/20261` can be read by every cluster user by default, including results in `runs/`.

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

## 7. Cluster facts (checked on 2026-11-05 to 2026-11-08)

- Partition `gpunodes`. Longest job: 5 days; a job without `-t` gets 12 hours. Jobs are not preempted.
- Without `-c` and `--mem`, a GPU job gets only 1 core and 1 GB of memory, so always ask for what you need.
- GPUs:
  - RTX 4090, 24 GB: 4 nodes (`gpunode4`, `5`, `33`, `34`), 32 cores and about 120 GB of memory each. Our main GPU; there are few, so you may have to wait.
  - RTX A4500, 20 GB: 16 nodes, but only 4 cores and 29 GB of memory each. Good for debugging and evaluation when the 4090s are busy.
  - RTX A6000, 48 GB: 2 nodes. RTX A2000 and A4000 nodes also exist.
- GPU nodes can reach the internet (Zenodo and Hugging Face were checked), so data and weights can be downloaded inside a job.
- The home directory is a network folder, the same on every machine. Each user can store only about 10 GB there: a download stopped with "Disk quota exceeded" at about 11 GB. The `quota` command is not installed. When the quota is full, every write fails, including the job's own output file, so the job stops without an error message.
- `/w/20261/YOUR_USERNAME` and `/w/20262/YOUR_USERNAME` are personal storage folders created for every MScAC student. Each sits on a 1.5 TB volume shared by about 120 students; the previous year's folders still exist, so they last at least a year. The personal limit there is unknown: about 80 GB were stored without an error (checked 2026-11-08). We use `/w/20261` only.
- GPU driver 595.71 (CUDA 13.2).
- Qwen2-VL-2B in bfloat16 loads in about 10 s and reads one image with at most 4.5 GB of GPU memory (checked on an RTX 4090, 2026-11-08). For inference only, the 20 GB `rtx_a4500` is more than enough.
- Each machine also has its own local disk: `/tmp`, and `/nobackup` (which points to `/var/tmp`); on `gpunode29` about 390 GB were free. Other machines cannot see it, so it only serves as a fast cache within one job.
- On GPU nodes, `/scratch` is a temporary disk shared by all GPU nodes (2.2 TB, 95% full). Its folders are named after their deletion date, such as `expires-2026-Nov-16`; do not keep anything there.
- No conda, Docker or Apptainer is installed. The system Python is 3.14.
