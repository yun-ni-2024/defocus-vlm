# Running on Runpod

Runpod rents GPU machines ("pods") by the hour. We use it when the CSLab GPUs are busy. The code is the same as on the cluster; only the setup differs: there is no Slurm, and data, models and results live on a Runpod network volume instead of `/w/20261`.

## 1. What we use

| | Choice | Why |
|---|---|---|
| Storage | one network volume, 100 GB, in data center `EU-RO-1` | it outlives every pod: data, environment and results stay when a pod is stopped or deleted |
| GPU pod | Secure Cloud, 1 × RTX 4090, in the same data center, volume mounted at `/workspace` | network volumes attach only to Secure Cloud pods in their own data center |
| CPU pod | same data center and volume, no GPU | for downloads and installs, which need no GPU and cost much less |
| Image | official template "CUDA 13.2.0 · Ubuntu 24.04" (`runpod/pytorch:2.2.0-cu1320-torch2150-ubuntu2404`) | only provides the system, SSH and Python; our own environment is installed on the volume |
| Access | SSH, port `22/tcp`, with a key registered in the Runpod account settings | |

Prices on 2026-11-25: RTX 4090 Secure Cloud $0.89 per hour while the pod runs; the volume is billed per GB from creation until deletion, whether or not a pod uses it.

## 2. Layout on the volume

```
/workspace/
  venv/                  Python environment
  defocus-vlm.git/       bare git repository: the code is pushed here (section 4)
  defocus-vlm/           working copy of the code; runs start here
  csc2529/               DEFOCUS_ROOT: data/, models/, runs/, logs/
```

Everything outside `/workspace` is lost when the pod stops.

## 3. One-time setup on a new volume

On a CPU pod with the volume, after the code is on it (section 4):

```bash
# environment
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
cd /workspace/defocus-vlm
pip install --no-cache-dir -r requirements.txt

# data (about 70 GB) and model weights, under DEFOCUS_ROOT
export DEFOCUS_ROOT=/workspace/csc2529
nohup python3 scripts/download_hdc.py all > /workspace/download_hdc.log 2>&1 &
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct', revision='895c3a49bc3fa70a340399125c650a463535e71c', local_dir='/workspace/csc2529/models/Qwen2-VL-2B-Instruct')"

# when the download has finished
python scripts/check_data.py
```

The same rules as on the cluster apply: after pulling a commit that changed `requirements.txt`, run `pip install --no-cache-dir -r requirements.txt` again.

## 4. Getting the code onto the pod

The repository is private, so the pod does not clone it from GitHub. Instead, the code is pushed over SSH from a computer that has the repository, into a bare repository on the volume, and the pod's working copy pulls from there. Commits, branches and history arrive unchanged, so `meta.json` records the same commit as anywhere else.

**Once per volume**, on the pod:

```bash
git init --bare /workspace/defocus-vlm.git
git clone /workspace/defocus-vlm.git /workspace/defocus-vlm
```

**Every time code changes**, after committing locally:

1. On your computer, push the branch to the pod. The address and port are in the pod's **Connect** panel (SSH over exposed TCP); they change every time a pod starts.

   ```bash
   git push ssh://root@<address>:<port>/workspace/defocus-vlm.git <branch>
   ```

   If the SSH key is not your default one, add `-c core.sshCommand="ssh -i ~/.ssh/<key>"` after `git`.

2. On the pod, bring the working copy to that branch:

   ```bash
   cd /workspace/defocus-vlm
   git fetch origin
   git checkout <branch>
   git reset --hard origin/<branch>
   ```

3. Check that it is clean and on the right commit before starting a final run (final runs need `"dirty": false`):

   ```bash
   git status --short
   git log --oneline -1
   ```

Never edit code on the pod: make every change locally, commit it, and push it again. The pod's working copy is only a copy.

## 5. Running

From the working copy on the GPU pod:

```bash
cd /workspace/defocus-vlm
bash runpod/run.sh train_c_real scripts/train_lora.py configs/train_c_real.yaml
```

`runpod/run.sh` is the counterpart of `slurm/gpu.sbatch`: it activates the environment, sets `DEFOCUS_ROOT`, `HF_HOME` and `TORCH_HOME`, and starts the script in the background, so it keeps running after you disconnect. Its output goes to `/workspace/csc2529/logs/<name>_<time>.out`:

```bash
tail -f /workspace/csc2529/logs/train_c_real_*.out
nvidia-smi
```

Run folders are written to `/workspace/csc2529/runs/`, as on the cluster. A final run is copied into the repository's `assets/results/` the same way (`pipeline.md`).

## 6. Costs and safety

- **Stop the pod when nothing is running.** A stopped pod bills no GPU time; the network volume keeps its data and keeps billing for its size.
- **Keep the account balance above zero.** When it reaches zero every pod stops, and a volume whose storage charges stay unpaid can be deleted, with no way to recover the data.
- **Copy results off the pod.** Final run folders go into the repository; anything else needed later should be copied to your computer or the cluster.
- Do not store passwords or tokens on a pod.
