# Troubleshooting

Problems we actually ran into, and what to do.

## A job stopped and its output file ends without an error

The storage quota is probably full. When no more can be written, even the job's own output file cannot grow, so the error message is lost. The home folder holds only about 10 GB: keep data, environments, weights and runs in `/w/20261/$USER` (`cluster.md`, section 3). Check what takes space with `du -sh ~/* ~/.[!.]*`.

## The job's output file is not where I asked

`~` is not expanded inside `#SBATCH` lines. Write the full path (`/u/YOUR_USERNAME/...`), or give `-o` on the `sbatch` command line, where the shell expands `~`. The folder must already exist.

## `srun ... --immediate`: "Requested nodes are busy"

All GPUs of that type are taken. Leave out `--immediate` to wait in the queue, or ask for `rtx_a4500`, which is enough for inference (`cluster.md`, section 7). `slurm_report -g` shows what is free.

## `sacct` fails with "Connection refused"

The Slurm accounting database is not reachable from `comps0`. Use `squeue -u $USER` while the job runs, and its output file afterwards.

## Warning: "Passing `min_pixels` and `max_pixels` to a processor call is deprecated"

Raised by transformers 5.19 when the Qwen2-VL processor gets `min_pixels`/`max_pixels`. It still works; new code should pass `size={'longest_edge': ..., 'shortest_edge': ...}` instead.

Pass that `size` with every processor call, as `src/vlm/qwen.py` does. Given to `AutoProcessor.from_pretrained`, it is silently ignored and the image keeps its full size (a 1024 x 633 image became 46 x 74 patches instead of 32 x 52). Check with the `image_grid_thw` the processor returns.

## Warning: "You are sending unauthenticated requests to the HF Hub"

Harmless. The weights are public; a Hugging Face token only raises the download rate limit.

## Git on Windows: "LF will be replaced by CRLF"

Harmless. The repository stores Unix line endings, and `.gitattributes` keeps them for scripts even in a Windows checkout.

## `ssh comps0.cs`: "This machine is currently not available"

`comps0` is closed for maintenance. `comps1.cs` and `comps2.cs` are equivalent: the same home folder, the same storage and the same Slurm queue (`comps3` was closed too on 2026-11-19).

## A job waits with reason `(Priority)` or `(Resources)`

All GPUs are taken. `--gres=gpu:1` accepts any free GPU type instead of waiting for a 4090.

## A shell script run over `ssh` skips its later commands

A command inside the script (for example `ssh` or `git` asking for input) read the rest of the script from standard input. Add `< /dev/null` to such commands.
