"""Training loop for method (c): LoRA on the vision encoder, trained by feature distillation."""
import csv
import time

import torch

from src.adapt.distill import add_lora, distill_loss
from src.adapt.pairs import endless_batches, split_pairs
from src.data.image import load_tif, to_pil
from src.eval.metrics import char_accuracy
from src.utils.run import defocus_root
from src.vlm.qwen import QwenReader


def validate(reader, samples, image_cfg, batch_size):
    """Read the blurred validation photographs with the current adapter and score them.

    Args:
        reader: QwenReader whose vision encoder carries the adapter being trained.
        samples: list of Sample held out for validation.
        image_cfg: the config's image section (white_percentile, long_side).
        batch_size: images per VLM call.

    Returns:
        List of dicts, one per image: step, font, number, truth, prediction, accuracy.
    """
    rows = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        # same preprocessing as the evaluation pipeline
        images = [to_pil(load_tif(s.blurred, image_cfg["white_percentile"]), image_cfg["long_side"])
                  for s in batch]
        preds = reader.read(images)
        # score with the same metric as the evaluation (whitespace ignored)
        for s, p in zip(batch, preds):
            rows.append({"step": s.step, "font": s.font, "number": s.number, "truth": s.text,
                         "prediction": p, "accuracy": round(char_accuracy(p, s.text), 4)})
    return rows


def write_csv(path, rows):
    """Write a list of dicts as a CSV file with one column per key.

    Args:
        path: output file.
        rows: list of dicts with the same keys.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def learning_rate(it, base, warmup):
    """Learning rate at an iteration: linear warm-up, then constant.

    Args:
        it: iteration, starting at 1.
        base: learning rate after the warm-up.
        warmup: number of warm-up iterations; 0 means none.

    Returns:
        Float learning rate.
    """
    return base * min(1.0, it / warmup) if warmup else base


def train(cfg, run):
    """Train a LoRA adapter as the config says and write everything into the run folder.

    Writes into `run`:
      log.csv: iteration, training loss, learning rate, validation accuracy, seconds since start
      val/it<N>.csv: the validation predictions at iteration N (N = 0 is the original encoder)
      adapters/it<N>/: the adapter at iteration N, loadable as model.lora in an evaluation config

    Args:
        cfg: the training config (see configs/train_c_real.yaml).
        run: the run folder created by new_run().
    """
    m, d, lo, t, v = cfg["model"], cfg["data"], cfg["lora"], cfg["train"], cfg["validation"]
    torch.manual_seed(t["seed"])

    # load the frozen model through the class the evaluation uses, so reading works the same
    reader = QwenReader(defocus_root() / m["path"], dtype=m["dtype"], min_pixels=m["min_pixels"],
                        max_pixels=m["max_pixels"], max_new_tokens=m["max_new_tokens"],
                        prompt=m["prompt"], device=m.get("device", "cuda"))
    for p in reader.model.parameters():
        p.requires_grad_(False)
    # put the adapter into the encoder; reader.read() then uses it as well
    student = add_lora(reader.model.model.visual, lo["rank"], lo["alpha"], lo["dropout"],
                       lo["target_modules"])
    reader.model.model.visual = student
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=0.0)

    # training and validation pairs
    train_set, val_set = split_pairs(d["source"], d.get("steps"), d["val_every"])
    print(f"{len(train_set)} training pairs, {len(val_set)} validation pairs", flush=True)
    batches = endless_batches(train_set, cfg["image"], t["batch_size"], t["seed"], t.get("workers", 0))

    (run / "val").mkdir()
    (run / "adapters").mkdir()
    log_file = open(run / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["iteration", "loss", "learning_rate", "val_accuracy", "seconds"])
    t0 = time.time()

    def checkpoint(it, loss, lr):
        """Validate, save the predictions and the adapter, and log one line.

        Args:
            it: current iteration (0 = before training).
            loss: training loss of this iteration, or None before training.
            lr: current learning rate, or None before training.
        """
        student.eval()
        rows = validate(reader, val_set, cfg["image"], v["batch_size"])
        write_csv(run / "val" / f"it{it}.csv", rows)
        acc = sum(r["accuracy"] for r in rows) / len(rows)
        # before training the adapter is zero, so there is nothing to save
        if it > 0:
            student.save_pretrained(run / "adapters" / f"it{it}")
        log.writerow([it, "" if loss is None else round(loss, 6), "" if lr is None else lr,
                      round(acc, 4), round(time.time() - t0)])
        log_file.flush()
        print(f"iteration {it}: validation accuracy {acc:.3f}", flush=True)
        student.train()

    # starting point: the original encoder (a new adapter changes nothing)
    checkpoint(0, None, None)

    for it in range(1, t["iterations"] + 1):
        # learning-rate schedule
        lr = learning_rate(it, t["learning_rate"], t["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        # one optimisation step on a batch of pairs
        sharp, blurred = next(batches)
        loss = distill_loss(student, reader.proc, reader.size, sharp, blurred, reader.device,
                            reader.model.dtype)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # validate at fixed intervals and at the end; otherwise log the loss every 10 iterations
        if it % v["every"] == 0 or it == t["iterations"]:
            checkpoint(it, loss.item(), lr)
        elif it % 10 == 0:
            log.writerow([it, round(loss.item(), 6), lr, "", round(time.time() - t0)])
            log_file.flush()
            print(f"iteration {it}: loss {loss.item():.5f}, {time.time() - t0:.0f} s", flush=True)
    log_file.close()
