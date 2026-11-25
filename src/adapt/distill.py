"""Feature distillation for the vision encoder of Qwen2-VL.

The teacher is the original encoder reading the sharp photograph; the student is the same encoder
with a LoRA adapter reading the blurred photograph of the same scene. Training pulls the student's
visual tokens (the encoder's output after the token merger, i.e. what the language model reads)
towards the teacher's. Teacher and student share the weights: the teacher is the student with its
adapter switched off.
"""
import re

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


def add_lora(visual, rank, alpha, dropout, target_modules):
    """Wrap the vision encoder with a trainable LoRA adapter.

    Args:
        visual: the vision encoder (model.model.visual of Qwen2VLForConditionalGeneration).
        rank: rank of the low-rank update.
        alpha: scaling of the update (the update is multiplied by alpha / rank).
        dropout: dropout on the adapter's input.
        target_modules: names of the linear layers inside each block to adapt, e.g. "attn.qkv".

    Returns:
        PeftModel wrapping the encoder; only the adapter's weights require gradients.
    """
    # match e.g. blocks.12.attn.qkv in every block, but nothing in the token merger
    names = "|".join(re.escape(m) for m in target_modules)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=rf".*blocks\.\d+\.({names})")
    return get_peft_model(visual, cfg)


def visual_tokens(encoder, processor, size, images, device, dtype):
    """Run images through the vision encoder and return its output tokens.

    Args:
        encoder: the vision encoder, with or without adapter.
        processor: the model's AutoProcessor (QwenReader.proc).
        size: pixel limits {"shortest_edge", "longest_edge"} (QwenReader.size), so the images are
            resized exactly as when the model reads them.
        images: list of PIL images.
        device: "cuda" or "cpu".
        dtype: dtype of the model's weights; the pixels are cast to it.

    Returns:
        Tensor (total number of tokens, hidden size): the tokens of all images, one after another.
    """
    # cut the images into patches, as the processor does when the model reads them
    inp = processor.image_processor(images=images, size=size, return_tensors="pt")
    pixels = inp["pixel_values"].to(device, dtype=dtype)
    out = encoder(pixels, grid_thw=inp["image_grid_thw"].to(device))
    # transformers 5.x returns an output object whose pooler_output holds the merged tokens
    return out.pooler_output if hasattr(out, "pooler_output") else out


def distill_loss(student, processor, size, sharp, blurred, device, dtype):
    """Feature-distillation loss for one batch of pairs.

    Args:
        student: the encoder wrapped by add_lora().
        processor: the model's AutoProcessor.
        size: pixel limits, as in visual_tokens().
        sharp: list of PIL images from the in-focus camera.
        blurred: list of PIL images of the same scenes from the defocused camera.
        device: "cuda" or "cpu".
        dtype: dtype of the model's weights.

    Returns:
        Scalar tensor: mean squared difference between student and teacher tokens.
    """
    # teacher: adapter off, sharp images, no gradients
    with torch.no_grad(), student.disable_adapter():
        target = visual_tokens(student, processor, size, sharp, device, dtype)
    # student: adapter on, blurred images
    pred = visual_tokens(student, processor, size, blurred, device, dtype)
    # the two cameras give images of the same size, so the tokens correspond one to one
    if pred.shape != target.shape:
        raise ValueError(f"token shapes differ: blurred {tuple(pred.shape)}, sharp {tuple(target.shape)}")
    # compare in float32 so the loss is not rounded to bfloat16
    return F.mse_loss(pred.float(), target.float())
