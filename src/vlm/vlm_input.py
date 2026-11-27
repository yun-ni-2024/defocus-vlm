"""Differentiable version of the VLM's input formatting, so gradients reach the filter in front of it.

In the evaluation pipeline a float image goes through
    to_pil():   clip to [0, 1], 8-bit, grey to RGB, shrink to a long side of 1024 (PIL Lanczos)
    processor:  resize to a multiple of 28 within the pixel limits (torchvision bicubic, antialiased),
                scale to [0, 1], normalise with the CLIP mean and std, cut into 14 x 14 patches
None of this passes gradients. The functions below do the same steps on torch tensors, without the
8-bit rounding and with an antialiased bicubic resize in place of Lanczos for the first shrink.
src/frontend/checks.py measures how close the result is to the real processor.
"""
import torch
import torch.nn.functional as F
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def pil_size(h, w, long_side):
    """Size of the image after to_pil(): the long side reduced to long_side, never enlarged.

    Args:
        h: height of the full-resolution image.
        w: width of the full-resolution image.
        long_side: target long side, or None to keep the full size.

    Returns:
        (height, width) in pixels, rounded as PIL's resize call in to_pil() rounds them.
    """
    if long_side is None or max(h, w) <= long_side:
        return h, w
    s = long_side / max(h, w)
    return round(h * s), round(w * s)


def to_vlm_size(x, long_side):
    """Clip to [0, 1] and shrink to the size to_pil() gives, keeping gradients.

    Args:
        x: tensor (B, H, W), float, paper about 1.
        long_side: the config's image.long_side.

    Returns:
        Tensor (B, h, w) in [0, 1].
    """
    h, w = pil_size(x.shape[-2], x.shape[-1], long_side)
    x = x.clamp(0, 1)
    if (h, w) == tuple(x.shape[-2:]):
        return x
    return F.interpolate(x[:, None], size=(h, w), mode="bicubic", antialias=True, align_corners=False)[:, 0]


def pixel_values(x, image_processor, size):
    """Turn shrunk grey images into the encoder's input, exactly as the processor lays it out.

    Args:
        x: tensor (B, h, w) in [0, 1], all images of the same size (output of to_vlm_size()).
        image_processor: the processor's image part (QwenReader.proc.image_processor), for the
            mean, std and patch settings.
        size: pixel limits {"shortest_edge", "longest_edge"} (QwenReader.size).

    Returns:
        (pixel_values, grid_thw): float32 tensor (B * grid_h * grid_w, 3 * 2 * 14 * 14) and long
        tensor (B, 3), as returned by the processor.
    """
    ip = image_processor
    p, m, t = ip.patch_size, ip.merge_size, ip.temporal_patch_size
    b, h, w = x.shape
    # the processor's target size: multiples of 28 within the pixel limits
    rh, rw = smart_resize(h, w, factor=p * m, min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"])
    # grey to RGB, then the processor's antialiased bicubic resize
    x = x[:, None].expand(b, 3, h, w)
    x = F.interpolate(x, size=(rh, rw), mode="bicubic", antialias=True, align_corners=False)
    # normalise with the CLIP mean and std (the processor scales 0..255 to 0..1 first; x is already 0..1)
    mean = torch.tensor(ip.image_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor(ip.image_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    x = (x - mean) / std
    # cut into patches with the same reshape and permutation as Qwen2VLImageProcessor.patchify()
    gh, gw = rh // p, rw // p
    x = x.reshape(b, 3, gh // m, m, p, gw // m, m, p).permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t, -1, -1).reshape(b * gh * gw, 3 * t * p * p)
    grid = torch.tensor([[1, gh, gw]] * b, dtype=torch.long, device=x.device)
    return x, grid


def visual_tokens(encoder, pixels, grid, dtype):
    """Run the vision encoder (with its token merger) and return the tokens the language model reads.

    Args:
        encoder: the vision encoder (QwenReader.model.model.visual).
        pixels: pixel values from pixel_values().
        grid: grid_thw from pixel_values().
        dtype: dtype of the encoder's weights; the pixels are cast to it.

    Returns:
        Tensor (total number of tokens, hidden size).
    """
    out = encoder(pixels.to(dtype), grid_thw=grid)
    # transformers 5.x returns an output object whose pooler_output holds the merged tokens
    return out.pooler_output if hasattr(out, "pooler_output") else out
