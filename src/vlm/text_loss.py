"""Text loss: how unlikely the VLM finds the correct transcription of an image.

Used by method (d) (the image comes from the trained filter, gradients reach the filter) and by
method (c) with the text loss (the image is the blurred photograph, gradients reach the LoRA adapter
of the vision encoder). For an image and its correct text t, the loss is the cross-entropy of t under
the VLM with
teacher forcing (the model is shown the prompt, the image and the correct first k tokens, and is
scored on token k + 1):

    L = -(1 / T) * sum_k log p(t_k | t_<k, prompt, image)

Unlike the pixel and feature losses, it is measured where the evaluation is measured (on the text the
model reads), and it does not prefer the average of several possible sharp images: a blurry image
makes the model unsure, which raises the loss. In bits, the teacher-forced token error rate is at most
this loss, since a wrongly predicted token has probability at most 1/2.
"""
import torch
import torch.nn.functional as F
from PIL import Image

from src.vlm.vlm_input import pil_size, pixel_values


class TextLoss:
    """Builds the token sequences once and scores batches of images against their texts."""

    def __init__(self, reader, full_shape, long_side, cap=None):
        """Prepare the prompt tokens for images of one size.

        Args:
            reader: QwenReader, whose prompt, processor and model are used (the model as it is at
                each call, e.g. with a LoRA adapter being trained).
            full_shape: (height, width) of the full-resolution photographs.
            long_side: the config's image.long_side (size after to_pil()).
            cap: largest cross-entropy (nats per token) one image may contribute, or None for no
                limit. Photographs no filter can make readable would otherwise dominate the gradient.
        """
        self.reader = reader
        self.cap = cap
        tok = reader.proc.tokenizer
        # the prompt with the image placeholder expanded to the right number of image tokens: a blank
        # image of the size the VLM sees gives exactly the token layout of the real images
        h, w = pil_size(*full_shape, long_side)
        blank = Image.new("RGB", (w, h))
        inp = reader.proc(text=[reader.text], images=[blank], size=reader.size, return_tensors="pt")
        self.prompt_ids = inp["input_ids"][0]
        # which prompt tokens are image tokens (needed by the model's multimodal position encoding)
        self.prompt_types = inp["mm_token_type_ids"][0]
        # the token that ends the model's answer in the chat format
        self.end = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = tok.pad_token_id

    def tokens(self, texts):
        """Token sequences and labels for a batch of correct transcriptions.

        Args:
            texts: list of strings.

        Returns:
            (input_ids, attention_mask, mm_token_type_ids, labels): long tensors (B, L) on the
            reader's device; labels are -100 on the prompt and the padding, so only the answer is
            scored; mm_token_type_ids marks the image tokens as the processor does.
        """
        tok = self.reader.proc.tokenizer
        seqs, labs = [], []
        for t in texts:
            # the answer as the model would write it, followed by the end token
            ans = torch.tensor(tok(t, add_special_tokens=False)["input_ids"] + [self.end])
            seqs.append(torch.cat([self.prompt_ids, ans]))
            labs.append(torch.cat([torch.full_like(self.prompt_ids, -100), ans]))
        # pad on the right to the longest sequence of the batch; answer and padding are text tokens
        n, p = max(len(s) for s in seqs), len(self.prompt_ids)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        types = torch.zeros((len(seqs), n), dtype=self.prompt_types.dtype)
        labels = torch.full((len(seqs), n), -100, dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            ids[i, :len(s)], mask[i, :len(s)], labels[i, :len(l)] = s, 1, l
            types[i, :p] = self.prompt_types
        dev = self.reader.device
        return ids.to(dev), mask.to(dev), types.to(dev), labels.to(dev)

    def score(self, pix, grid, texts):
        """Cross-entropy of each correct transcription given images already in the processor's layout.

        Args:
            pix: pixel values (B * patches, 3 * 2 * 14 * 14), as returned by the processor or by
                vlm_input.pixel_values().
            grid: grid_thw (B, 3) of the same images.
            texts: list of B strings.

        Returns:
            Tensor (B,): mean cross-entropy per answer token, in nats, uncapped.
        """
        r = self.reader
        ids, mask, types, labels = self.tokens(texts)
        out = r.model(input_ids=ids, attention_mask=mask, mm_token_type_ids=types,
                      pixel_values=pix.to(r.device, dtype=r.model.dtype), image_grid_thw=grid.to(r.device))
        # position k predicts token k + 1; score only the answer tokens, in float32
        logits = out.logits[:, :-1].float()
        target = labels[:, 1:]
        ce = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=-100, reduction="none")
        n = (target != -100).sum(1).clamp(min=1)
        return ce.sum(1) / n

    def per_image(self, restored, texts):
        """Cross-entropy of each correct transcription given its restored image (method (d)).

        Args:
            restored: tensor (B, h, w) in [0, 1], the filter's output after to_vlm_size().
            texts: list of B strings.

        Returns:
            Tensor (B,): mean cross-entropy per answer token, in nats, uncapped; differentiable with
            respect to `restored`.
        """
        # the differentiable image input, in the processor's layout
        pix, grid = pixel_values(restored, self.reader.proc.image_processor, self.reader.size)
        return self.score(pix, grid, texts)

    def per_image_pil(self, images, texts):
        """Cross-entropy of each correct transcription given its PIL image (method (c)).

        The images go through the real processor, exactly as when the model reads them.

        Args:
            images: list of B PIL images from to_pil().
            texts: list of B strings.

        Returns:
            Tensor (B,): mean cross-entropy per answer token, in nats, uncapped.
        """
        inp = self.reader.proc.image_processor(images=images, size=self.reader.size, return_tensors="pt")
        return self.score(inp["pixel_values"], inp["image_grid_thw"], texts)

    def capped_mean(self, ce):
        """Training loss of a batch from its per-image cross-entropies.

        Args:
            ce: tensor (B,) from per_image() or per_image_pil().

        Returns:
            Scalar tensor: mean over images, each capped at `cap` if one is set.
        """
        # images above the cap contribute a constant (no gradient), so hopeless ones do not dominate
        if self.cap is not None:
            ce = torch.clamp(ce, max=self.cap)
        return ce.mean()

    def __call__(self, restored, texts):
        """Training loss of a batch of restored images (method (d)).

        Args:
            restored: tensor (B, h, w) in [0, 1].
            texts: list of B strings.

        Returns:
            Scalar tensor.
        """
        return self.capped_mean(self.per_image(restored, texts))
