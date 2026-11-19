"""Frozen Qwen2-VL-2B as a text reader."""
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

DEFAULT_PROMPT = ("Transcribe the text in this image exactly as written. "
                  "Output only the text, one line per line of text.")


class QwenReader:
    """Wraps Qwen2-VL so that a list of PIL images goes in and a list of transcriptions comes out.

    The image is resized by the processor so that its pixel count lies between min_pixels and
    max_pixels (Qwen2-VL cuts the image into 28 x 28 patches, one visual token per patch).
    """

    def __init__(self, path, dtype="bfloat16", min_pixels=64 * 28 * 28, max_pixels=448 * 28 * 28,
                 max_new_tokens=48, prompt=DEFAULT_PROMPT, device="cuda"):
        """Load the processor and the model.

        Args:
            path: folder with the downloaded weights.
            dtype: "bfloat16" or "float32".
            min_pixels: smallest image area after resizing, in pixels.
            max_pixels: largest image area after resizing, in pixels.
            max_new_tokens: longest transcription the model may write.
            prompt: instruction given with every image.
            device: "cuda" or "cpu".
        """
        self.proc = AutoProcessor.from_pretrained(path)
        # transformers 5.x takes the pixel limits as size={"shortest_edge", "longest_edge"}. They
        # only take effect when passed with every processor call (read()); given to
        # from_pretrained they are silently ignored and the image keeps its full size.
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        # batched generation needs the padding on the left, so every prompt ends at the same place
        self.proc.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            path, dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        # the chat-formatted prompt is the same for every image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        self.text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def read(self, images):
        """Transcribe a batch of images with greedy decoding.

        Args:
            images: list of PIL images.

        Returns:
            List of strings, one per image, stripped of surrounding whitespace.
        """
        # tokenise the prompt once per image and turn the images into patches
        inp = self.proc(text=[self.text] * len(images), images=images, size=self.size, padding=True,
                        return_tensors="pt").to(self.device)
        out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        # keep only the newly generated tokens
        new = out[:, inp["input_ids"].shape[1]:]
        return [t.strip() for t in self.proc.batch_decode(new, skip_special_tokens=True)]
