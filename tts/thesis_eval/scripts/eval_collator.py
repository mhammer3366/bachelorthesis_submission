"""Eval-only speech collator: drop samples whose speech labels are entirely masked."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

IGNORE_ID = -100


@dataclass
class EvalSpeechDataCollator:
    """Same padding/masking as training collator, but skips all-masked samples."""

    t3_config: Any
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid = [f for f in features if f is not None]
        if not valid:
            return {}

        # Keep collator on CPU; model forward moves batch to device.
        valid = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in f.items()} for f in valid]

        # Pre-filter: keep only samples with at least one valid speech label position
        kept = []
        for f in valid:
            speech_tokens = f["speech_tokens"]
            speech_len = int(f["speech_token_lens"].item())
            prompt_len = self.t3_config.speech_cond_prompt_len
            # positions that would NOT be masked: t >= prompt_len and t < speech_len-1
            if speech_len - 1 <= prompt_len:
                continue
            kept.append(f)

        if not kept:
            return {}

        batch_size = len(kept)
        text_tokens_list = [f["text_tokens"] for f in kept]
        speech_tokens_list = [f["speech_tokens"] for f in kept]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        padded_text_tokens = torch.stack(
            [F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id) for t in text_tokens_list]
        )
        padded_speech_tokens = torch.stack(
            [F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id) for s in speech_tokens_list]
        )

        text_token_lens = torch.stack([f["text_token_lens"] for f in kept])
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in kept])
        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in kept])
        t3_cond_prompt_speech_tokens = torch.stack([f["t3_cond_prompt_speech_tokens"] for f in kept])
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in kept])
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        prompt_len = self.t3_config.speech_cond_prompt_len

        shifted_text = padded_text_tokens[:, 1:].contiguous()
        T_text = shifted_text.size(1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)
        dev = shifted_text.device
        arange_text = torch.arange(T_text, device=dev)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]
        labels_text = shifted_text.clone()
        labels_text[mask_pad_text] = IGNORE_ID

        shifted_speech = padded_speech_tokens[:, 1:].contiguous()
        T_speech = shifted_speech.size(1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)
        arange_speech = torch.arange(T_speech, device=dev)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]
        mask_prompt = arange_speech[None] < prompt_len
        mask_prompt = mask_prompt.expand(batch_size, T_speech)
        mask_speech_total = mask_pad_speech | mask_prompt
        labels_speech = shifted_speech.clone()
        labels_speech[mask_speech_total] = IGNORE_ID

        return {
            "text_tokens": padded_text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,
            "labels_speech": labels_speech,
        }
