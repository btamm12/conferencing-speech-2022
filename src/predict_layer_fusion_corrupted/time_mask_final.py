import random

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class TimeMaskFinal(BaseWaveformTransform):
    """
    Mask the audio periodically with jitter.
    """

    supports_multichannel = True

    def __init__(
        self,
        mask_ms: float = 200,
        unmask_ms: float = 800,
        jitter_ms: float = 200,
        p: float = 0.5,
    ):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Must be between 0.0 and 1.0
        :param fade: When set to True, add a linear fade in and fade out of the silent
            part. This can smooth out an unwanted abrupt change between two consecutive
            samples (which sounds like a transient/click/pop).
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if mask_ms < 0.0:
            raise ValueError("mask_ms must be non-negative")
        if unmask_ms < 0.0:
            raise ValueError("unmask_ms must be non-negative")
        if jitter_ms < 0.0:
            raise ValueError("jitter_ms must be non-negative")
        self.mask_ms = mask_ms
        self.unmask_ms = unmask_ms
        self.jitter_ms = jitter_ms

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            period_s = (self.mask_ms + self.unmask_ms) / 1000.0
            period_samples = int(sample_rate * period_s)
            offset = random.randint(0, period_samples)
            half_jitter_samples = int(sample_rate * (self.jitter_ms / 1000.0) / 2.0)
            half_mask_samples = int(sample_rate * (self.mask_ms / 1000.0) / 2.0)
            self.parameters["mask_len"] = 2 * half_mask_samples
            if self.parameters["mask_len"] == 0:
                return

            masks = []
            ptr = offset
            while ptr < num_samples:
                jitter = random.randint(-half_jitter_samples, half_jitter_samples)
                center_idx = ptr + jitter
                start_idx = max(center_idx - half_mask_samples, 0)
                end_idx = min(center_idx + half_mask_samples, num_samples)
                mask_audio_overlap = end_idx >= 0 and start_idx < num_samples
                if mask_audio_overlap:
                    masks.append((start_idx, end_idx))
                ptr += period_samples
            self.parameters["masks"] = masks

    def apply(self, samples: np.ndarray, sample_rate: int):
        new_samples = samples.copy()
        mask_len = self.parameters["mask_len"]
        if mask_len == 0:
            return new_samples
        masks = self.parameters["masks"]
        for mask_start, mask_end in masks:
            new_samples[..., mask_start:mask_end] = 0
        return new_samples
