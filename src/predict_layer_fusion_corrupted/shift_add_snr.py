
from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    calculate_rms,
)
from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy as np
import random
import warnings

class ShiftAddSNR(BaseWaveformTransform):
    """
    Shift the samples forwards or backwards, with or without rollover
    + Add to original signal with given SNR.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        min_fraction: float = -0.5,
        max_fraction: float = 0.5,
        rollover: bool = True,
        fade: bool = False,
        fade_duration: float = 0.01,
        p: float = 0.5,
    ):
        """
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB
        :param min_fraction: Minimum fraction of total sound length to shift
        :param max_fraction: Maximum fraction of total sound length to shift
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param fade: When set to True, there will be a short fade in and/or out at the "stitch"
            (that was the start or the end of the audio before the shift). This can smooth out an
            unwanted abrupt change between two consecutive samples (which sounds like a
            transient/click/pop).
        :param fade_duration: If `fade=True`, then this is the duration of the fade in seconds.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_snr_in_db <= max_snr_in_db
        assert min_fraction >= -1
        assert max_fraction <= 1
        assert type(fade_duration) in [int, float] or not fade
        assert fade_duration > 0 or not fade
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover
        self.fade = fade
        self.fade_duration = fade_duration

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["fraction_to_shift"] = random.uniform(
                self.min_fraction, self.max_fraction
            )
            self.parameters["snr_in_db"] = random.uniform(
                self.min_snr_in_db, self.max_snr_in_db
            )

    def apply(self, samples: np.ndarray, sample_rate: int):
        noise_sound = samples.copy()

        _rms = calculate_rms(samples)
        if _rms < 1e-9:
            warnings.warn(
                "The file is too silent to be added as noise. Returning the input"
                " unchanged."
            )
            return samples

        noise_rms = _rms
        clean_rms = _rms

        desired_noise_rms = calculate_desired_noise_rms(
            clean_rms, self.parameters["snr_in_db"]
        )

        # Adjust the noise to match the desired noise RMS
        noise_sound = noise_sound * (desired_noise_rms / noise_rms)

        # Shift the noise.
        num_places_to_shift = round(
            self.parameters["fraction_to_shift"] * noise_sound.shape[-1]
        )
        shifted_samples = np.roll(noise_sound, num_places_to_shift, axis=-1)

        if not self.rollover:
            if num_places_to_shift > 0:
                shifted_samples[..., :num_places_to_shift] = 0.0
            elif num_places_to_shift < 0:
                shifted_samples[..., num_places_to_shift:] = 0.0

        if self.fade:
            fade_length = int(sample_rate * self.fade_duration)

            fade_in = np.linspace(0, 1, num=fade_length)
            fade_out = np.linspace(1, 0, num=fade_length)

            if num_places_to_shift > 0:

                fade_in_start = num_places_to_shift
                fade_in_end = min(
                    num_places_to_shift + fade_length, shifted_samples.shape[-1]
                )
                fade_in_length = fade_in_end - fade_in_start

                shifted_samples[
                    ...,
                    fade_in_start:fade_in_end,
                ] *= fade_in[:fade_in_length]

                if self.rollover:

                    fade_out_start = max(num_places_to_shift - fade_length, 0)
                    fade_out_end = num_places_to_shift
                    fade_out_length = fade_out_end - fade_out_start

                    shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                        -fade_out_length:
                    ]

            elif num_places_to_shift < 0:

                positive_num_places_to_shift = (
                    shifted_samples.shape[-1] + num_places_to_shift
                )

                fade_out_start = max(positive_num_places_to_shift - fade_length, 0)
                fade_out_end = positive_num_places_to_shift
                fade_out_length = fade_out_end - fade_out_start

                shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                    -fade_out_length:
                ]

                if self.rollover:
                    fade_in_start = positive_num_places_to_shift
                    fade_in_end = min(
                        positive_num_places_to_shift + fade_length,
                        shifted_samples.shape[-1],
                    )
                    fade_in_length = fade_in_end - fade_in_start
                    shifted_samples[
                        ...,
                        fade_in_start:fade_in_end,
                    ] *= fade_in[:fade_in_length]

        return samples + shifted_samples
