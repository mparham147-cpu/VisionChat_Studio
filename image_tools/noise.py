"""Additional noise experiments implemented with the existing NumPy library.

All functions accept a uint8 gray/BGR image, leave it unchanged and return
another uint8 array of the same shape. Seeds make Live and Apply repeatable.
Salt/pepper applies to whole pixels; other distributions act per channel.
These are extensions: the supplied generator implements Gaussian noise only.
"""
import numpy as np


def noise_generator(image, seed):
    """Validate an 8-bit input/seed and create an isolated NumPy random stream.

    A local RandomState avoids changing global randomness used by other tasks.
    Seed must be an integer from 0 to 2**31-1, matching the UI's typed input.
    """
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('Noise requires an 8-bit grayscale or color image.')
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2147483647:
        raise ValueError('Seed must be an integer between 0 and 2147483647.')
    return np.random.RandomState(seed)


def apply_salt_pepper_noise(image, amount, salt_ratio, seed):
    """Replace a random fraction of whole pixels with white or black.

    amount: probability in [0,1] that a pixel is replaced; the actual count
    fluctuates. salt_ratio: share of replaced pixels assigned 255 (salt);
    the rest become 0 (pepper). At 0.5 both are equally likely. seed selects
    a repeatable pattern. This is useful for testing median filtering.
    """
    if not 0 <= amount <= 1 or not 0 <= salt_ratio <= 1:
        raise ValueError('Amount and salt ratio must be between 0 and 1.')
    generator = noise_generator(image, seed)
    pattern = generator.uniform(0, 1, image.shape[:2])
    result = image.copy()
    result[pattern < amount * salt_ratio] = 255
    result[(pattern >= amount * salt_ratio) & (pattern < amount)] = 0
    return result


def apply_salt_noise(image, amount, seed):
    """Replace pixels with white (255) with probability amount in [0,1].

    This is the salt-only case of impulse noise. It shares the seeded,
    non-mutating behavior of apply_salt_pepper_noise.
    """
    return apply_salt_pepper_noise(image, amount, 1, seed)


def apply_pepper_noise(image, amount, seed):
    """Replace pixels with black (0) with probability amount in [0,1].

    This is the pepper-only case of impulse noise. Use Median blur to
    investigate recovery; high amounts can destroy recoverable information.
    """
    return apply_salt_pepper_noise(image, amount, 0, seed)


def apply_uniform_noise(image, lower, upper, seed):
    """Add independent uniform offsets between lower and upper byte units.

    Values are real-number brightness offsets, not pixel probabilities.
    lower must not exceed upper. Negative values darken; positive values
    brighten. Float arithmetic prevents uint8 wraparound; clipping limits
    the result to 0–255 before conversion back to uint8.
    """
    if not -255 <= lower <= upper <= 255:
        raise ValueError('Uniform bounds must satisfy -255 <= lower <= upper <= 255.')
    generator = noise_generator(image, seed)
    offsets = generator.uniform(lower, upper, image.shape)
    return np.clip(image.astype(np.float32) + offsets, 0, 255).astype(np.uint8)


def apply_speckle_noise(image, sigma, seed):
    """Add multiplicative Gaussian noise: result = image + image * noise.

    noise has mean 0 and dimensionless standard deviation sigma in [0,2].
    Thus brighter pixels receive larger changes; black stays black. This
    simulates signal-dependent variation. Output is clipped to byte range.
    """
    if not 0 <= sigma <= 2:
        raise ValueError('Speckle sigma must be between 0 and 2.')
    generator = noise_generator(image, seed)
    values = image.astype(np.float32)
    variation = generator.normal(0, sigma, image.shape)
    return np.clip(values + values * variation, 0, 255).astype(np.uint8)


def apply_poisson_noise(image, peak_counts, seed):
    """Simulate photon-counting noise with peak_counts at full brightness.

    Each channel is normalized to [0,1], multiplied by peak_counts, sampled
    from a Poisson distribution, then scaled back to byte intensity. A larger
    peak_counts value (1–4096) generally means less relative noise. Black
    stays black because its expected photon count is zero. Output is clipped.
    """
    if not 1 <= peak_counts <= 4096:
        raise ValueError('Peak counts must be between 1 and 4096.')
    generator = noise_generator(image, seed)
    expected = image.astype(np.float32) / 255 * peak_counts
    values = generator.poisson(expected) / peak_counts * 255
    return np.clip(values, 0, 255).astype(np.uint8)
