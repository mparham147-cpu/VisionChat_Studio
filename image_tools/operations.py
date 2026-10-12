"""
image_tools/operations.py
=========================

Every pure image function the studio can apply. A "pure" function:

* takes one image plus (optionally) some numeric settings,
* returns a new image,
* never modifies the input.

No Streamlit, no session state, no file I/O. That means these functions
can be unit-tested or reused anywhere.

Each function is deliberately written out so the code that runs is
exactly the code you read — no aliases, no dict lookups, no lambdas.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color and value operations
# ---------------------------------------------------------------------------

def convert_to_grayscale(image):
    """Convert an 8-bit BGR image to one-channel grayscale using OpenCV.

    Brightness combines blue/green/red values; output keeps width/height but
    loses color. The input array is not modified. Useful before intensity analysis."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_to_reduced_grayscale(image):
    """Convert BGR to gray and reduce width/height to one eighth.

    Each dimension stays at least one pixel. Returns a new one-channel array,
    useful for lightweight previews. OpenCV uses its default resize interpolation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return cv2.resize(gray, (max(1, width // 8), max(1, height // 8)))


def extract_channel(image, channel_index):
    """
    Return one channel of a multi-channel image as grayscale.

    channel_index: 0 = Blue, 1 = Green, 2 = Red, 3 = Alpha.
    """
    if image.ndim != 3 or channel_index >= image.shape[2]:
        raise ValueError("This image does not have that channel. Upload a PNG for alpha.")
    return cv2.split(image)[channel_index]


def rescale_to_unit_range(image):
    """Return a float32 image divided by 255, preserving shape and channels.

    For byte input, values become 0–1. This changes representation rather than
    stretching contrast. The app converts it to byte form before subsequent filters."""
    return image.astype(np.float32) / 255.0


def normalize_to_unit_range(image):
    """Stretch the global minimum/maximum of all channels to 0–1.

    Returns float32 of the same shape. Constant images become zeros to avoid
    division by zero. This is the min/max normalization in Session2."""
    data = image.astype(np.float32)
    span = float(data.max() - data.min())
    if span == 0:
        return np.zeros_like(data)
    return (data - data.min()) / span


def denormalize_to_bytes(image):
    """Normalize then scale back to 0–255 bytes. Useful for contrast demos."""
    return (normalize_to_unit_range(image) * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def resize_image(image, width, height):
    """Resize to integer pixel width/height using OpenCV default interpolation.

    The UI accepts 1–4096 for each dimension. Unequal scale ratios stretch
    proportions. Returns a new array; source pixels remain unchanged."""
    return cv2.resize(image, (width, height))


def flip_image(image, direction):
    """
    Mirror the image.

    direction: "Horizontal", "Vertical" or "Both".
    """
    flip_code = {"Horizontal": 1, "Vertical": 0, "Both": -1}[direction]
    return cv2.flip(image, flip_code)


def rotate_image(image, angle_degrees, scale, expand_canvas):
    """
    Rotate around the center by ``angle_degrees``.

    When ``expand_canvas`` is True, the output is enlarged so the
    rotated corners fit inside it.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, scale)

    if expand_canvas:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int(np.ceil(height * sin + width * cos))
        new_height = int(np.ceil(height * cos + width * sin))
        matrix[0, 2] += (new_width - width) / 2
        matrix[1, 2] += (new_height - height) / 2
        width, height = new_width, new_height

    if width * height > 25_000_000:
        raise ValueError("Result is too large. Reduce scale or input dimensions.")
    return cv2.warpAffine(image, matrix, (width, height))


# ---------------------------------------------------------------------------
# Blur (four kinds)
# ---------------------------------------------------------------------------

def apply_mean_blur(image, kernel_size):
    """Smooth with an equal-weight square neighborhood of kernel_size pixels.

    The UI offers odd sizes 1–31; size 1 leaves values unchanged. Larger kernels
    remove more fine detail. Returns the same shape/dtype without changing input."""
    return cv2.blur(image, (kernel_size, kernel_size))


def apply_gaussian_blur(image, kernel_size, sigma_x, sigma_y):
    """Smooth with a square Gaussian kernel and horizontal/vertical spread.

    kernel_size is odd (1–31 in the UI); sigma_x/sigma_y are 0–100. Zero sigma_x
    derives spread from the kernel, and zero sigma_y uses sigma_x. Unlike mean
    blur, nearby pixels have larger weights. Source is unchanged."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)


def apply_median_blur(image, kernel_size):
    """Replace each channel value with the median of its square neighborhood.

    kernel_size is odd, 1–31 for the uint8 image workflow. Median filtering
    reduces isolated salt/pepper impulses and may remove small features."""
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_blur(image, diameter, sigma_color, sigma_space):
    """Bilateral smoothing using diameter, sigma_color and sigma_space.

    diameter sets neighborhood width in pixels (1–31). sigma_color controls
    tolerated intensity differences; sigma_space controls spatial range (0–255).
    Larger settings smooth more and can cost more time. Input is unchanged."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Detail and noise
# ---------------------------------------------------------------------------

def apply_sharpen(image, strength):
    """
    Emphasize edges with a 3×3 kernel.

    strength = 0 leaves the image unchanged; higher values sharpen more.
    """
    kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + 4 * strength, -strength],
        [0, -strength, 0],
    ])
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_noise(image, sigma, seed, mean=0):
    """
    Add Gaussian intensity offsets with user-controlled mean and sigma.

    mean is the average brightness offset (-255 to 255 byte units); sigma
    is its spread (0 to 100). Values are clipped before conversion to uint8,
    avoiding the unsigned wraparound in the supplied demonstration.

    ``seed`` makes the noise reproducible: the same seed always gives
    the same result, which is why the preview is stable.
    """
    from image_tools.noise import noise_generator
    if not 0 <= sigma <= 100 or not -255 <= mean <= 255:
        raise ValueError('Gaussian sigma must be 0–100 and mean -255–255.')
    rng = noise_generator(image, seed)
    noisy = image.astype(np.float32) + rng.normal(mean, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_random_pipeline(image, seed, mean_chance=0.5, mean_kernel=5,
                          gaussian_chance=0.5, gaussian_kernel=5, sigma_x=0, sigma_y=0,
                          sharpen_chance=0.3, strength=1.0, median_chance=0.5,
                          median_kernel=3, flip_chance=0.5, direction="Horizontal",
                          rotate_chance=0.7, angle_degrees=45, scale=1.0,
                          expand_canvas=False, noise_chance=0.2, noise_sigma=10,
                          noise_mean=0):
    """Apply the lesson pipeline in order, with user-set chances/settings.

    Order: mean blur, Gaussian blur, sharpen, median blur, flip, rotate,
    Gaussian noise. Each chance is a probability in [0,1]: 0 disables the
    step, 1 always applies it. Other arguments match the individual effects.
    Seed controls step selection and noise so preview/apply remain repeatable.
    The function returns (new image, names of steps actually applied), leaving
    the source unchanged. Unlike the original demo, parameters are not hidden
    random choices; users set each kernel, sigma, direction and angle explicitly.
    """
    import random
    rng = random.Random(seed)
    chances = [mean_chance, gaussian_chance, sharpen_chance, median_chance,
               flip_chance, rotate_chance, noise_chance]
    if any(not 0 <= chance <= 1 for chance in chances):
        raise ValueError('Every pipeline chance must be between 0 and 1.')
    result, applied = image.copy(), []
    if rng.random() < mean_chance:
        result = apply_mean_blur(result, mean_kernel); applied.append('Mean blur')
    if rng.random() < gaussian_chance:
        result = apply_gaussian_blur(result, gaussian_kernel, sigma_x, sigma_y); applied.append('Gaussian blur')
    if rng.random() < sharpen_chance:
        result = apply_sharpen(result, strength); applied.append('Sharpen')
    if rng.random() < median_chance:
        result = apply_median_blur(result, median_kernel); applied.append('Median blur')
    if rng.random() < flip_chance:
        result = flip_image(result, direction); applied.append('Flip')
    if rng.random() < rotate_chance:
        result = rotate_image(result, angle_degrees, scale, expand_canvas); applied.append('Rotate')
    if rng.random() < noise_chance:
        result = apply_gaussian_noise(result, noise_sigma, seed, noise_mean); applied.append('Gaussian noise')
    return result, applied


def apply_custom_convolution(image, kernel, normalize_kernel, delta):
    """Run the user's numeric kernel through the lesson's cv2.filter2D.

    kernel is a finite odd 2D matrix. normalize_kernel divides coefficients
    by their sum (a zero-sum kernel cannot be normalized). delta adds a byte
    brightness offset after filtering. Output keeps the input shape and dtype;
    negative or oversized intensities saturate in uint8. Source is unchanged.
    """
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 2 or not np.isfinite(kernel).all() or any(s < 1 or s > 31 or s % 2 == 0 for s in kernel.shape):
        raise ValueError('Use a finite odd kernel, from 1×1 to 31×31.')
    if normalize_kernel:
        total = float(kernel.sum())
        if abs(total) < 1e-8:
            raise ValueError('Zero-sum kernels cannot be normalized; turn normalization off.')
        kernel = kernel / total
    return cv2.filter2D(image, -1, kernel, delta=delta)


def apply_binary_threshold(image, threshold):
    """Return a grayscale byte mask: intensity above threshold becomes white.

    threshold is 0–255. Color inputs first use the lesson's BGR-to-gray
    conversion; comparison uses NumPy. This implements the threshold idea
    mentioned in Session2, with a visible user-controlled cutoff.
    """
    if not 0 <= threshold <= 255:
        raise ValueError('Threshold must be between 0 and 255.')
    gray = convert_to_grayscale(image) if image.ndim == 3 else image
    return (gray > threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Combine two images
# ---------------------------------------------------------------------------

def blend_two_images(image, second_image, second_weight):
    """
    Add two normalized images and re-normalize the result.

    ``second_weight`` scales the second image before the addition.
    """
    if image.shape != second_image.shape:
        raise ValueError("Both images must have the same width and height.")
    first_normalized = normalize_to_unit_range(image)
    second_normalized = normalize_to_unit_range(second_image)
    combined = first_normalized + second_weight * second_normalized
    return normalize_to_unit_range(combined)


def subtract_background(image, background_image):
    """
    Subtract a normalized background and re-normalize the difference.

    Useful for a simple motion-detection demo.
    """
    if image.shape != background_image.shape:
        raise ValueError("Background and input must have identical dimensions.")
    return normalize_to_unit_range(
        normalize_to_unit_range(image) - normalize_to_unit_range(background_image)
    )


def create_background_mask(image, background_image, threshold):
    """Threshold the lesson's normalized background difference into a mask.

    Both images must have identical shapes. threshold is a normalized cutoff
    from 0 to 1 (the source comments suggest 0.4). A pixel becomes white when
    any channel's normalized difference exceeds the cutoff. This follows the
    signed subtraction/min-max normalization lesson, not absolute difference.
    Constant differences normalize to zero. Source images are unchanged.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('Difference threshold must be between 0 and 1.')
    difference = subtract_background(image, background_image)
    if difference.ndim == 3:
        difference = difference.max(axis=2)
    return (difference > threshold).astype(np.uint8) * 255
