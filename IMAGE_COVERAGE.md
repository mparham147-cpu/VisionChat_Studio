# Image operations and noise audit

## What the uploaded files contain

All uploaded Python files and notebook code cells were scanned. Session3 implements a Gaussian noise generator. Its median-filter comments mention salt-and-pepper impulses in Persian, but do not implement that generator. No uniform, speckle or Poisson image-noise generator was found. Uniform sampling in db_helpers creates sales data, not image noise.

Session2 covers upload/save, rescaling, grayscale/reduced grayscale, channels/alpha, min/max normalization, combining images, background subtraction, and a comment suggesting a difference threshold. Session3 covers resize, four filters, flip, rotation, sharpening and a random pipeline. Sessions4/5 cover canvases, shapes, text, markers/arrows, and video sources. Those image operations are present in the registry or separate Upload/Canvas/Inspect/Save tasks.

## Current noise menu

| Option | User parameters | Meaning / origin |
|---|---|---|
| Gaussian | Mean, sigma, seed | Additive normal noise; source implementation corrected to prevent uint8 wraparound |
| Salt | Amount, seed | White impulse pixels; extension of the mentioned impulse-noise idea |
| Pepper | Amount, seed | Black impulse pixels; extension |
| Salt and pepper | Amount, salt ratio, seed | Combined white/black impulses; extension |
| Uniform | Lower/upper offsets, seed | Bounded equally likely additive noise; NumPy extension |
| Speckle | Sigma, seed | Multiplicative signal-dependent Gaussian noise; NumPy extension |
| Poisson | Peak counts, seed | Photon-counting simulation; NumPy extension |

These seven options are not a claim to implement every noise distribution in mathematics. Every supplied image-noise implementation is covered; common additions are identified above. All noise outputs are clipped to byte range, source arrays stay unchanged, and a fixed seed repeats the pattern.

## User-controlled filters

- Mean/Gaussian/Median kernels now offer odd values 1–31. Gaussian exposes both sigmas; Bilateral exposes diameter and both sigmas.
- Custom convolution uses the source's cv2.filter2D with your comma/semicolon-separated matrix. Odd dimensions 1–31; optional coefficient normalization and output offset. Invalid/ragged/nonfinite kernels or normalization of a zero-sum kernel show an error.
- The pipeline now exposes seven step probabilities and each step's kernel, sigma, strength, direction, rotation/scale/canvas setting and noise mean/spread. Probability 0 disables a step; 1 always applies it. Steps retain the lesson order.
- Binary threshold exposes a grayscale cutoff. Background mask exposes the source comment's normalized difference threshold, with 0.4 as the default.
- Highlight offers true opacity blending and the lesson's normalized additive mode.

## UI and tests

Menu descriptions no longer print under buttons. Hover tooltips retain explanations, and numeric ranges/current values remain visible. Dedicated Help remains a readable reference. Live and Apply recalculate from the fixed current edit snapshot; selecting another operation chains the latest unsaved result.

There are 36 registered effects. All defaults were exercised by the registry test; 18 total tests pass. Noise tests cover reproducibility, unchanged source, grayscale shape, impulse extremes, zero-strength identity, clipping and invalid inputs. Custom kernels are checked for identity/error behavior. Pipeline probability 0/1 and user-specified Gaussian mean were checked. UI tests verify hover-only menu descriptions and invalid-kernel error panels.
