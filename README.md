# Fourier Epicycle Animation

`epicycles.py` generates Fourier epicycle animations from either:

- a built-in example shape
- a contour extracted from an image, optionally stitched across multiple internal components

The script samples a 2D curve, computes its discrete Fourier transform, and animates the reconstruction using rotating circles ("epicycles"). You can preview the result interactively or save it as a `.gif` or `.mp4`.

## Requirements

- Python 3.9+ is recommended
- Python packages:
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  - `Pillow`
- `ffmpeg` must be installed and available on your `PATH` if you want to save `.mp4` files

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib opencv-python Pillow
```

If you plan to save MP4 videos, also install `ffmpeg` with your system package manager.

Examples:

```bash
# macOS with Homebrew
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Quick Start

Run one of the built-in shapes:

```bash
python3 epicycles.py --example heart
```

Animate an image contour and show the epicycle arms:

```bash
python3 epicycles.py --image path/to/logo.png --arms
```

Stitch multiple image contours into one path to capture internal details:

```bash
python3 epicycles.py --image path/to/logo.png --stitch-contours --arms
```

Save a GIF:

```bash
python3 epicycles.py --example star --terms 80 --output star.gif
```

Save an MP4:

```bash
python3 epicycles.py --image path/to/drawing.png --terms 120 --output drawing.mp4
```

## Command-Line Usage

One of `--image` or `--example` is required. They are mutually exclusive.

```bash
python3 epicycles.py (--image PATH | --example NAME) [options]
```

### Input Source

#### `--example NAME`

Use one of the built-in parametric shapes:

- `heart`
- `star`
- `lissajous`
- `trefoil`

Example:

```bash
python3 epicycles.py --example trefoil
```

#### `--image PATH`

Load an image and extract a contour from it. The script:

1. reads the image with OpenCV
2. extracts a binary mask from transparency or image intensity
3. finds the main outer contour by default
4. optionally collects multiple disconnected internal components with `--stitch-contours`
5. resamples the resulting path to the requested number of points
6. centers and normalizes it before animation

For best results, use a simple high-contrast image with clear dark linework or a transparent background.

Example:

```bash
python3 epicycles.py --image assets/shape.png
```

## Options and Defaults

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--image PATH` | path | none | Path to an input image. Mutually exclusive with `--example`. |
| `--example NAME` | choice | none | Built-in example shape: `heart`, `star`, `lissajous`, or `trefoil`. Mutually exclusive with `--image`. |
| `--terms N` | integer | `50` | Number of Fourier terms / epicycles used for reconstruction. Larger values usually improve detail but add visual complexity. |
| `--points N` | integer | `1000` | Number of sample points used to represent the source curve before taking the DFT. Higher values preserve more detail but increase computation time. |
| `--stitch-contours` | flag | off | For image input, stitch multiple detected contour components into one continuous path so internal details are included. This adds bridge segments between components. |
| `--arms` | flag | off | Draw the spinning arm segments and circles. If omitted, only the traced path and faint target outline are shown. |
| `--output VALUE` | string | `show` | Output mode. Use `show` for an interactive window, a filename ending in `.gif` for GIF export, or any other filename for MP4 export. |
| `--fps N` | integer | `30` | Frames per second for the animation. Applies to both display timing and saved output. |
| `--duration N` | integer | `8` | Total animation length in seconds. Total frames = `fps * duration`. |
| `--build-duration N` | float | `3.0` | Seconds spent progressively adding Fourier terms before the final full-term trace begins. Use `0` to disable the buildup phase. |

## Output Behavior

### Interactive Preview

The default is:

```bash
--output show
```

This opens a Matplotlib window and plays the animation interactively.

Example:

```bash
python3 epicycles.py --example heart --output show
```

### GIF Export

If `--output` ends with `.gif`, the script uses Matplotlib's `PillowWriter`.

Example:

```bash
python3 epicycles.py --example star --output star.gif
```

### MP4 Export

Any `--output` value other than `show` or a filename ending in `.gif` is treated as MP4 output and uses Matplotlib's `FFMpegWriter`.

Example:

```bash
python3 epicycles.py --example lissajous --output lissajous.mp4
```

Important: the current script does not validate the file extension for MP4 output. If you pass something like `--output result.mov`, it will still try to encode using the MP4 writer path. In practice, use a `.mp4` filename.

## Common Examples

Preview a heart with defaults:

```bash
python3 epicycles.py --example heart
```

Use more epicycles for a smoother reconstruction:

```bash
python3 epicycles.py --example heart --terms 120
```

Increase contour resolution for an image input:

```bash
python3 epicycles.py --image path/to/input.png --points 2000
```

Capture inner details by stitching multiple image components into one path:

```bash
python3 epicycles.py --image path/to/input.png --stitch-contours --points 2000
```

Show the epicycle arms and save a GIF:

```bash
python3 epicycles.py --example star --arms --output star.gif
```

Create a longer, higher-frame-rate MP4:

```bash
python3 epicycles.py --example trefoil --fps 60 --duration 12 --output trefoil.mp4
```

Spend more time on the term-by-term buildup before tracing:

```bash
python3 epicycles.py --example heart --terms 80 --arms --build-duration 5
```

## How the Main Settings Affect the Result

### `--terms`

- Lower values produce a rougher approximation with fewer circles
- Higher values capture more detail
- The actual number used is capped to the number of available Fourier coefficients

Because the script samples `--points` points, the maximum practical number of terms is also bounded by that sample count.

### `--points`

- Higher values preserve more source detail
- Higher values also increase DFT work and can make the animation slower to generate
- For simple example shapes, the default `1000` is usually sufficient
- For detailed image contours, increasing this can help

### `--arms`

- Off: cleaner animation showing mostly the traced path
- On: shows the rotating circles and connecting arm segments used to build the path

### `--fps` and `--duration`

- Total frame count is `fps * duration`
- Increasing either one increases render time and output size

## Tips for Image Inputs

- Use a shape with a strong silhouette
- Prefer a plain background with high contrast
- Avoid very noisy images or photographs with many unrelated edges
- If contour extraction looks wrong, simplify the image first
- If no contour is found, try a cleaner black-and-white source image

## What the Animation Shows

The rendered scene includes:

- a faint "ghost" outline of the reconstructed full shape
- the traced path accumulated over time
- the current drawing tip
- optional arm segments and epicycle circles when `--arms` is enabled

## Troubleshooting

### `ModuleNotFoundError`

Install the missing Python dependencies:

```bash
pip install numpy matplotlib opencv-python Pillow
```

### `could not open image`

The path passed to `--image` could not be read. Check that the file exists and that the path is correct.

### `no contours found in image`

The script could not detect a usable external contour. Use a simpler image with clearer edges.

### MP4 saving fails

Make sure `ffmpeg` is installed and available on your shell `PATH`.

## Script Summary

File:

- [`epicycles.py`](/Users/ewellmeyer/Documents/fourier/epicycles.py)

Default behavior:

- source must be provided with `--image` or `--example`
- `--terms 50`
- `--points 1000`
- `--arms` disabled
- `--output show`
- `--fps 30`
- `--duration 8`
