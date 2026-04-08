import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import cv2
import argparse
import sys
import os

# built-in example shapes defined as parametric curves sampled at n points
def make_example_shape(name, n=1000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        return x, -y  # flip so it faces up

    if name == "star":
        # five-pointed star via radius modulation
        r = 1 + 0.5 * np.cos(5 * t)
        return r * np.cos(t), r * np.sin(t)

    if name == "lissajous":
        return np.sin(3 * t + np.pi/4), np.sin(2 * t)

    if name == "trefoil":
        r = np.cos(3 * t / 2)
        return r * np.cos(t), r * np.sin(t)

    raise ValueError(f"unknown example shape: {name}")


# extract a contour from an image file and return it as (x, y) arrays
def contour_from_image(path, n=1000):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not open image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur slightly, then canny edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("no contours found in image, try a simpler high-contrast image")

    # pick the longest contour and resample it uniformly
    longest = max(contours, key=lambda c: len(c))
    pts = longest[:, 0, :]  # shape (m, 2)

    # parameterize by arc length then resample to n points
    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = arc[-1]
    t_new = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t_new, arc, pts[:, 0])
    y = np.interp(t_new, arc, pts[:, 1])

    # center and normalize so the shape fits nicely
    x -= x.mean()
    y -= y.mean()
    scale = max(x.ptp(), y.ptp()) / 2
    x /= scale
    y /= scale
    y = -y  # flip because image y goes downward

    return x, y


# compute the dft coefficients sorted by amplitude, largest first
# we treat the curve as complex signal z = x + iy
def compute_dft(x, y):
    z = x + 1j * y
    n = len(z)
    coeffs = np.fft.fft(z) / n

    freqs = np.fft.fftfreq(n, d=1.0/n).astype(int)
    amplitudes = np.abs(coeffs)

    order = np.argsort(amplitudes)[::-1]
    return freqs[order], coeffs[order]


# evaluate the partial reconstruction at a given phase t in [0, 2pi)
# using only the first num_terms epicycles
def reconstruct(freqs, coeffs, t, num_terms):
    terms = np.arange(num_terms)
    angles = 2 * np.pi * freqs[terms] * t
    return np.sum(coeffs[terms] * np.exp(1j * angles))


# build the positions of each epicycle tip as we walk outward from center
def epicycle_tips(freqs, coeffs, t, num_terms):
    pos = 0 + 0j
    tips = [pos]
    for k in range(num_terms):
        angle = 2 * np.pi * freqs[k] * t
        pos += coeffs[k] * np.exp(1j * angle)
        tips.append(pos)
    return tips


def run(freqs, coeffs, num_terms, show_arms, output, fps=30, duration=8):
    n_frames = fps * duration
    times = np.linspace(0, 1, n_frames, endpoint=False)

    # pre-trace the full target curve at this many terms so we can draw the ghost
    ghost_t = np.linspace(0, 1, 800, endpoint=False)
    ghost = np.array([reconstruct(freqs, coeffs, t, num_terms) for t in ghost_t])

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.axis("off")

    # set limits with a bit of padding around the unit circle scale
    pad = 1.4
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    # ghost outline of the full shape
    ax.plot(ghost.real, ghost.imag, color="#ffffff18", lw=0.8, zorder=1)

    # the traced path so far, the arm lines, and the endpoint dot
    traced_x, traced_y = [], []
    path_line, = ax.plot([], [], color="#e8c46a", lw=1.2, zorder=3)
    dot, = ax.plot([], [], "o", color="#e8c46a", ms=3, zorder=4)

    # arms and circles only drawn if show_arms is on
    arm_lines = []
    arm_circles = []
    if show_arms:
        for _ in range(num_terms):
            line, = ax.plot([], [], color="#ffffff30", lw=0.7, zorder=2)
            arm_lines.append(line)
            circ = Circle((0, 0), 0, fill=False, color="#ffffff18", lw=0.5, zorder=2)
            ax.add_patch(circ)
            arm_circles.append(circ)

    title_str = f"{num_terms} epicycles"
    ax.set_title(title_str, color="#888888", fontsize=10, pad=8)

    def init():
        path_line.set_data([], [])
        dot.set_data([], [])
        return [path_line, dot] + arm_lines + arm_circles

    def update(frame):
        t = times[frame]
        tips = epicycle_tips(freqs, coeffs, t, num_terms)

        if show_arms:
            for k, (line, circ) in enumerate(zip(arm_lines, arm_circles)):
                p0 = tips[k]
                p1 = tips[k + 1]
                line.set_data([p0.real, p1.real], [p0.imag, p1.imag])
                r = abs(coeffs[k])
                circ.center = (p0.real, p0.imag)
                circ.radius = r

        tip = tips[-1]
        traced_x.append(tip.real)
        traced_y.append(tip.imag)
        path_line.set_data(traced_x, traced_y)
        dot.set_data([tip.real], [tip.imag])

        return [path_line, dot] + arm_lines + arm_circles

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=1000 // fps, blit=True
    )

    if output == "show":
        plt.show()
    elif output.endswith(".gif"):
        print(f"saving gif to {output} (this may take a moment)...")
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer)
        print("done.")
    else:
        print(f"saving mp4 to {output}...")
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)
        print("done.")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="draw fourier epicycle animations from images or built-in shapes"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", metavar="PATH",
        help="path to an image file to extract contour from")
    source.add_argument("--example", metavar="NAME",
        choices=["heart", "star", "lissajous", "trefoil"],
        help="use a built-in example shape")

    parser.add_argument("--terms", type=int, default=50,
        help="number of epicycle terms to use (default 50)")
    parser.add_argument("--points", type=int, default=1000,
        help="number of contour sample points (default 1000)")
    parser.add_argument("--arms", action="store_true",
        help="show spinning epicycle arms and circles")
    parser.add_argument("--output", default="show",
        help="'show' to display interactively, or a filename ending in .mp4 or .gif")
    parser.add_argument("--fps", type=int, default=30,
        help="frames per second (default 30)")
    parser.add_argument("--duration", type=int, default=8,
        help="animation duration in seconds (default 8)")

    args = parser.parse_args()

    print("loading shape...")
    if args.image:
        x, y = contour_from_image(args.image, n=args.points)
    else:
        x, y = make_example_shape(args.example, n=args.points)

    print("computing dft...")
    freqs, coeffs = compute_dft(x, y)

    num_terms = min(args.terms, len(freqs))
    print(f"animating with {num_terms} terms, arms={'on' if args.arms else 'off'}, output={args.output}")

    run(freqs, coeffs, num_terms, args.arms, args.output, args.fps, args.duration)


if __name__ == "__main__":
    main()
