import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import cv2
import argparse
import sys
import os


def normalize_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x -= x.mean()
    y -= y.mean()

    scale = max(np.ptp(x), np.ptp(y)) / 2
    if scale == 0:
        raise ValueError("shape has zero size after sampling")

    x /= scale
    y /= scale
    return x, y


def contour_score(contour, width, height):
    x, y, w, h = cv2.boundingRect(contour)
    touches_border = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1

    score = cv2.arcLength(contour, True)
    if touches_border:
        score *= 0.5

    # De-prioritize full-frame regions, which are usually the background.
    if w >= width - 2 and h >= height - 2:
        score *= 0.25

    return score


def contour_set_score(contours, width, height):
    if not contours:
        return -np.inf

    scores = [contour_score(contour, width, height) for contour in contours[:6]]
    return scores[0] + 0.2 * sum(scores[1:])


def find_candidate_contours(mask, include_internal=False):
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    height, width = closed.shape[:2]
    min_perimeter = max(20.0, 0.02 * (width + height))

    raw_contours = []
    if include_internal:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        min_area = max(8, int(0.0001 * width * height))

        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_area:
                continue

            component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                raw_contours.append(max(contours, key=lambda contour: cv2.arcLength(contour, True)))
    else:
        raw_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not raw_contours:
        return []

    candidates = []
    for contour in raw_contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter < min_perimeter:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w >= width - 2 and h >= height - 2:
            continue

        candidates.append(contour)

    candidates.sort(key=lambda contour: contour_score(contour, width, height), reverse=True)
    return candidates


def extract_contours(img, include_internal=False):
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        if np.any(alpha < 250):
            contours = find_candidate_contours((alpha > 0).astype(np.uint8) * 255, include_internal)
            if contours:
                return contours

    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_contours = []
    best_score = -np.inf
    height, width = gray.shape[:2]

    for mask in (255 - otsu, otsu):
        contours = find_candidate_contours(mask, include_internal)
        if not contours:
            continue

        score = contour_set_score(contours, width, height)
        if score > best_score:
            best_contours = contours
            best_score = score

    if best_contours:
        return best_contours

    edges = cv2.Canny(blurred, 50, 150)
    contours = find_candidate_contours(edges, include_internal)
    if contours:
        return contours

    raise RuntimeError("no contours found in image, try a simpler high-contrast image")


def contour_to_points(contour):
    pts = contour[:, 0, :].astype(float)
    if len(pts) < 2:
        return None

    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts = pts[keep]
    if len(pts) < 2:
        return None

    return pts


def nearest_point_index(points, anchor):
    deltas = points - anchor
    return int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))


def nearest_pair_indices(points_a, points_b):
    deltas = points_a[:, None, :] - points_b[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", deltas, deltas)
    index = int(np.argmin(dist2))
    i, j = divmod(index, dist2.shape[1])
    return i, j


def rotate_loop(points, start_idx):
    return np.roll(points, -start_idx, axis=0)


def orient_loop(points, anchor, start_idx):
    rotated = rotate_loop(points, start_idx)
    if len(rotated) <= 2:
        return rotated

    reversed_rotated = np.concatenate((rotated[:1], rotated[:0:-1]), axis=0)
    if np.linalg.norm(reversed_rotated[1] - anchor) < np.linalg.norm(rotated[1] - anchor):
        return reversed_rotated

    return rotated


def stitch_contours(contours):
    loops = [contour_to_points(contour) for contour in contours]
    loops = [loop for loop in loops if loop is not None]
    if not loops:
        raise RuntimeError("no contours found in image, try a simpler high-contrast image")

    current = loops.pop(0)
    if loops:
        next_choice = min(
            (
                (nearest_pair_indices(current, loop), idx)
                for idx, loop in enumerate(loops)
            ),
            key=lambda item: np.linalg.norm(current[item[0][0]] - loops[item[1]][item[0][1]]),
        )
        start_idx, _ = next_choice[0]
        current = rotate_loop(current, start_idx)

    stitched = [current]
    current_anchor = current[0]

    while loops:
        best_idx = None
        best_start_idx = None
        best_distance = np.inf

        for idx, loop in enumerate(loops):
            start_idx = nearest_point_index(loop, current_anchor)
            distance = np.linalg.norm(loop[start_idx] - current_anchor)
            if distance < best_distance:
                best_idx = idx
                best_start_idx = start_idx
                best_distance = distance

        next_loop = loops.pop(best_idx)
        next_loop = orient_loop(next_loop, current_anchor, best_start_idx)
        stitched.append(next_loop)
        current_anchor = next_loop[0]

    path = np.vstack([stitched[0], stitched[0][0]])
    for loop in stitched[1:]:
        path = np.vstack([path, loop, loop[0]])

    return path


def sample_path(points, n, closed=True):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        raise RuntimeError("detected contour has too few points")

    if closed:
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    keep = np.concatenate(([True], segment_lengths > 0))
    pts = pts[keep]

    diffs = np.diff(pts, axis=0)
    arc = np.concatenate([[0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = arc[-1]
    if total == 0:
        raise RuntimeError("detected contour has zero length")

    t_new = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t_new, arc, pts[:, 0])
    y = np.interp(t_new, arc, pts[:, 1])
    return x, y


# built-in example shapes defined as parametric curves sampled at n points
def make_example_shape(name, n=1000):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        return normalize_curve(x, y)

    if name == "star":
        # five-pointed star via radius modulation
        r = 1 + 0.5 * np.cos(5 * t)
        return normalize_curve(r * np.cos(t), r * np.sin(t))

    if name == "lissajous":
        return normalize_curve(np.sin(3 * t + np.pi/4), np.sin(2 * t))

    if name == "trefoil":
        r = np.cos(3 * t / 2)
        return normalize_curve(r * np.cos(t), r * np.sin(t))

    raise ValueError(f"unknown example shape: {name}")


# extract a contour from an image file and return it as (x, y) arrays
def contour_from_image(path, n=1000, stitch=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not open image: {path}")

    contours = extract_contours(img, include_internal=stitch)
    if stitch:
        pts = stitch_contours(contours)
        x, y = sample_path(pts, n, closed=True)
    else:
        pts = contour_to_points(contours[0])
        x, y = sample_path(pts, n, closed=True)

    # center and normalize so the shape fits nicely
    x, y = normalize_curve(x, y)
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
    if num_terms <= 0:
        return 0 + 0j

    active_freqs = freqs[:num_terms]
    active_coeffs = coeffs[:num_terms]
    angles = 2 * np.pi * active_freqs * t
    return np.sum(active_coeffs * np.exp(1j * angles))


def progressive_term_weights(num_terms, progress):
    if num_terms <= 0:
        return np.zeros(0, dtype=float), 0.0

    cursor = 1 + progress * (num_terms - 1)
    weights = np.clip(cursor - np.arange(num_terms), 0.0, 1.0)
    weights = weights * weights * (3 - 2 * weights)
    return weights, cursor


# build the positions of each epicycle tip as we walk outward from center
def epicycle_tips(freqs, coeffs, t, num_terms, weights=None):
    if num_terms <= 0:
        return np.array([0 + 0j])

    active_freqs = freqs[:num_terms]
    active_coeffs = coeffs[:num_terms]
    if weights is None:
        weights = np.ones(num_terms, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    angles = 2 * np.pi * active_freqs * t
    contributions = active_coeffs * weights * np.exp(1j * angles)
    return np.concatenate(([0 + 0j], np.cumsum(contributions)))


def run(freqs, coeffs, num_terms, show_arms, output, fps=30, duration=8, build_duration=3.0):
    n_frames = fps * duration
    build_frames = min(int(round(max(build_duration, 0) * fps)), max(n_frames - 1, 0))
    trace_frames = n_frames - build_frames

    build_times = np.linspace(0, 1, build_frames, endpoint=False) if build_frames else np.array([])
    trace_times = np.linspace(0, 1, trace_frames, endpoint=False) if trace_frames else np.array([])

    active_freqs = freqs[:num_terms]
    active_coeffs = coeffs[:num_terms]

    # pre-trace the full target curve at this many terms so we can draw the ghost
    ghost_t = np.linspace(0, 1, 800, endpoint=False)
    ghost_basis = np.exp(2j * np.pi * np.outer(ghost_t, active_freqs))
    ghost = ghost_basis @ active_coeffs

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.axis("off")

    # fit both the traced curve and the epicycle arm chain in view.
    curve_extent = max(np.abs(ghost.real).max(), np.abs(ghost.imag).max())
    arm_extent = np.abs(coeffs[:num_terms]).sum()
    pad = max(curve_extent, arm_extent) * 1.1
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    # ghost outline of the full shape
    ax.plot(ghost.real, ghost.imag, color="#ffffff18", lw=0.8, zorder=1)

    # the traced path so far, the arm lines, and the endpoint dot
    traced_x, traced_y = [], []
    build_line, = ax.plot([], [], color="#e8c46a", lw=1.1, alpha=0.85, zorder=3)
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

    status_text = ax.text(
        0.5, 0.98, "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#888888",
        fontsize=10,
        zorder=5,
    )
    trace_started = False

    def init():
        nonlocal trace_started
        trace_started = False
        traced_x.clear()
        traced_y.clear()

        build_line.set_data([], [])
        path_line.set_data([], [])
        dot.set_data([], [])
        status_text.set_text(f"building 1/{num_terms} terms" if build_frames else f"{num_terms} epicycles")

        for line in arm_lines:
            line.set_data([], [])

        for circ in arm_circles:
            circ.center = (0, 0)
            circ.radius = 0

        return [build_line, path_line, dot, status_text] + arm_lines + arm_circles

    def update(frame):
        nonlocal trace_started

        if frame < build_frames:
            progress = frame / max(build_frames - 1, 1)
            weights, cursor = progressive_term_weights(num_terms, progress)
            partial_curve = ghost_basis @ (active_coeffs * weights)
            build_line.set_data(partial_curve.real, partial_curve.imag)
            path_line.set_data([], [])

            t = build_times[frame]
            tips = epicycle_tips(active_freqs, active_coeffs, t, num_terms, weights)
            status_text.set_text(f"building {min(num_terms, int(np.ceil(cursor)))}/{num_terms} terms")
        else:
            if not trace_started:
                traced_x.clear()
                traced_y.clear()
                trace_started = True

            build_line.set_data([], [])

            t = trace_times[frame - build_frames]
            tips = epicycle_tips(active_freqs, active_coeffs, t, num_terms)
            status_text.set_text(f"{num_terms} epicycles")

        if show_arms:
            for k, (line, circ) in enumerate(zip(arm_lines, arm_circles)):
                p0 = tips[k]
                p1 = tips[k + 1]
                line.set_data([p0.real, p1.real], [p0.imag, p1.imag])
                r = abs(p1 - p0)
                circ.center = (p0.real, p0.imag)
                circ.radius = r

        tip = tips[-1]
        if frame >= build_frames:
            traced_x.append(tip.real)
            traced_y.append(tip.imag)
            path_line.set_data(traced_x, traced_y)

        dot.set_data([tip.real], [tip.imag])

        return [build_line, path_line, dot, status_text] + arm_lines + arm_circles

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
    parser.add_argument("--stitch-contours", action="store_true",
        help="for image input, stitch multiple detected contours into one continuous path to preserve interior details")
    parser.add_argument("--arms", action="store_true",
        help="show spinning epicycle arms and circles")
    parser.add_argument("--output", default="show",
        help="'show' to display interactively, or a filename ending in .mp4 or .gif")
    parser.add_argument("--fps", type=int, default=30,
        help="frames per second (default 30)")
    parser.add_argument("--duration", type=int, default=8,
        help="animation duration in seconds (default 8)")
    parser.add_argument("--build-duration", type=float, default=3.0,
        help="seconds spent progressively adding Fourier terms before tracing (default 3; use 0 to disable)")

    args = parser.parse_args()

    print("loading shape...")
    if args.image:
        x, y = contour_from_image(args.image, n=args.points, stitch=args.stitch_contours)
    else:
        x, y = make_example_shape(args.example, n=args.points)

    print("computing dft...")
    freqs, coeffs = compute_dft(x, y)

    num_terms = min(args.terms, len(freqs))
    print(f"animating with {num_terms} terms, arms={'on' if args.arms else 'off'}, output={args.output}")

    run(freqs, coeffs, num_terms, args.arms, args.output, args.fps, args.duration, args.build_duration)


if __name__ == "__main__":
    main()
