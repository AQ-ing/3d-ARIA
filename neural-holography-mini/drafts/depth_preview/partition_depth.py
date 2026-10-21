"""Standalone fixed-threshold depth partition previews; no model or training imports."""

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_image(path, flags):
    # imdecode supports Windows paths containing non-ASCII characters.
    result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if result is None:
        raise ValueError(f"Could not read image: {path}")
    return result


def write_image(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", pixels)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def partition_depth(depth, threshold):
    """Input: grayscale depth / 255 after resizing. Dark region includes equality."""
    dark = depth <= threshold
    return dark, ~dark


def tile(pixels, label, width, height):
    if pixels.ndim == 2:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    # Preserve the processing aspect ratio, including for tall input images.
    scale = min(width / pixels.shape[1], height / pixels.shape[0])
    resized = cv2.resize(pixels, (max(1, round(pixels.shape[1] * scale)),
                                  max(1, round(pixels.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 38, width, 3), 28, dtype=np.uint8)
    left = (width - resized.shape[1]) // 2
    top = 38 + (height - resized.shape[0]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.51, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def partition_images(rgb, dark, other):
    return (dark.astype(np.uint8) * 255, other.astype(np.uint8) * 255,
            np.where(dark[..., None], rgb, 0), np.where(other[..., None], rgb, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="data/image/drink.png")
    parser.add_argument("--depth", default="data/depth/drink.png")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Selected grayscale threshold in [0, 1], not an area fraction")
    parser.add_argument("--compare-thresholds", type=float, nargs="+",
                        default=[0.02, 0.05, 0.10, 0.50])
    parser.add_argument("--size", type=int, nargs=2, default=[1920, 1080],
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", default="output/depth_threshold_comparison_chinese")
    args = parser.parse_args()
    thresholds = sorted(set([args.threshold, *args.compare_thresholds]))
    if any(not 0 <= t <= 1 for t in thresholds):
        parser.error("All thresholds must be in [0, 1]")
    if min(args.size) < 1:
        parser.error("--size dimensions must be positive")

    rgb = read_image(project_path(args.image), cv2.IMREAD_COLOR)
    depth = read_image(project_path(args.depth), cv2.IMREAD_GRAYSCALE)
    if rgb.shape[:2] != depth.shape:
        parser.error("RGB and depth must have the same source dimensions and alignment")
    rgb = cv2.resize(rgb, tuple(args.size), interpolation=cv2.INTER_LINEAR)
    # Match training's ordering: normalize first, bilinear resize, then threshold.
    depth = cv2.resize(depth.astype(np.float32) / 255.0, tuple(args.size),
                       interpolation=cv2.INTER_LINEAR)
    depth_u8 = np.rint(np.clip(depth, 0, 1) * 255).astype(np.uint8)
    output = project_path(args.output)
    comparison_rows = [np.hstack([
        tile(rgb, "RGB input", 720, 270),
        tile(depth_u8, "Depth / 255 (black = low value)", 720, 270),
    ])]
    selected_preview = None

    for threshold in thresholds:
        dark, other = partition_depth(depth, threshold)
        images = partition_images(rgb, dark, other)
        dark_pct = dark.mean() * 100
        other_pct = 100 - dark_pct
        tag = format(threshold, ".8g").replace(".", "p")
        folder = output / f"t{tag}"
        for name, pixels in zip(("mask_plane1", "mask_plane2", "rgb_plane1", "rgb_plane2"), images):
            write_image(folder / f"{name}.png", pixels)
        labels = [f"t={threshold:g} dark mask: {dark_pct:.2f}%",
                  f"other mask: {other_pct:.2f}%", "Dark region RGB", "Other region RGB"]
        comparison_rows.append(np.hstack([
            tile(pixels, label, 360, 203) for pixels, label in zip(images, labels)
        ]))
        if threshold == args.threshold:
            selected_preview = np.vstack([
                np.hstack([tile(rgb, "RGB input", 600, 338),
                           tile(depth_u8, f"Depth; threshold = {threshold:g} ({threshold * 255:g}/255)", 600, 338)]),
                np.hstack([tile(images[0], f"Plane 1 mask: D <= {threshold:g}; {dark_pct:.2f}%", 600, 338),
                           tile(images[2], "Plane 1: dark region RGB", 600, 338)]),
                np.hstack([tile(images[1], f"Plane 2 mask: D > {threshold:g}; {other_pct:.2f}%", 600, 338),
                           tile(images[3], "Plane 2: other region RGB", 600, 338)]),
            ])
        print(f"threshold={threshold:g} ({threshold * 255:g}/255): "
              f"dark={dark_pct:.4f}%, other={other_pct:.4f}%; {folder}")

    write_image(output / "threshold_comparison.png", np.vstack(comparison_rows))
    write_image(output / "selected_partition_preview.png", selected_preview)
    print(f"Selected preview: {output / 'selected_partition_preview.png'}")


if __name__ == "__main__":
    main()
