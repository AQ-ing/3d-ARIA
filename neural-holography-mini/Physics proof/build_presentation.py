"""Build publication-style figures from existing results; no inference or training."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, ConnectionPatch
from PIL import Image
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST = ROOT / "test"
sys.path.insert(0, str(ROOT))
from utils.augmented_image_loader import resize_keep_aspect, pad_crop_to_res
from utils.utils import crop_image

GAUGE = TEST / "experiments/20261103_150411_encoder_gauge"
PHYSICS = TEST / "experiments/20261103_110638_depth_physics"
LENS = TEST / "experiments/20261103_150105_quadratic_depth"
FIGURES = HERE / "figures"
DATA = HERE / "data"
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.titleweight": "medium", "pdf.fonttype": 42})
TEAL, ORANGE, GRAY = "#008f86", "#d58116", "#6c7480"
manifest = {"reference_layout": "PHAROS main.pdf, Fig. 6-8; layout analogy only",
            "reference_pdf": "D:/wy/文献/main.pdf",
            "training_steps_added": 0,
            "display": "Saved 8-bit amplitude PNGs; no gamma, sharpening or histogram changes",
            "metrics": "Original floating-point amplitude PSNR from saved JSON, ROI 1600x880, optimal scalar; not recomputed from PNGs",
            "source_files": {}, "crops_xywh": {}}


def track(path):
    manifest["source_files"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(name, run=GAUGE):
    p = run / "output" / (name + ".png")
    track(p)
    return np.array(Image.open(p).convert("L"))


def load(run, name):
    p = run / "results.json"
    track(p)
    shutil.copy2(p, DATA / (name + ".json"))
    return json.loads(p.read_text(encoding="utf-8"))


gauge, physics, lens = load(GAUGE, "encoder_gauge"), load(PHYSICS, "depth_physics"), load(LENS, "quadratic_depth")
target = read("target")
depth_path = ROOT / "data/depth/0031.png"
track(depth_path)
dep = np.array(Image.open(depth_path).convert("L"), dtype=np.float32)[None] / 255
dep = pad_crop_to_res(resize_keep_aspect(dep, (880, 1600)), (1072, 1920))
dep = crop_image(dep, (880, 1600), pytorch=False, stacked_complex=False)[0]
mask = dep > 0.1
assert mask.shape == target.shape == (880, 1600)
Image.fromarray(np.uint8(mask)*255).save(DATA / "mask_near.png")
Image.fromarray(np.uint8(~mask)*255).save(DATA / "mask_far.png")
manifest["near_fraction_roi"] = float(mask.mean())
CROP = (760, 360, 230, 210)
manifest["crops_xywh"]["single_plane_inset"] = CROP


def panel(ax, im, title="", inset=None, color=TEAL, contour=False):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=7)
    for sp in ax.spines.values():
        sp.set_color("#d0d5da")
        sp.set_linewidth(.6)
    if contour:
        ax.contour(mask.astype(float), levels=[.5], colors=[TEAL], linewidths=.65)
    if inset is not None:
        x, y, w, h = inset
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1))
        ia = ax.inset_axes([.015, .025, .25, .45])
        ia.imshow(im[y:y+h, x:x+w], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ia.set_xticks([])
        ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.2)
        con = ConnectionPatch(xyA=(x, y+h), coordsA=ax.transData,
                              xyB=(1, 1), coordsB=ia.transAxes,
                              color=color, linewidth=.75, linestyle=":")
        ax.add_artist(con)


pdf = PdfPages(HERE / "results_figures.pdf")


def save(fig, name):
    fig.savefig(FIGURES / (name+".png"), dpi=200, facecolor="white")
    fig.savefig(FIGURES / (name+".pdf"), dpi=200, facecolor="white")
    pdf.savefig(fig, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved", name, flush=True)


# Figure 1: the paper's aligned columns, metric labels and identical detail insets.
fig, ax = plt.subplots(2, 4, figsize=(17.6, 7.4))
fig.subplots_adjust(left=.045, right=.99, top=.86, bottom=.11, wspace=.035, hspace=.25)
fig.suptitle("Single-plane depth transfer with a frozen HoloNet", x=.5, y=.97, fontsize=19)
fig.text(.5, .922, "No additional training  |  Same target, pixel pitch and output coordinates", ha="center", color=GRAY)
for r, z in enumerate((.18, .22)):
    row = next(x for x in gauge["depths"] if x["depth"] == z)
    ims = [target, read("carrier_aligned_0.2"), read(f"raw_{z}"), read(f"carrier_aligned_{z}")]
    names = ["Target amplitude", "Original model at 20 cm\n31.13 dB  (reference)",
             f"Change distance only\n{row['raw']['psnr']:.2f} dB",
             f"Carrier-aligned transfer\n{row['carrier_aligned']['psnr']:.2f} dB"]
    for c in range(4):
        panel(ax[r,c], ims[c], names[c], CROP)
    ax[r,0].text(-.07, .5, f"To {round(z*100)} cm", transform=ax[r,0].transAxes,
                 rotation=90, va="center", ha="center", weight="bold", fontsize=12)
fig.text(.045, .045, "Amplitude PSNR on the same 1600 x 880 ROI, with fitted brightness. The 20 cm column is a single-depth reference.\n"
         "All panels use the saved amplitude scale; identical green boxes locate the enlarged details. One image; simulation only.",
         fontsize=10, color=GRAY)
save(fig, "fig1_single_depth_transfer")

# Figure 2: full planes make the physically expected defocus visible.
fig, ax = plt.subplots(3, 3, figsize=(15.5, 10.7))
fig.subplots_adjust(left=.055, right=.99, top=.88, bottom=.10, wspace=.035, hspace=.25)
fig.suptitle("One phase-only hologram, two reconstruction depths", y=.975, fontsize=19)
fig.text(.5, .936, "18 cm / 22 cm  |  Frozen initial-phase and final encoding networks  |  No additional training", ha="center", color=GRAY)
titles = ["Desired content / target", "Without carrier alignment", "With carrier alignment"]
raw_metrics = gauge["multiplane"]["raw_sum_frozen_encoder"]
aligned_metrics = gauge["multiplane"]["aligned_sum_frozen_encoder"]
for r, kind in enumerate(("near", "far", "merged")):
    desired = target * (mask if r == 0 else ~mask) if r < 2 else target
    raw = read("raw_sum_frozen_encoder_"+kind)
    aligned = read("aligned_sum_frozen_encoder_"+kind)
    metric_key = kind+"_psnr" if r < 2 else "psnr"
    for c, im in enumerate((desired, raw, aligned)):
        label = titles[c] if r == 0 else ""
        if c:
            m = raw_metrics if c == 1 else aligned_metrics
            label += ("\n" if label else "") + f"{m[metric_key]:.2f} dB" + ("  (assigned region)" if r < 2 else "  (merged)")
        else:
            label += ("\n" if label else "") + (f"{18 if r == 0 else 22} cm layer mask" if r < 2 else "Full target")
        panel(ax[r,c], im, label)
    ax[r,0].text(-.07, .5, ("18 cm plane", "22 cm plane", "Depth-masked merge")[r],
                 transform=ax[r,0].transAxes, rotation=90, va="center", ha="center", weight="bold")
fig.text(.055, .035, "Each reconstruction column comes from ONE hologram; the last row selects pixels by the depth mask.\n"
         "Black areas in the target layer mark unscored pixels, not required darkness. Defocused light remains visible in the complete planes.\n"
         "One fitted brightness factor is shared by both planes and the merge within each method.", color=GRAY, fontsize=10)
save(fig, "fig2_two_depth_planes")

# Figure 3: full target, depth and binary layer assignment.
fig, ax = plt.subplots(1, 4, figsize=(17.6, 4.2))
fig.subplots_adjust(left=.025, right=.99, top=.79, bottom=.19, wspace=.045)
fig.suptitle("Input geometry and fixed two-depth assignment", y=.965, fontsize=18)
panel(ax[0], target, "Target amplitude")
ax[1].imshow(dep, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[1].set_title("Depth guide (normalized)")
ax[1].set_axis_off()
panel(ax[2], target*mask, f"18 cm: depth > 0.1\n{100*mask.mean():.1f}% of ROI pixels")
panel(ax[3], target*(~mask), f"22 cm: depth <= 0.1\n{100*(~mask).mean():.1f}% of ROI pixels")
fig.text(.025, .06, "RGB and depth use the same aspect-preserving resize/crop and padding as the recorded experiments.\n"
         "The guide supplies a binary assignment; its values are not interpreted as metric depth.", color=GRAY, fontsize=10)
save(fig, "fig3_input_and_depth")

# Figure 4: identical crops, including a boundary where artifacts are still visible.
crops = [("Near: locomotive grille", (870, 250, 280, 190), TEAL),
         ("Far: fine branches", (580, 0, 280, 190), ORANGE),
         ("Depth boundary: roof edge", (1120, 150, 280, 190), "#7651a8")]
fig, ax = plt.subplots(3, 4, figsize=(14, 8.8))
fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.095, wspace=.045, hspace=.22)
fig.suptitle("Local details at fixed pixel coordinates", y=.975, fontsize=19)
fig.text(.5, .929, "No local contrast normalization or sharpening; nearest-neighbor display", ha="center", color=GRAY)
ims = [target, read("raw_sum_frozen_encoder_merged"), read("aligned_sum_frozen_encoder_merged"), read("carrier_aligned_0.2")]
names = ["Target", "Two-depth: unaligned", "Two-depth: aligned", "Single-depth 20 cm\n(reference task)"]
for r, (name, box, color) in enumerate(crops):
    x, y, w, h = box
    manifest["crops_xywh"][name] = box
    for c in range(4):
        panel(ax[r,c], ims[c][y:y+h, x:x+w], names[c] if r == 0 else "")
        for sp in ax[r,c].spines.values():
            sp.set_color(color)
            sp.set_linewidth(1)
    ax[r,0].text(-.075, .5, name, transform=ax[r,0].transAxes, rotation=90,
                 va="center", ha="center", fontsize=10, color=color)
fig.text(.065, .025, "The aligned two-depth result still shows boundary ringing and residual texture. The 20 cm result is contextual,\n"
         "not a two-depth reconstruction. Exact crop coordinates and source hashes are recorded in manifest.json.", color=GRAY, fontsize=10)
save(fig, "fig4_detail_crops")

# Figure 5: honest quantitative comparison, separating tasks.
fig, ax = plt.subplots(1, 2, figsize=(13.8, 5.6))
fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.2, wspace=.27)
fig.suptitle("Quantitative validation without additional training", y=.97, fontsize=18)
depths = [x["depth"]*100 for x in gauge["depths"]]
raw = [x["raw"]["psnr"] for x in gauge["depths"]]
aligned = [x["carrier_aligned"]["psnr"] for x in gauge["depths"]]
ax[0].plot(depths, raw, "o--", color=GRAY, label="Change distance only", lw=1.8)
ax[0].plot(depths, aligned, "o-", color=TEAL, label="Carrier aligned", lw=2.3)
lens_samples = sorted([(x["depth"]*100, x["prewarp_then_lens_vs_original_target"]["psnr"]) for x in lens["depths"]]
                      + [(20., lens["baseline"]["psnr"])])
ax[0].plot([x[0] for x in lens_samples], [x[1] for x in lens_samples],
           "s:", color=ORANGE, label="Prewarp + quadratic phase", lw=1.5)
ax[0].set(xlabel="Reconstruction depth (cm)", ylabel="Amplitude PSNR (dB)", title="Same image moved to one depth", xticks=depths, ylim=(10,34))
ax[0].legend(fontsize=9, frameon=False)
ax[0].grid(alpha=.2)
xx = np.arange(3)
for offset, metric, label, color in ((-.18, raw_metrics, "Unaligned", GRAY), (.18, aligned_metrics, "Aligned", TEAL)):
    vals = [metric[k] for k in ("near_psnr", "far_psnr", "psnr")]
    bars = ax[1].bar(xx+offset, vals, .34, label=label, color=color)
    ax[1].bar_label(bars, fmt="%.2f", fontsize=10, padding=4)
ax[1].set(xticks=xx, xticklabels=["18 cm region", "22 cm region", "Merged"], ylim=(0,30),
          ylabel="Amplitude PSNR (dB)", title="Two depths displayed simultaneously")
ax[1].legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
ax[1].grid(axis="y", alpha=.2)
ax[1].set_axisbelow(True)
fig.text(.065, .055, "Metrics use original float records. At 20 cm the lens transform is identity; its recorded baseline is shown.\n"
         "Single-image simulation only. The two panels are different tasks; neither is directly comparable with PHAROS dataset scores.",
         fontsize=10, color=GRAY)
save(fig, "fig5_quantitative")
pdf.close()

# Keep an independent proof snapshot and a derivation appendix with the figures.
source = TEST / "DEPTH_PHYSICS_VALIDATION.md"
track(source)
proof = source.read_text(encoding="utf-8")
proof = proof.replace("全部验证与新增文件均在 test 下；未修改主目录，未更新网络权重。",
                      "数值验证脚本位于 test 下；本文与展示图按要求归档于 Physics proof。未修改主目录训练代码，未更新网络权重。")
proof += "\n\n" + (HERE / "derivation_appendix.md").read_text(encoding="utf-8")
(HERE / "physics_proof.md").write_text(proof, encoding="utf-8")
(HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

rows = ["task,method,depth_cm,psnr_db"]
for row in gauge["depths"]:
    for method in ("raw", "carrier_aligned"):
        rows.append(f"single_depth,{method},{row['depth']*100:g},{row[method]['psnr']:.8f}")
for method, row in gauge["multiplane"].items():
    for key, depth in (("near_psnr", "18"), ("far_psnr", "22"), ("psnr", "merged")):
        rows.append(f"two_depth,{method},{depth},{row[key]:.8f}")
for depth, psnr in lens_samples:
    rows.append(f"single_depth,prewarp_quadratic,{depth:g},{psnr:.8f}")
(DATA / "metrics.csv").write_text("\n".join(rows)+"\n", encoding="utf-8")
print("Finished figures and proof snapshot.", flush=True)
