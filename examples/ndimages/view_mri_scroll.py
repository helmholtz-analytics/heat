import nibabel as nib
import matplotlib.pyplot as plt

# ============================================================
# Paths (ABSOLUTE – adjust if needed)
# ============================================================

BASE = "/Users/marka.k/1900_Image_transformations/heat/heat/datasets"

paths = {
    "Original":   f"{BASE}/flair.nii.gz",
    "Identity":   f"{BASE}/mri_identity.nii.gz",
    "Scaled":     f"{BASE}/mri_scaled.nii.gz",
    "Rotated":    f"{BASE}/mri_rotated.nii.gz",
    "Translated": f"{BASE}/mri_translated.nii.gz",
}

# ============================================================
# Load volumes
# ============================================================

volumes = {}
for name, path in paths.items():
    volumes[name] = nib.load(path).get_fdata()

# Sanity check: all shapes equal
shapes = {v.shape for v in volumes.values()}
assert len(shapes) == 1, "Not all volumes have the same shape!"

D, H, W = next(iter(shapes))
slice_idx = D // 2

# ============================================================
# Create figure
# ============================================================

titles = list(volumes.keys())
data = list(volumes.values())

fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 5))
images = []

for ax, title, vol in zip(axes, titles, data):
    img = ax.imshow(vol[slice_idx], cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    images.append(img)

fig.suptitle(f"Slice {slice_idx}/{D - 1}")

# ============================================================
# Keyboard navigation
# ============================================================

def on_key(event):
    global slice_idx

    if event.key == "up":
        slice_idx = min(slice_idx + 1, D - 1)
    elif event.key == "down":
        slice_idx = max(slice_idx - 1, 0)
    else:
        return

    for img, vol in zip(images, data):
        img.set_data(vol[slice_idx])

    fig.suptitle(f"Slice {slice_idx}/{D - 1}")
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
