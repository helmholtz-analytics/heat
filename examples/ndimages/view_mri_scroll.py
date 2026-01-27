import os
import nibabel as nib
import matplotlib.pyplot as plt

# ============================================================
# Paths (ONLY files that actually exist)
# ============================================================

paths = {
    "Original": "/Users/marka.k/1900_Image_transformations/heat/heat/datasets/flair.nii.gz",

}

# ============================================================
# Load volumes safely
# ============================================================

volumes = {}
for name, path in paths.items():
    if not os.path.exists(path):
        print(f"[SKIP] {name}: file not found")
        continue

    vol = nib.load(path).get_fdata()
    volumes[name] = vol
    print(f"[LOAD] {name}: shape={vol.shape}")

if not volumes:
    raise RuntimeError("No volumes loaded")

# ============================================================
# Setup figure
# ============================================================

titles = list(volumes.keys())
data = list(volumes.values())
depths = [v.shape[0] for v in data]

slice_indices = [d // 2 for d in depths]  # one index per volume

fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 5))
if len(data) == 1:
    axes = [axes]

images = []

for ax, title, vol, idx in zip(axes, titles, data, slice_indices):
    img = ax.imshow(vol[idx])
    ax.set_title(f"{title}\nslice {idx}")
    ax.axis("off")
    images.append(img)

fig.suptitle("Independent slice scrolling per volume")

# ============================================================
# Keyboard navigation (ALL volumes together)
# ============================================================

def on_key(event):
    for i, vol in enumerate(data):
        if event.key == "up":
            slice_indices[i] = min(slice_indices[i] + 1, vol.shape[0] - 1)
        elif event.key == "down":
            slice_indices[i] = max(slice_indices[i] - 1, 0)
        else:
            return

        images[i].set_data(vol[slice_indices[i]])
        axes[i].set_title(f"{titles[i]}\nslice {slice_indices[i]}")

    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
