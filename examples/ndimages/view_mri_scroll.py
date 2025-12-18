import nibabel as nib
import matplotlib.pyplot as plt

# ============================================================
# Load original and transformed MRI
# ============================================================

orig_nii = nib.load("heat/datasets/flair.nii.gz")
trans_nii = nib.load("heat/datasets/x_transformed.nii.gz")

orig = orig_nii.get_fdata()
trans = trans_nii.get_fdata()

# Sanity check
assert orig.shape == trans.shape, "Original and transformed shapes do not match!"

num_slices = orig.shape[0]
slice_idx = num_slices // 2  # start in the middle


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

img_orig = ax[0].imshow(orig[slice_idx], cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

img_trans = ax[1].imshow(trans[slice_idx], cmap="gray")
ax[1].set_title("Transformed")
ax[1].axis("off")

fig.suptitle(f"Slice {slice_idx}/{num_slices - 1}")


# ============================================================
# Keyboard interaction
# ============================================================

def on_key(event):
    global slice_idx

    if event.key == "up":
        slice_idx = min(slice_idx + 1, num_slices - 1)
    elif event.key == "down":
        slice_idx = max(slice_idx - 1, 0)
    else:
        return

    img_orig.set_data(orig[slice_idx])
    img_trans.set_data(trans[slice_idx])
    fig.suptitle(f"Slice {slice_idx}/{num_slices - 1}")
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
