import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def load_nifti(path):
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    return nii, data


def save_nifti(data, affine, header, path):
    out = nib.Nifti1Image(data, affine, header=header)
    nib.save(out, str(path))


def keep_largest_component(mask, min_size=10000):
    labeled_mask, num_components = ndi.label(mask)

    if num_components == 0:
        return mask

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)

    if sizes[largest_label] < min_size:
        return mask

    return labeled_mask == largest_label


def create_body_mask(image, threshold, closing_iters=1, opening_iters=0, min_size=10000):
    mask = image > threshold

    if closing_iters > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iters)

    if opening_iters > 0:
        mask = ndi.binary_opening(mask, iterations=opening_iters)

    mask = ndi.binary_fill_holes(mask)
    mask = keep_largest_component(mask, min_size)

    return mask


def get_bounding_box(mask):
    coords = np.where(mask)

    if coords[0].size == 0:
        return None

    x0, y0, z0 = [int(c.min()) for c in coords]
    x1, y1, z1 = [int(c.max()) + 1 for c in coords]

    return x0, x1, y0, y1, z0, z1


def expand_bounding_box(bbox, image_shape, margin_xy, margin_z, crop_mode):
    x0, x1, y0, y1, z0, z1 = bbox

    x0 = max(0, x0 - margin_xy)
    x1 = min(image_shape[0], x1 + margin_xy)

    y0 = max(0, y0 - margin_xy)
    y1 = min(image_shape[1], y1 + margin_xy)

    if crop_mode == "xy_only":
        z0, z1 = 0, image_shape[2]
    else:
        z0 = max(0, z0 - margin_z)
        z1 = min(image_shape[2], z1 + margin_z)

    return (
        slice(x0, x1),
        slice(y0, y1),
        slice(z0, z1),
    )


def update_affine_for_crop(original_affine, crop_slices):
    crop_start = np.array([
        crop_slices[0].start,
        crop_slices[1].start,
        crop_slices[2].start,
        1.0
    ])

    new_affine = original_affine.copy()
    # Adjust spatial origin after cropping
    new_affine[:3, 3] = (original_affine @ crop_start)[:3]

    return new_affine


def process_case(
    image_path,
    label_path,
    output_image_path,
    output_label_path,
    threshold,
    margin_xy,
    margin_z,
    crop_mode
):
    image_nii, image = load_nifti(image_path)
    label_nii, label = load_nifti(label_path)

    if image.ndim != 3:
        raise ValueError(f"Image is not 3D: {image_path}")

    if label.ndim != 3:
        raise ValueError(f"Label is not 3D: {label_path}")

    original_shape = list(image.shape)

    body_mask = create_body_mask(image, threshold)
    bbox = get_bounding_box(body_mask)

    if bbox is None:
        crop_slices = (
            slice(0, image.shape[0]),
            slice(0, image.shape[1]),
            slice(0, image.shape[2]),
        )
    else:
        crop_slices = expand_bounding_box(
            bbox,
            image.shape,
            margin_xy,
            margin_z,
            crop_mode,
        )

    cropped_image = image[crop_slices].astype(np.float32)
    cropped_label = np.rint(label[crop_slices]).astype(np.uint8)

    image_header = image_nii.header.copy()
    label_header = label_nii.header.copy()

    image_header.set_data_dtype(np.float32)
    label_header.set_data_dtype(np.uint8)

    save_nifti(
        cropped_image,
        update_affine_for_crop(image_nii.affine, crop_slices),
        image_header,
        output_image_path
    )

    save_nifti(
        cropped_label,
        update_affine_for_crop(label_nii.affine, crop_slices),
        label_header,
        output_label_path
    )

    return {
        "case": image_path.name.replace("_0000.nii.gz", ""),
        "original_shape": original_shape,
        "cropped_shape": list(cropped_image.shape),
        "crop_bbox": {
            "x": [crop_slices[0].start, crop_slices[0].stop],
            "y": [crop_slices[1].start, crop_slices[1].stop],
            "z": [crop_slices[2].start, crop_slices[2].stop],
        },
        "label_voxels_after_crop": int(np.sum(cropped_label > 0)),
        "image_stats_after_crop": {
            "min": float(cropped_image.min()),
            "max": float(cropped_image.max()),
            "mean": float(cropped_image.mean()),
            "std": float(cropped_image.std()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Crop CT scans to body region for nnU-Net")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--body-threshold", type=float, default=-650.0)
    parser.add_argument("--crop-mode", choices=["xy_only", "xyz"], default="xy_only")
    parser.add_argument("--margin-xy", type=int, default=20)
    parser.add_argument("--margin-z", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    src_images = src / "imagesTr"
    src_labels = src / "labelsTr"
    dst_images = dst / "imagesTr"
    dst_labels = dst / "labelsTr"

    if dst.exists():
        shutil.rmtree(dst)

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_images.glob("*_0000.nii.gz"))
    if not image_files:
        raise RuntimeError(f"No training images found in {src_images}")

    reports = []

    for i, image_path in enumerate(image_files, start=1):
        case_id = image_path.name.replace("_0000.nii.gz", "")
        label_path = src_labels / f"{case_id}.nii.gz"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for case {case_id}: {label_path}")

        report = process_case(
            image_path,
            label_path,
            dst_images / f"{case_id}_0000.nii.gz",
            dst_labels / f"{case_id}.nii.gz",
            args.body_threshold,
            args.margin_xy,
            args.margin_z,
            args.crop_mode,
        )

        reports.append(report)
        print(f"[{i}/{len(image_files)}] {case_id}: {report['cropped_shape']}")

    with open(src / "dataset.json", "r") as f:
        dataset_json = json.load(f)

    dataset_json["channel_names"] = {"0": "CT"}
    dataset_json["numTraining"] = len(image_files)

    with open(dst / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    selected_cases = src / "selected_cases.txt"
    if selected_cases.exists():
        shutil.copy2(selected_cases, dst / "selected_cases.txt")

    with open(dst / "preprocessing_report.json", "w") as f:
        json.dump({
            "pipeline_name": "ImageTBADBodyCropCT",
            "n_cases": len(image_files),
            "reports": reports
        }, f, indent=2)

    print(f"Done. Cropped dataset saved to {dst}")


if __name__ == "__main__":
    main()