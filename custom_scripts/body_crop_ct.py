import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

#SIDE NOTE FOR MYSELF
#could the reason there is no crop be taht the mask is entering the if statement where it is empty and ust returns the 
#original image? 
#need to check this out after the 2d experiment (if unsuccessful)


def load_nifti(path):
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    return nii, data


def save_nifti(data, affine, header, path):
    out = nib.Nifti1Image(data, affine, header=header)
    nib.save(out, str(path))
'''
 DEMO PROVED THIS DOESNT WORK SO REMOVING FOR NOW - MAYBE REVISIT LATER
def keep_largest_component(mask, min_size=10000):
    #labeled_mask, num_components = ndi.label(mask)

    if num_components == 0:
        return mask

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_label = np.argmax(sizes)

    if sizes[largest_label] < min_size:
        return mask

    return labeled_mask == largest_label
'''
# new approach 
#remove anything touching the border as this shouldnt be the patient and taht way i can gaurentee that the largest component is the patient body
def remove_border_components_2d(mask2d):
    labeled, num = ndi.label(mask2d)
    if num == 0:
        return mask2d

    border_labels = set()

    # collect labels touching the border
    border_labels.update(np.unique(labeled[0, :]))
    border_labels.update(np.unique(labeled[-1, :]))
    border_labels.update(np.unique(labeled[:, 0]))
    border_labels.update(np.unique(labeled[:, -1]))

    cleaned = labeled.copy()
    for lab in border_labels:
        cleaned[cleaned == lab] = 0

    return cleaned > 0

#new 2d helper
def largest_component_2d(mask2d, min_pixels=5000):
    labeled, num = ndi.label(mask2d)
    if num == 0:
        return mask2d

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = np.argmax(sizes)

    if sizes[largest] < min_pixels:
        return np.zeros_like(mask2d, dtype=bool)

    return labeled == largest

'''
remove too as im doing it with 2d slices now 

def create_body_mask(image, threshold, closing_iters=1, opening_iters=0, min_size=10000):
    mask = image > threshold

    if closing_iters > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iters)

    if opening_iters > 0:
        mask = ndi.binary_opening(mask, iterations=opening_iters)

    mask = ndi.binary_fill_holes(mask)
    mask = keep_largest_component(mask, min_size)

    return mask
'''
#new 2d version 
def get_body_mask(image, threshold, min_size=10000, debug=False):
    """
    image shape expected: (X, Y, Z)
    returns a 3D body mask, built slice-by-slice
    """
    body_mask = np.zeros_like(image, dtype=bool)

    for z in range(image.shape[2]):
        sl = image[:, :, z]

        # threshold this 2D slice
        mask = sl > threshold
        threshold_sum = int(mask.sum())
        # fill small gaps inside the body region
        mask = ndi.binary_fill_holes(mask)
        fill_sum = int(mask.sum())
        # remove anything touching the slice border
        mask = remove_border_components_2d(mask)
        border_removed_sum = int(mask.sum())
        # keep the largest remaining structure
        mask = largest_component_2d(mask, min_pixels=5000)
        largest_sum = int(mask.sum())
        # light smoothing
        mask = ndi.binary_closing(mask, iterations=1)
        closed_sum = int(mask.sum())


        body_mask[:, :, z] = mask
        if debug and z in [0, image.shape[2]//4, image.shape[2]//2, 3*image.shape[2]//4, image.shape[2]-1]:
            print(
                f"Slice {z}: "
                f"threshold={threshold_sum}, "
                f"fill={fill_sum}, "
                f"border_removed={border_removed_sum}, "
                f"largest={largest_sum}, "
                f"closed={closed_sum}"
            )
    # 3D cleanup after stacking slices
    before_3d_fill = int(body_mask.sum())
    body_mask = ndi.binary_fill_holes(body_mask)
    after_3d_fill = int(body_mask.sum())

    if debug:
        print(f"3D fill holes: before={before_3d_fill}, after={after_3d_fill}")

    return body_mask


def get_bbox(mask):
    coords = np.where(mask)

    if coords[0].size == 0:
        return None

    x0, y0, z0 = [int(c.min()) for c in coords]
    x1, y1, z1 = [int(c.max()) + 1 for c in coords]

    return x0, x1, y0, y1, z0, z1


def expand_bbox(bbox, shape, margin_xy, margin_z, crop_mode):
    x0, x1, y0, y1, z0, z1 = bbox

    x0 = max(0, x0 - margin_xy)
    x1 = min(shape[0], x1 + margin_xy)
    y0 = max(0, y0 - margin_xy)
    y1 = min(shape[1], y1 + margin_xy)

    if crop_mode == "xy_only":
        z0, z1 = 0, shape[2]
    else:
        z0 = max(0, z0 - margin_z)
        z1 = min(shape[2], z1 + margin_z)

    return (slice(x0, x1), slice(y0, y1), slice(z0, z1))


def cropped_affine(affine, crop_slices):
    start = np.array([
        crop_slices[0].start,
        crop_slices[1].start,
        crop_slices[2].start,
        1.0
    ])
    new_affine = affine.copy()
    new_affine[:3, 3] = (affine @ start)[:3]
    return new_affine


def process_case(img_path, lab_path, out_img_path, out_lab_path, threshold, margin_xy, margin_z, crop_mode):
    img_nii, img = load_nifti(img_path)
    lab_nii, lab = load_nifti(lab_path)

    if img.ndim != 3:
        raise ValueError(f"Image is not 3D: {img_path}")
    if lab.ndim != 3:
        raise ValueError(f"Label is not 3D: {lab_path}")

    original_shape = list(img.shape)

    body_mask = get_body_mask(img, threshold, debug=True)
    print("Body mask true voxels:", int(body_mask.sum()))
    bbox = get_bbox(body_mask)
    print("Raw bbox:", bbox)

    if bbox is None:
        crop = (slice(0, img.shape[0]), slice(0, img.shape[1]), slice(0, img.shape[2]))
    else:
        crop = expand_bbox(bbox, img.shape, margin_xy, margin_z, crop_mode)

    cropped_img = img[crop].astype(np.float32)
    cropped_lab = np.rint(lab[crop]).astype(np.uint8)

    img_header = img_nii.header.copy()
    lab_header = lab_nii.header.copy()
    img_header.set_data_dtype(np.float32)
    lab_header.set_data_dtype(np.uint8)

    save_nifti(cropped_img, cropped_affine(img_nii.affine, crop), img_header, out_img_path)
    save_nifti(cropped_lab, cropped_affine(lab_nii.affine, crop), lab_header, out_lab_path)

    return {
        "case": img_path.name.replace("_0000.nii.gz", ""),
        "original_shape": original_shape,
        "cropped_shape": list(cropped_img.shape),
        "crop_bbox": {
            "x": [crop[0].start, crop[0].stop],
            "y": [crop[1].start, crop[1].stop],
            "z": [crop[2].start, crop[2].stop],
        },
        "label_voxels_after_crop": int(np.sum(cropped_lab > 0)),
        "image_stats_after_crop": {
            "min": float(cropped_img.min()),
            "max": float(cropped_img.max()),
            "mean": float(cropped_img.mean()),
            "std": float(cropped_img.std()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Crop CT scans to body region for nnU-Net")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)

    # OLD default threshold:
    # parser.add_argument("--body-threshold", type=float, default=-650.0)

    # NEW  default threshold:
    parser.add_argument("--body-threshold", type=float, default= 50) 

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

    for i, img_path in enumerate(image_files, 1):
        case = img_path.name.replace("_0000.nii.gz", "")
        lab_path = src_labels / f"{case}.nii.gz"

        if not lab_path.exists():
            raise FileNotFoundError(f"Missing label for case {case}: {lab_path}")

        report = process_case(
            img_path,
            lab_path,
            dst_images / f"{case}_0000.nii.gz",
            dst_labels / f"{case}.nii.gz",
            args.body_threshold,
            args.margin_xy,
            args.margin_z,
            args.crop_mode,
        )

        reports.append(report)
        print(
            f"[{i}/{len(image_files)}] {case}: "
            f"original={report['original_shape']} "
            f"cropped={report['cropped_shape']} "
            f"bbox={report['crop_bbox']}"
        )

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