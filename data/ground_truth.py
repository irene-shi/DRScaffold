"""Prepare aligned semantic overlays and class/layout context for visual review."""

import argparse
import json
from pathlib import Path

from data.io import write_json
from data.schema import require

NYU40 = ("unknown", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
         "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
         "dresser", "pillow", "mirror", "floor_mat", "clothes", "ceiling", "books", "refrigerator",
         "television", "paper", "towel", "shower_curtain", "box", "whiteboard", "person", "night_stand",
         "toilet", "sink", "lamp", "bathtub", "bag", "otherstructure", "otherfurniture", "otherprop")
# IDs from Cityscapes labelIds, NOT trainIds. Only the 19 evaluation classes.
CITY19 = {7: "road", 8: "sidewalk", 11: "building", 12: "wall", 13: "fence", 17: "pole",
          19: "traffic_light", 20: "traffic_sign", 21: "vegetation", 22: "terrain", 23: "sky",
          24: "person", 25: "rider", 26: "car", 27: "truck", 28: "bus", 31: "train",
          32: "motorcycle", 33: "bicycle"}


def read_hdf(path):
    import h5py
    with h5py.File(path, "r") as source:
        require("dataset" in source, "Expected the Hypersim HDF5 dataset key")
        return source["dataset"][:]


def class_layout(labels, names):
    import numpy as np
    result = []
    height, width = labels.shape
    for value in np.unique(labels):
        if int(value) not in names:
            continue
        ys, xs = np.where(labels == value)
        result.append({"class_id": int(value), "class": names[int(value)], "pixels": int(len(xs)),
                       "fraction": float(len(xs) / labels.size),
                       "center_uv": [float((xs.mean() + .5) / width), float((ys.mean() + .5) / height)],
                       "bbox_xyxy": [float(xs.min() / width), float(ys.min() / height),
                                     float((xs.max() + 1) / width), float((ys.max() + 1) / height)]})
    return sorted(result, key=lambda row: -row["pixels"])


def prepare_gt(domain, image, semantic, output, color=None, instances=None, positions=None):
    import numpy as np
    from PIL import Image
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "GT output directory must be empty")
    photograph = Image.open(image).convert("RGB")
    if domain == "hypersim":
        labels = read_hdf(semantic).astype(np.int32)
        names = {i: name for i, name in enumerate(NYU40) if i > 0}
    else:
        labels = np.asarray(Image.open(semantic))
        names = CITY19
        require(color is not None, "Cityscapes requires the matching gtFine_color.png")
    require(labels.ndim == 2, "Semantic labels must be a single-channel map")
    height, width = labels.shape
    require(photograph.width * height == photograph.height * width, "Image and GT aspect ratios differ; verify alignment")
    visible = class_layout(labels, names)
    require(bool(visible), "No recognized semantic classes; verify label encoding")
    mask = np.isin(labels, list(names))
    if domain == "cityscapes":
        colored = np.asarray(Image.open(color).convert("RGB"))
        require(colored.shape[:2] == labels.shape, "Cityscapes color and label map sizes differ")
    else:
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        for value in names:
            # Deterministic display colors, explicitly labeled in the context legend.
            rgb = ((value * 67) % 192 + 32, (value * 101) % 192 + 32, (value * 149) % 192 + 32)
            colored[labels == value] = rgb
        for entry in visible:
            entry["display_rgb"] = [int(v) for v in colored[labels == entry["class_id"]][0]]
    display = Image.fromarray(colored).resize(photograph.size, Image.Resampling.NEAREST)
    valid = Image.fromarray((mask * 255).astype(np.uint8)).resize(photograph.size, Image.Resampling.NEAREST)
    blend = Image.blend(photograph, display, .4)
    overlay = Image.composite(blend, photograph, valid)
    context = {"domain": domain, "image": str(Path(image).resolve()), "label_encoding": "NYU40" if domain == "hypersim" else "Cityscapes labelIds (19 classes)",
               "coordinates": "Normalized image coordinates: u/x increases rightward; v/y increases downward.",
               "limitations": "Classes can cover multiple instances. Their aggregate center/bbox is not an instance relation. Unlabeled pixels and absent class labels do not prove a small object/accessory is absent.",
               "classes": visible}
    require(not positions or instances, "Position maps require a matching instance map")
    if instances:
        require(domain == "hypersim", "HDF5 instance/position inputs are for Hypersim")
        instance_map = read_hdf(instances).astype(np.int32)
        require(instance_map.shape == labels.shape, "Instance and semantic maps must match")
        position_map = read_hdf(positions) if positions else None
        if position_map is not None:
            require(position_map.shape == (*labels.shape, 3), "Position map must be H x W x 3")
        objects = []
        for iid in np.unique(instance_map):
            if iid < 0:
                continue
            selected = instance_map == iid
            values, counts = np.unique(labels[selected], return_counts=True)
            cls = int(values[counts.argmax()])
            if cls not in names:
                continue
            ys, xs = np.where(selected)
            item = {"instance_id": int(iid), "class": names[cls], "pixels": int(selected.sum()),
                    "center_uv": [float((xs.mean() + .5) / width), float((ys.mean() + .5) / height)]}
            if position_map is not None:
                points = position_map[selected]
                points = points[np.isfinite(points).all(axis=1)]
                if len(points):
                    item["world_xyz_asset_units"] = np.median(points, axis=0).tolist()
            objects.append(item)
        context["instances"] = objects
    output.mkdir(parents=True, exist_ok=True)
    overlay.save(output / "overlay.png")
    write_json(output / "context.json", context)
    return {"gt_image": str((output / "overlay.png").resolve()), "gt_context": str((output / "context.json").resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("hypersim", "cityscapes"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--color", type=Path, help="Cityscapes gtFine_color.png")
    parser.add_argument("--instances", type=Path, help="Optional Hypersim semantic_instance.hdf5")
    parser.add_argument("--positions", type=Path, help="Optional Hypersim position.hdf5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare_gt(**vars(args)), indent=2))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
