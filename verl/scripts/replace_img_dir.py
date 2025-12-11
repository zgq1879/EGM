import os
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--base_img_root", type=str, required=True)
    parser.add_argument("--images_key", type=str, default="images")
    return parser.parse_args()


def relative_from_coco(path: str) -> str:
    keys = ("coco_flip", "coco")
    for key in keys:
        idx = path.rfind(key)
        if idx != -1:
            return path[idx:]
    return os.path.basename(path)


def main():
    args = parse_args()

    dataset_dict = load_dataset("parquet", data_files=args.parquet_path)
    ds = dataset_dict["train"]

    def update_images(example):
        paths = example[args.images_key]
        new_paths = []
        for p in paths:
            rel = relative_from_coco(p)
            new_p = os.path.join(args.base_img_root, rel)
            new_paths.append(new_p)
        example[args.images_key] = new_paths
        return example

    ds = ds.map(update_images)

    tmp_path = args.parquet_path + ".tmp"
    ds.to_parquet(tmp_path)
    os.replace(tmp_path, args.parquet_path)


if __name__ == "__main__":
    main()
