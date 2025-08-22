import os
import glob
import argparse
import shutil
from tqdm import tqdm
from ultralytics import YOLO

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def get_images(input_dir):
    """Recursively collect all image file paths from a directory."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
    return files


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference and Positive Image Filter")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model")
    parser.add_argument("--input", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for positive images")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--objects", type=str, required=True,
                        help="Comma-separated list of desired objects (class names)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()

    # Parse desired objects
    desired_objects = [obj.strip() for obj in args.objects.split(",") if obj.strip()]
    if not desired_objects:
        raise ValueError("At least one object must be specified via --objects")

    # Prepare model
    model = YOLO(args.model)

    # Collect image paths
    image_paths = get_images(args.input)
    if not image_paths:
        print("No compatible images found in input directory.")
        return

    os.makedirs(args.output, exist_ok=True)

    # Run inference in batches
    for i in tqdm(range(0, len(image_paths), args.batch), desc="Processing batches"):
        batch_paths = image_paths[i:i + args.batch]

        results = model.predict(batch_paths, conf=args.threshold, verbose=False)

        for path, result in zip(batch_paths, results):
            # Get predicted class names
            names = [result.names[int(c)] for c in result.boxes.cls]

            # Check if at least one desired object is present
            if any(name in desired_objects for name in names):
                shutil.copy(path, args.output)


if __name__ == "__main__":
    main()
