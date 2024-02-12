import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    print(
        "`segment_anything` not found. Please install first: `pip install git+https://github.com/facebookresearch/segment-anything.git`"
    )

DEFAULT_CKPT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
DEFAULT_CKPT_PATH = "~/.cache/torch/hub/sam_vit_h_4b8939.pth"
DEFAULT_CKPT_PATH = os.path.expanduser(DEFAULT_CKPT_PATH)


def annotate(image_path, output_path, ckpt_path=DEFAULT_CKPT_PATH):
    window_name = "annotation"
    cv2.namedWindow(window_name)

    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image = np.array(image_pil)
    cv2.imshow(window_name, image[..., ::-1])
    print("Press any key to continue...")
    cv2.waitKey(0)

    # Initialize SAM
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.exists(ckpt_path):
        if ckpt_path == DEFAULT_CKPT_PATH:
            os.makedirs(ckpt_path, exist_ok=True)
            torch.hub.download_url_to_file(DEFAULT_CKPT_URL, DEFAULT_CKPT_PATH)
        else:
            raise FileNotFoundError(ckpt_path)

    print("Loading SAM...")
    sam = sam_model_registry["default"](checkpoint=ckpt_path)
    print("Loaded SAM")
    predictor = SamPredictor(sam)
    print("Setting image...")
    predictor.set_image(np.array(image_pil))
    print("Image set")

    # Set callback
    def on_click(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            point_coords.append((x, y))
            point_labels.append(1)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, image[..., ::-1])
        elif event == cv2.EVENT_MBUTTONDOWN:
            point_coords.append((x, y))
            point_labels.append(0)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow(window_name, image[..., ::-1])

    cv2.setMouseCallback(window_name, on_click)

    while True:
        # Reset variables
        image = np.array(image_pil)
        point_coords = []
        point_labels = []

        # Annotate positive and negative points
        cv2.imshow(window_name, image[..., ::-1])
        print("Left click to add positive point, right click to add negative point")
        print("Press any key to stop annotation.")
        cv2.waitKey(0)

        print(point_coords, point_labels)
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        if len(point_coords) == 0:
            print("No points annotated. Skipping.")
            continue

        # Inference
        logits = None
        MAX_ITERS = 8
        for _ in range(MAX_ITERS):
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logits,
                multimask_output=False,
                return_logits=True,
            )
        mask = masks[scores.argmax()] > 0  # [H, W]
        cv2.imshow(window_name, np.uint8(mask * 255))
        print("Press ESC to finish annotation. Otherwise reset annotation.")
        key = cv2.waitKey(0)
        if key == 27:  # escape
            break

    masked_image = np.array(image_pil) * np.uint8(mask)[..., None]
    cv2.imshow(window_name, masked_image[..., ::-1])
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path is not None:
        Image.fromarray(mask).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        required=True,
        help="input image path to annotate",
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="output path to save annotated mask"
    )
    args = parser.parse_args()
    annotate(args.image_path, args.output_path)


if __name__ == "__main__":
    main()
