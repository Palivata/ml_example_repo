import argparse

import cv2
import numpy as np
from openvino import Core

from src.augmentations.augmentations import PadResizeOCR, TextEncode
from src.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_path", help="Path to image")
    parser.add_argument("--config_file", type=str, help="path to configuration file")
    args = parser.parse_args()
    config = Config.from_yaml(args.config_file)

    img = cv2.imread(args.img_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if config.experiment != "ocr":
        image = cv2.resize(image, (224, 224), cv2.INTER_CUBIC).astype(np.float32)
    else:
        x_min, y_min, x_max, y_max = (179, 516, 500, 703)
        image = image[y_min:y_max, x_min:x_max]
        image = PadResizeOCR(
            p=1, target_height=config.height, target_width=config.width, mode="left"
        )(image)["image"].astype(np.float32)
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]

    ie = Core()
    model = ie.read_model(f"models/{config.experiment}_model.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    output = compiled_model([image])[compiled_model.output(0)]

    if config.experiment == "segmentation":
        pr_mask = output
        pr_mask = cv2.resize(pr_mask, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
        pr_mask = (pr_mask > 0.5).astype(int)
        min_area = 100
        contours, _ = cv2.findContours(
            pr_mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area >= min_area:  # Filter out small contours
                bounding_boxes.append((x, y, x + w, y + h))
        for (x_min, y_min, x_max, y_max) in bounding_boxes:
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)
        cv2.imwrite("boxes.jpg", img)
        print(bounding_boxes)
    elif config.experiment == "ocr":
        enc = TextEncode(p=1, vocab="0123456789", target_text_size=13)
        code = "4810153026194"
        code = enc(image=None, text=code)
        character_set = ["blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        probs = np.exp(output) / np.sum(np.exp(output), axis=2, keepdims=True)
        predicted_indices = np.argmax(probs, axis=2)
        predicted_indices = predicted_indices.squeeze(1)
        decoded_sequence = []
        previous_char = None
        for index in predicted_indices:
            if index != 0:
                if index != previous_char:
                    decoded_sequence.append(character_set[index])
            previous_char = index
        decoded_text = "".join(decoded_sequence)
        print(decoded_text)
        print("4810153026194")
