import csv
import glob
import os
from typing import List, Tuple

from PIL import Image
from photutils.aperture import BoundingBox
import json


DISTANCE_DANGEROUS_MAPPING = {
    1: "low",
    2: "low",
    3: "low",
    4: "low",
    5: "medium",
    6: "low",
    7: "medium",
    8: "high",
    9: "medium",
}
COLUMN_NAMES = [
    "label",
    "truncated",
    "occluded",
    "alpha",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "dim_height",
    "dim_width",
    "dim_length",
    "loc_x",
    "loc_y",
    "loc_z",
    "rotation_y",
    # "score",
]

LABEL_SORT_VARIABLE = {"high": 1, "medium": 2, "low": 3}
# IMG_PATH, LABEL_PATH, LABEL_OUTPUT_PATH = "data", "data", "data/new_label"
# Configuración para google drive
ROOT_PATH = (
    "drive/MyDrive/notebooks_python/tfm/datos/"
)
IMG_PATH = os.path.join(ROOT_PATH, "data_object_image_2/training/images")
LABEL_PATH = os.path.join(ROOT_PATH, "exported_data/training/label_2")
LABEL_OUTPUT_PATH = os.path.join(
    ROOT_PATH, "data_object_label_2/yolo_multilabel/training/yolov5_multilabels_4_groups/"
)

LABEL_MAPPING = dict()


def get_bbox_given_an_image_path(image_path: str) -> Tuple[Image.Image, List[Tuple]]:
    img = Image.open(image_path)
    img_width, img_height = img.size
    height = int(img_height / 3)
    width = int(img_width / 3)
    bbox_list_for_intersection = list()

    num_start = 1
    i = 0
    for x in range(0, 3, 1):
        j = 0
        for num, y in enumerate(range(0, 3, 1), start=num_start):
            bbox_list_for_intersection.append(
                (j, j + width, i, i + height, DISTANCE_DANGEROUS_MAPPING.get(num))
                #  x min, x max, y min, y max
            )
            j += width

        i += height
        num_start += 3

    return img, bbox_list_for_intersection


def get_image_intersections(label_file_name_path):
    with open(label_file_name_path, "r") as f:
        reader = csv.DictReader(f=f, fieldnames=COLUMN_NAMES, delimiter=" ")
        return list(row for row in reader)


def detect_object_intersection(
        bbox_object_detected: Tuple, bbox_list_base_image: List, debug: bool = False
):
    for i, bbox_base_image in enumerate(
            sorted(bbox_list_base_image, key=lambda x: LABEL_SORT_VARIABLE[x[-1]]), start=1
    ):
        *bbox_intersection, label = bbox_base_image
        base_bbox = BoundingBox(*bbox_intersection)

        bbox_intersection = base_bbox.intersection(
            BoundingBox(*map(lambda x: int(float(x)), bbox_object_detected))
        )
        if bbox_intersection:
            if debug:
                print(
                    f"Found intersection in {i} for bbox {bbox_intersection} with label {label}"
                )
            return label


def manage_label_mapping(label):
    if label not in LABEL_MAPPING:
        if not LABEL_MAPPING:
            LABEL_MAPPING[label] = 0
        else:
            LABEL_MAPPING[label] = max(LABEL_MAPPING.values()) + 1


def generate_new_label_file_with_new_label(object_identifier):
    final_labels = [
        'Car',
        'Others',
        'Person',
        'Big Vehicle',
        'high',
        'low',
        'medium'
    ]
    new_label_mapping = {
        "Truck": "Big Vehicle",
        "Van": "Big Vehicle",
        "Tram": "Big Vehicle",
        "Pedestrian": "Person",
        "Person_sitting": "Person",
        "Cyclist": "Person",
        "Misc": "Others",
        "DontCare": "Others"
    }

    image_path = os.path.join(IMG_PATH, object_identifier + ".png")
    label_path = os.path.join(LABEL_PATH, object_identifier + ".txt")
    label_output_path = os.path.join(LABEL_OUTPUT_PATH, object_identifier + ".txt")
    img, bbox_list_base_image = get_bbox_given_an_image_path(image_path)
    entries_file = get_image_intersections(label_path)

    img_width, img_height = img.size

    with open(label_output_path, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        for row in entries_file:
            bbox_width = (float(row["bbox_xmax"]) - float(row["bbox_xmin"])) / img_width
            bbox_height = (float(row["bbox_ymax"]) - float(row["bbox_ymin"])) / img_height
            x_center = (float(row["bbox_xmax"]) + float(row["bbox_xmin"])) / (2 * img_width)
            y_center = (float(row["bbox_ymax"]) + float(row["bbox_ymin"])) / (2 * img_height)

            # x min, x max, y min, y max
            if bbox_list_base_image:
                dangerous_level = detect_object_intersection(
                    (
                        row["bbox_xmin"],
                        row["bbox_xmax"],
                        row["bbox_ymin"],
                        row["bbox_ymax"],
                    ),
                    bbox_list_base_image,
                )
            else:
                print(f"Not dangerous level detected, object: {object_identifier}.")
                dangerous_level = None

            # escribir aquí ambos label
            label = row["label"]
            if label in new_label_mapping:
                label = new_label_mapping[label]

            if label not in final_labels:
                continue  # we do not save labels we do not want

            manage_label_mapping(label)
            tmp_output = [
                LABEL_MAPPING[label],
                x_center,
                y_center,
                bbox_width,
                bbox_height
            ]
            writer.writerow(tmp_output)

            if dangerous_level:
                manage_label_mapping(dangerous_level)
                tmp_output = [
                    LABEL_MAPPING[dangerous_level],
                    x_center,
                    y_center,
                    bbox_width,
                    bbox_height
                ]
                # row.update({"label": row["label"] + "-" + dangerous_level})
                writer.writerow(tmp_output)


def generate_label_of_all_images():
    print("Starting to generate files.")
    identifiers = list(
        map(
            lambda x: os.path.splitext(os.path.split(x)[-1])[0],
            glob.glob(os.path.join(IMG_PATH, "*.png")),
        )
    )
    total_ids = len(identifiers)
    print(f"Starting to proceed {total_ids} identifiers")
    count = 0
    for i, identifier in enumerate(identifiers, start=1):
        if i == 1:
            print("Generating 1.")
        if i % 100 == 0:
            count += 100
            print(f"Generating {count} labels")
        generate_new_label_file_with_new_label(identifier)

    label_file_path = os.path.join(ROOT_PATH, "label_mapping_4_groups.json")
    print(f"Writing label mapping in : {label_file_path}")
    with open(label_file_path, "w") as f:
        json.dump(LABEL_MAPPING, f)


if __name__ == '__main__':
    generate_label_of_all_images()
