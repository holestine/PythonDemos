
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "sports ball"],
    max_samples=200,
    overwrite=True
)

files = dataset.values("filepath")

print("done")
