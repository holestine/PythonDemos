
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["bird", "cat"],
    max_samples=20,
    overwrite=True
)

files = dataset.values("filepath")

print("done")
