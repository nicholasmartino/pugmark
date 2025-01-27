import glob
import os

from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import login


def create_hf_dataset(data_dir):
    # Get all image paths
    train_images = glob.glob(os.path.join(data_dir, "train/*.jpg"))
    test_images = glob.glob(os.path.join(data_dir, "test/*.jpg"))

    # Create datasets for train and test
    def create_dataset_dict(image_paths):
        return {
            "image_path": image_paths,
            "image": [Image().encode_example(img_path) for img_path in image_paths],
        }

    # Create HuggingFace datasets
    train_dataset = Dataset.from_dict(create_dataset_dict(train_images))
    test_dataset = Dataset.from_dict(create_dataset_dict(test_images))

    # Create DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset_dict


if __name__ == "__main__":
    # Login to Hugging Face
    login()  # You'll need your HF token here

    # Create dataset
    dataset = create_hf_dataset("path/to/your/local/dataset")

    # Push to hub
    dataset.push_to_hub(
        "your-username/your-dataset-name",
        private=True,  # Set to False if you want it public
    )
