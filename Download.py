import argparse
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

def download_and_extract_dataset(dataset_name):
    datasets = {
        "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
        "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    }

    if dataset_name not in datasets:
        raise ValueError(f"Invalid dataset name. Choose either ml-32m or ml-latest-small.")

    url = datasets[dataset_name]
    data_folder = Path("./data")
    data_folder.mkdir(exist_ok=True)
    dataset_folder = data_folder / dataset_name

    file_path = data_folder / f"{dataset_name}.zip"

    if not dataset_folder.exists() or not any(dataset_folder.iterdir()):
        print(f"Downloading MovieLens {dataset_name} dataset to: {file_path}")
        urlretrieve(url, str(file_path))

        print("Extracting dataset...")
        with zipfile.ZipFile(str(file_path), "r") as zip_ref:
            zip_ref.extractall(str(data_folder))
    else:
        print("Dataset already present")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract MovieLens dataset")
    parser.add_argument("dataset", choices=["ml-32m", "ml-latest-small"], help="Dataset to download (ml-32m or ml-latest-small)")
    args = parser.parse_args()
    download_and_extract_dataset(args.dataset)