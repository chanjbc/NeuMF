import argparse
from NeuMF import NeuMF
import torch
from datetime import datetime
from Download import download_and_extract_dataset
from PrepareData import prepare_data
from TrainEvaluate import train_and_evaluate

def main(dataset):
    print("Downloading dataset...")
    download_and_extract_dataset(dataset)

    print("Preparing data...")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    encoding_path = f"./encodings/encoding-{dataset}-{current_time}.pkl"
    train_loader, test_data, num_users, num_items = prepare_data(f"./data/{dataset}/ratings.csv", num_negatives=4, encoding_path=encoding_path)
    print(f"Encodings saved as: {encoding_path}")

    mf_dim = 8
    mlp_layers = [32, 16, 8]

    print("Creating model...")
    model = NeuMF(num_users, num_items, mf_dim, mlp_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Training and evaluating model...")
    model_path = f"./models/model-{dataset}-{current_time}.pth"
    train_and_evaluate(model, train_loader, test_data, num_epochs=20, learning_rate=0.001, device=device, model_path=model_path)
    print(f"Training and evaluation completed. Final model saved as: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuMF model on MovieLens dataset")
    parser.add_argument("dataset", choices=["ml-32m", "ml-latest-small"], help="Dataset to use (ml-32m or ml-latest-small)")
    args = parser.parse_args()
    main(args.dataset)