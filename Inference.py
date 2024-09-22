import argparse
import torch
import pandas as pd
from pathlib import Path
import pickle
from NeuMF import NeuMF

def load_model_and_encodings(model_path, encoding_path):
    # load encodings
    with open(encoding_path, "rb") as f:
        encodings = pickle.load(f)
    
    user_encoder = encodings["user_encoder"]
    item_encoder = encodings["item_encoder"]
    item_decoder = encodings["item_decoder"]
    
    # load model parameters
    model_state = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_users = len(user_encoder)
    num_items = len(item_encoder)
    
    # set model architecture parameters (can be adjusted)
    mf_dim = 8
    mlp_layers = [32, 16, 8]
    
    # create and load model
    model = NeuMF(num_users, num_items, mf_dim, mlp_layers)
    model.load_state_dict(model_state)
    model.eval()
    
    return model, user_encoder, item_encoder, item_decoder

def load_movies_data(movies_file):
    movies_df = pd.read_csv(movies_file)
    title_to_id = dict(zip(movies_df["title"], movies_df["movieId"]))
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    return title_to_id, id_to_title

def get_recommendations(model, user_movies, title_to_id, id_to_title, item_encoder, item_decoder, top_n=50):
    # convert user movies to item IDs
    user_item_ids = [item_encoder[title_to_id[title]] for title in user_movies if title in title_to_id and title_to_id[title] in item_encoder]
    
    if not user_item_ids:
        print("Warning: None of the provided movies were found in the dataset.")
        return []

    # get all item IDs
    all_item_ids = list(item_encoder.values())
    
    # compute average embeddings for user's watched movies
    with torch.no_grad():
        mf_avg_embedding = model.mf_item_embedding(torch.LongTensor(user_item_ids)).mean(dim=0)
        mlp_avg_embedding = model.mlp_item_embedding(torch.LongTensor(user_item_ids)).mean(dim=0)
    
    # create tensors for all items
    item_tensor = torch.LongTensor(all_item_ids)
    
    # repeat the average embeddings for each item
    mf_user_embedding = mf_avg_embedding.unsqueeze(0).repeat(len(all_item_ids), 1)
    mlp_user_embedding = mlp_avg_embedding.unsqueeze(0).repeat(len(all_item_ids), 1)
    
    # get predictions
    with torch.no_grad():
        mf_vector = torch.mul(mf_user_embedding, model.mf_item_embedding(item_tensor))
        mlp_vector = torch.cat([mlp_user_embedding, model.mlp_item_embedding(item_tensor)], dim=-1)
        for layer in model.mlp:
            mlp_vector = layer(mlp_vector)
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        predictions = model.sigmoid(model.output(predict_vector)).squeeze()
    
    # sort predictions
    _, indices = torch.topk(predictions, len(all_item_ids))
    recommended_items = [item_decoder[all_item_ids[idx.item()]] for idx in indices]
    
    # filter out movies the user has already interacted with
    new_recommendations = [item for item in recommended_items if item not in user_item_ids]
    
    # get top N recommendations
    top_recommendations = new_recommendations[:top_n]
    
    # convert back to movie titles
    return [id_to_title[item] for item in top_recommendations]


def main(args: argparse.Namespace) -> list[str]:
    top_n = int(args.top_n)
    encoding_path = Path(f"./encodings/{args.encoding_file}")
    model_path = Path(f"./models/{args.model_file}")
    movies_file = Path(f"./data/{args.dataset}/movies.csv")
    
    # load model, encodings, and movie data
    model, user_encoder, item_encoder, item_decoder = load_model_and_encodings(model_path, encoding_path)
    title_to_id, id_to_title = load_movies_data(movies_file)
    
    # example usage
    user_movies = [
        "How to Train Your Dragon (2010)",
        "How to Train Your Dragon 2 (2014)",
        "Lord of the Rings: The Fellowship of the Ring, The (2001)",
        "Sound of Music, The (1965)",
        "That Darn Cat (1997)",
        "Enchanted (2007)"
    ]
    
    recommendations = get_recommendations(model, user_movies, title_to_id, id_to_title, item_encoder, item_decoder, top_n=top_n)
    
    print("User's watched movies:")
    for movie in user_movies:
        print(f"- {movie}")
    
    print("\nRecommended movies:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

    return recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuMF model on MovieLens dataset")
    parser.add_argument("dataset", choices=["ml-32m", "ml-latest-small"], help="Dataset to use (ml-32m or ml-latest-small)")
    parser.add_argument("top_n", type=int, help="Number of recommendations to return")
    parser.add_argument("encoding_file", type=str, help=".pkl encoding file name")
    parser.add_argument("model_file", type=str, help=".pth model file name")
    args = parser.parse_args()
    main(args)