import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader

class UserItemRatingDataset(Dataset):
    def __init__(self, ratings_file, num_negative_train=4, num_negative_test=99):
        self.ratings_frame = pd.read_csv(ratings_file)
        self.num_negative_train = num_negative_train
        self.num_negative_test = num_negative_test
        
        # get unique userIds/movieIds and their counts
        self.user_ids = self.ratings_frame["userId"].unique()
        self.item_ids = self.ratings_frame["movieId"].unique()
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
    
        # create userId, movieId encoders to remove unused ids and convert to 0-index
        self.user_encoder = {x: i for i, x in enumerate(self.user_ids)}
        self.item_encoder = {x: i for i, x in enumerate(self.item_ids)}
        self.user_decoder = {i: x for x, i in self.user_encoder.items()}
        self.item_decoder = {i: x for x, i in self.item_encoder.items()}

        # remap using encodings
        self.ratings_frame["userId"] = self.ratings_frame["userId"].map(self.user_encoder)
        self.ratings_frame["movieId"] = self.ratings_frame["movieId"].map(self.item_encoder)
        
        # split into train and test sets
        print("Performing train/test split...")
        self.train_ratings, self.test_ratings = self._split_train_test()
        
        # extract train and test data
        print("Extracting train/test data...")
        self.train_user_item_dict = self._extract_user_item_dict(self.train_ratings)
        self.test_user_item_dict = self._extract_user_item_dict(self.test_ratings)
        
        print("Preparing training/testing data...")
        self.train_data = self._prepare_train_data()
        self.test_data = self._prepare_test_data()

    def _split_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each user, reserve the most recent rating by timestamp for the test set, leaving
        all other samples for training.
        """
        # get index of last timestamp per user
        last_timestamp_idx = self.ratings_frame.groupby("userId")["timestamp"].idxmax()

        # split based on whether set contains last timestamp or not
        train_ratings = self.ratings_frame[~self.ratings_frame.index.isin(last_timestamp_idx)]
        test_ratings = self.ratings_frame[self.ratings_frame.index.isin(last_timestamp_idx)]
        return train_ratings, test_ratings

    def _extract_user_item_dict(self, ratings) -> dict[int: list[int]]:
        """
        Return dict of user:movieId of rated movies: {userId: list[movieId1, ... , movieIdn]}
        """
        return {user: items["movieId"].tolist() for user, items in ratings.groupby("userId")}

    def _prepare_train_data(self) -> list[tuple[int, int, int]]:
        """
        Create training set: list[(userId, movieId, interaction)]
        
        For each user rating, also num_negative_train (default: 4) negative samples (i.e., movies the
        user has not rated) for training.
        """
        train_data = []
        for user, pos_items in self.train_user_item_dict.items():
            if len(pos_items) > self.num_items // 5:
                # handle case where user has rated > 1/5 the datatset so 4 neg. samples/pos. cannot be generated
                neg_items = list(set(range(self.num_items)) - set(pos_items))
            else:
                neg_items = np.random.choice(
                    list(set(range(self.num_items)) - set(pos_items)),
                    size=self.num_negative_train*len(pos_items),
                    replace=False
                )
            train_data.extend([(user, pos_item, 1) for pos_item in pos_items])
            train_data.extend([(user, neg_item, 0) for neg_item in neg_items])
        return train_data

    def _prepare_test_data(self) -> list[tuple[int, int]]:
        """
        Create test set: list[(userId, list[movieId1, ... , movieId100])]
        
        For each user rating, also num_negative_test (default: 99) negative samples (i.e., movies the
        user has not rated) for training.
        """
        test_data = []
        for user, pos_items in self.test_user_item_dict.items():
            # there should only be one positive sample per user
            pos_item = pos_items[0]
            neg_items = np.random.choice(
                list(set(range(self.num_items)) - set(self.train_user_item_dict.get(user, [])) - {pos_item}),
                size=self.num_negative_test,
                replace=False
            )
            test_data.append((user, [pos_item] + neg_items.tolist()))
        return test_data

    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        user, item, label = self.train_data[idx]
        return user, item, label
    
    def save_encoders(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump({
                "user_encoder": self.user_encoder,
                "item_encoder": self.item_encoder,
                "user_decoder": self.user_decoder,
                "item_decoder": self.item_decoder
            }, f)

    def get_test_data(self):
        return self.test_data

def prepare_data(ratings_file, batch_size=256, num_negatives=4, encoding_path="./encodings/encoding-latest.pkl"):
    dataset = UserItemRatingDataset(ratings_file, num_negatives)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset.save_encoders(encoding_path)
    return train_loader, dataset.get_test_data(), dataset.num_users, dataset.num_items
