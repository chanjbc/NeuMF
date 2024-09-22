import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, mlp_layers, dropout=0.1):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_layers = mlp_layers

        # MF embeddings
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)

        # MLP embeddings
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_layers[0]//2)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_layers[0]//2)

        # MLP layers
        self.mlp = nn.ModuleList()
        for i in range(1, len(mlp_layers)):
            self.mlp.append(nn.Linear(mlp_layers[i-1], mlp_layers[i]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))

        # output layer
        self.output = nn.Linear(mf_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # MF part
        mf_user_latent = self.mf_user_embedding(user_input)
        mf_item_latent = self.mf_item_embedding(item_input)
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)

        # MLP part
        mlp_user_latent = self.mlp_user_embedding(user_input)
        mlp_item_latent = self.mlp_item_embedding(item_input)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)

        for layer in self.mlp:
            mlp_vector = layer(mlp_vector)

        # concatenate MF and MLP parts
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        # final prediction
        prediction = self.sigmoid(self.output(predict_vector))

        return prediction.view(-1)