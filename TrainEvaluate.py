import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_and_evaluate(model, train_loader, test_data, num_epochs, learning_rate, device, model_path):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_hr = 0
    best_ndcg = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user, item, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user = user.to(device)
            item = item.to(device)
            label = label.to(device).float()

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate the model
        hr, ndcg = evaluate_model(model, test_data, device)
        print(f"HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}\n")

        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            torch.save(model.state_dict(), model_path)

    print(f"Best HR@10: {best_hr:.4f}, Best NDCG@10: {best_ndcg:.4f}")
    return model

def evaluate_model(model, test_data, device, k=10):
    model.eval()
    hits = []
    ndcgs = []

    with torch.no_grad():
        for user, items in tqdm(test_data, desc="Evaluating"):
            user_tensor = torch.LongTensor([user] * len(items)).to(device)
            items_tensor = torch.LongTensor(items).to(device)
            
            predictions = model(user_tensor, items_tensor)
            
            _, indices = torch.topk(predictions, k)
            recommends = torch.take(items_tensor, indices).cpu().numpy().tolist()
            
            gt_item = items[0]
            
            hr = get_hit_ratio(recommends, gt_item)
            ndcg = get_ndcg(recommends, gt_item)
            
            hits.append(hr)
            ndcgs.append(ndcg)

    return np.mean(hits), np.mean(ndcgs)

def get_hit_ratio(ranklist, gt_item):
    return 1 if gt_item in ranklist else 0

def get_ndcg(ranklist, gt_item):
    for i, item in enumerate(ranklist):
        if item == gt_item:
            return np.log(2) / np.log(i+2)
    return 0