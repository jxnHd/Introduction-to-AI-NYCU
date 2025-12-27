import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Import from our custom modules
from utils import load_data
from model import GraphSAGE

# Configuration
STUDENT_ID = "111511281"  # Updated Student ID
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.005
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 1024  # Updated Hidden Dimension
EPOCHS = 300
PATIENCE = 30  # For early stopping

def train(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    acc = correct / mask.sum().item()
    return acc, pred

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    data = load_data()
    data = data.to(DEVICE)
    
    # 2. Create Train/Validation Split from the provided training data
    # We manually split the indices where train_mask is True
    all_train_indices = torch.nonzero(data.train_mask).squeeze().cpu().numpy()
    all_train_labels = data.y[all_train_indices].cpu().numpy()
    
    # Stratified split 90/10
    train_idx, val_idx = train_test_split(
        all_train_indices,
        test_size=0.1,
        stratify=all_train_labels,
        random_state=42
    )
    
    # Update masks for actual training
    # We create new mask tensors
    actual_train_mask = torch.zeros_like(data.train_mask)
    actual_train_mask[train_idx] = True
    
    val_mask = torch.zeros_like(data.train_mask)
    val_mask[val_idx] = True
    
    # 3. Initialize Model
    model = GraphSAGE(
        in_channels=data.num_features,
        hidden_channels=HIDDEN_DIM,
        out_channels=data.num_classes,
        num_layers=3,
        dropout=0.5
    ).to(DEVICE)
    
    print(model)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.0001)

    # 4. Training Loop
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, data, optimizer, actual_train_mask)
        train_acc, _ = test(model, data, actual_train_mask)
        val_acc, _ = test(model, data, val_mask)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save best model temporarily
            torch.save(model, f"{STUDENT_ID}.pth")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # 5. Generate Submission
    print("Generating submission...")
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Predict on Test Data
    _, preds = test(model, data, data.test_mask)
    
    submission_rows = []
    # Get all indices where test_mask is True
    test_indices = torch.nonzero(data.test_mask).squeeze().cpu().numpy()
    
    for idx in test_indices:
        original_id = data.id_map_reverse[idx]
        predicted_label = preds[idx].item()
        submission_rows.append([original_id, predicted_label])
        
    submission_df = pd.DataFrame(submission_rows, columns=['node_id', 'label'])
    
    csv_filename = f"{STUDENT_ID}.csv"
    submission_df.to_csv(csv_filename, index=False)
    print(f"Submission saved to {csv_filename}")
    
    # Save the final full model object as requested
    torch.save(model, f"{STUDENT_ID}.pth")
    print(f"Model saved to {STUDENT_ID}.pth")

if __name__ == "__main__":
    main()
