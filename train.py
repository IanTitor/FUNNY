import torch as pt
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from grokfast_pytorch import GrokFastAdamW
from tqdm import tqdm
import h5py as hp
from model import Veganom
from dataset import StockDataset


# --- Collate Function for Equal-Length Sequences ---
def collate_fn(batch):
    """
    Collate a batch of samples. Each sample has shape (T, 6) with T=2048.
    This function splits each sample into:
      - state features: columns 0 to 4
      - positions: column 5
    It then shifts both state features and positions so that:
      - last_state is sample[:-1, :5]
      - target is sample[1:, :5]
      - current_pos is sample[:-1, 5]
      - future_pos is sample[1:, 5]
    
    Args:
        batch (list): List of tensors, each with shape (2048, 6).
    
    Returns:
        Tuple of:
            last_state: Tensor of shape (B, T-1, 5)
            current_pos: Tensor of shape (B, T-1)
            future_pos: Tensor of shape (B, T-1)
            target: Tensor of shape (B, T-1, 5)
    """
    # Stack batch along the new batch dimension.
    batch_tensor = pt.stack(batch)  # (B, 2048, 6)
    
    # Split state features and positions.
    state = batch_tensor[:, :, :5]   # (B, 2048, 5)
    pos = batch_tensor[:, :, 5]      # (B, 2048)
    
    # Shift sequences: last_state and current_pos are all but the last time step;
    # target and future_pos are all but the first time step.
    last_state = state[:, :-1, :]    # (B, 2047, 5)
    target = state[:, 1:, :]         # (B, 2047, 5)
    current_pos = pos[:, :-1]        # (B, 2047)
    future_pos = pos[:, 1:]          # (B, 2047)
    
    return last_state, current_pos, future_pos, target


if __name__ == '__main__':
    # Hyperparameters.
    data_path = "../data/dataset.h5"  # Update with your data location.
    model_save_path = "backup.ckpt"
    log_path = "log.h5"
    seq_len = 1024
    seq_stride = 1024
    batch_size = 4
    train_p = 0.8
    pt.manual_seed(42)
    num_epochs = 20

    # Model hyperparameters.
    # We set state_dim=6 because each sample has 6 features (and 6 is even for RoPE).
    state_dim = 5
    model_dim = 256
    num_layers = 4
    num_heads = 8
    hidden_dim = 512

    # Instantiate dataset and DataLoader.
    dataset = StockDataset(data_path, seq_len, seq_stride)

    train_size = int(train_p * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Instantiate model, loss function, and optimizer.
    model = Veganom(state_dim, model_dim, num_layers, num_heads, hidden_dim)
    loss_fn = nn.MSELoss()  # Adjust the loss if needed.
    #optimizer = GrokFastAdamW(model.parameters(), lr=1e-6, weight_decay=1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.99))


    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        pbar = tqdm(train_loader, desc=f"TE{epoch}...", unit="batch")
        for last_state, current_pos, future_pos, target in pbar:
            last_state = last_state.to(device)
            #print(last_state.shape, last_state.min(), last_state.max(), last_state.isnan().any())
            current_pos = current_pos.to(device)
            future_pos = future_pos.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass.
            output = model(last_state, current_pos, future_pos)  # (B, T, state_dim)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(L=f"{loss.item():.4f}")

            train_loss += loss.item()
            #input()
        model.eval()
        pbar = tqdm(val_loader, desc=f"VE{epoch}...", unit="batch")
        for last_state, current_pos, future_pos, target in pbar:
            last_state = last_state.to(device)
            #print(last_state.shape, last_state.min(), last_state.max(), last_state.isnan().any())
            current_pos = current_pos.to(device)
            future_pos = future_pos.to(device)
            target = target.to(device)

            # Forward pass.
            output = model(last_state, current_pos, future_pos)  # (B, T, state_dim)
            loss = loss_fn(output, target)
            pbar.set_postfix(L=f"{loss.item():.4f}")

            val_loss += loss.item()
            #input()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    pt.save(model.state_dict(), model_save_path)
    with hp.File(log_path, "w") as f:
        f.create_dataset("train_loss", data=pt.tensor(train_losses), chunks=None)
        f.create_dataset("val_loss", data=pt.tensor(val_losses), chunks=None)
