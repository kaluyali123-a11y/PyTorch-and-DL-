import torch
import random
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split


# ========== REPRODUCIBILITY ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Read our data ==========
train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor(), target_transform=None
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)

# ========== SPLIT TRAIN-DATA TO TRAIN AND VALID ==========
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
train_subset, valid_subset = random_split(train_data, [train_size, valid_size])

# ========== MAKE DATA ITERABLE ==========
BATCH_SIZE = 32
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


class FashionMNISTMODELV0(nn.Module):
    def __init__(self, input=784, hidden_sizes=(256, 128), output=10):
        super().__init__()
        layers = [nn.Flatten()]
        prev_size = input
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# ========== Initialize our model object, loss & optimizer ==========
model = FashionMNISTMODELV0().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# ========== TRAINING LOOP ==========
def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()  # <--Set model to Training mode

    n_correct = 0
    total_loss = 0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_logits = model(X_batch)
        predictions = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        n_correct += (predictions == y_batch).sum().item()

    return total_loss / total_samples, (n_correct / total_samples) * 100


# ========== EVALUATION LOOP ==========
def eval_loop(model, dataloader, loss_fn):
    model.eval()  # <--Set model to Training mode

    n_correct = 0
    total_loss = 0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_logits = model(X_batch)
        predictions = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        n_correct += (predictions == y_batch).sum().item()

    return total_loss / total_samples, (n_correct / total_samples) * 100

# ========== RUN TRAINING ==========
EPOCHS = 20
PATIENCE = 5
best_valid = float("inf")
patience_counter = 0
history = {"train_acc": [], "train_loss": [], "valid_acc": [], "valid_loss": []}

print(f"            🎇STARTED TRAINING MODEL!!\n")
for epoch in tqdm(range(EPOCHS), desc="TRAINING PROGRESS"):
    train_loss, train_acc = train_loop(model, train_loader, loss_fn, optimizer)
    valid_loss, valid_acc = eval_loop(model, valid_loader, loss_fn)

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    if epoch % 1 == 0:
        print(
            f"\n               Epoch: {epoch}\nTrain-Loss: {train_loss:.4f} | Train-ACC: {train_acc:.2f}%\nValid-Loss: {valid_loss:.4f} | Valid-ACC: {valid_acc:.2f}%"
        )

    if valid_loss < best_valid:
        best_valid = valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), "FASHIONMNIST Classifier.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏺ STOPPING EARLY!!\n")
            break
