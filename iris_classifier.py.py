import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# =========================
# Ensure reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)  # Python's built-in random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU

    # THESE ARE CRITICAL FOR REPRODUCIBILITY:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()
# ====================================
# DATA READING AND CLEANING
# ====================================
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("iris.data", sep=",", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df["class"] = df["class"].map(
    {"Iris-versicolor": 0, "Iris-virginica": 1, "Iris-setosa": 2}
)


# =============== Split data to X, y ===============
X = df.drop("class", axis=1).values
y = df[["class"]].values


# ===============Create 3 sets from X, y===============
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# =============== Scale our data ===============
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# =============== CONVERT TO TORCH TENSORS & PUT IN THE SAME DEVICE ===============
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.long, device=device).squeeze()
X_valid = torch.tensor(X_valid, dtype=torch.float32, device=device)
y_valid = torch.tensor(y_valid, dtype=torch.long, device=device).squeeze()
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.long, device=device).squeeze()

# =============== Create Data Loaders ===============
BATCH_SIZE = 32
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

valid_data = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# =============== Define our model ===============
class IrisClassifier(nn.Module):
    def __init__(self, input=4, hidden_sizes=(32, 16), output=3):
        super().__init__()
        layers = []
        prev_size = input
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# =============== Initialize our model object, loss & optimizer ===============
model = IrisClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


# =============== TRAINING LOOP ===============
def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()

    total_loss = 0
    total_samples = 0
    correct_predictions = 0

    for X_batch, y_batch in dataloader:
        y_logits = model(X_batch)
        predictions = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        correct_predictions += (predictions == y_batch).sum().item()
    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100

    return avg_loss, accuracy


# =============== EVALUATION/TESTING LOOP ===============
def eval_loop(model, dataloader, loss_fn):
    model.eval()

    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            y_logits = model(X_batch)
            predictions = torch.argmax(y_logits, dim=1)
            loss = loss_fn(y_logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
            correct_predictions += (predictions == y_batch).sum().item()
    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100

    return avg_loss, accuracy


# =============== RUN TRAINING ===============
EPOCHS = 60
PATIENCE = 20
history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
best_valid_loss = float("inf")
patience_counter = 0


for epoch in range(EPOCHS):
    train_loss, train_acc = train_loop(model, train_loader, loss_fn, optimizer)
    valid_loss, valid_acc = eval_loop(model, valid_loader, loss_fn)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Valid-loss: {valid_loss:.4f} | Valid-Acc: {valid_acc:.2f}%"
        )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        torch.save(model.state_dict(), "iris_classifier.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("EARLY STOPPING!!")
            break
