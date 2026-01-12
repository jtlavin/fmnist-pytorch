import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import shutil
import glob

# Load Data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, hidden_size=512, activation_fn=nn.ReLU()):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, hidden_size),
            activation_fn,
            nn.Linear(hidden_size, hidden_size),
            activation_fn,
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Train and Evaluate Function
def train_and_evaluate(config, train_data, test_data, device):
    hidden_size = config.get('hidden_size', 512)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 5)
    optimizer_name = config.get('optimizer', 'SGD')
    activation_name = config.get('activation', 'ReLU')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    activation_map = {'ReLU': nn.ReLU(), 'Tanh': nn.Tanh(), 'Sigmoid': nn.Sigmoid()}
    activation_fn = activation_map.get(activation_name, nn.ReLU())

    model = NeuralNetwork(hidden_size=hidden_size, activation_fn=activation_fn).to(device)

    loss_fn = nn.CrossEntropyLoss()
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = {'accuracy': [], 'loss': [], 'config': config}

    print(f"Training with config: {config}")

    for t in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        history['accuracy'].append(correct)
        history['loss'].append(test_loss)
        
    return history

# --- Correction Experiment ---

baseline_config = {
    'hidden_size': 512,
    'lr': 1e-3,
    'batch_size': 64,
    'optimizer': 'SGD',
    'activation': 'ReLU',
    'epochs': 10
}

failed_config = {
    'hidden_size': 512,
    'lr': 0.01,
    'batch_size': 32,
    'optimizer': 'Adam',
    'activation': 'Tanh',
    'epochs': 10
}

corrected_config = {
    'hidden_size': 512,
    'lr': 0.001,  # Corrected LR for Adam
    'batch_size': 32,
    'optimizer': 'Adam',
    'activation': 'Tanh',
    'epochs': 10
}

print("\n--- Running Correction Comparison ---")

print("1. Training Baseline Model...")
baseline_hist = train_and_evaluate(baseline_config, training_data, test_data, device)

print("2. Training Failed Winner (High LR)...")
failed_hist = train_and_evaluate(failed_config, training_data, test_data, device)

print("3. Training Corrected Winner (Normal LR)...")
corrected_hist = train_and_evaluate(corrected_config, training_data, test_data, device)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(baseline_hist['accuracy'], label='Baseline (SGD, 0.001)', linestyle='--')
plt.plot(failed_hist['accuracy'], label='Failed Winner (Adam, 0.01)', color='red')
plt.plot(corrected_hist['accuracy'], label='Corrected Winner (Adam, 0.001)', color='green', linewidth=2)

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Correction: Impact of Learning Rate on Adam Optimizer')
plt.legend()
plt.grid(True)
plt.savefig('comparison_correction.png')
print("Saved comparison plot to comparison_correction.png")

# --- Cleanup ---
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

png_files = glob.glob('*.png')
for file in png_files:
    shutil.move(file, os.path.join(plots_dir, file))

print(f"Moved plot files to {plots_dir}/")
