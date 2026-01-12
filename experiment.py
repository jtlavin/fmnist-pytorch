import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import copy

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

def run_experiment_set(experiment_name, vary_param, param_values, baseline_config):
    print(f"\n--- Running {experiment_name} Experiment ---")
    results = []
    
    for val in param_values:
        config = copy.deepcopy(baseline_config)
        config[vary_param] = val
        hist = train_and_evaluate(config, training_data, test_data, device)
        results.append(hist)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for res in results:
        val = res['config'][vary_param]
        plt.plot(res['accuracy'], label=f'{vary_param}={val}')
        
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Impact of {vary_param} on Model Accuracy')
    plt.legend()
    plt.grid(True)
    filename = f'impact_{vary_param}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# --- One Factor At A Time (OFAT) Experiments ---

# Baseline Configuration
BASELINE_CONFIG = {
    'hidden_size': 512,
    'lr': 1e-3,
    'batch_size': 64,
    'optimizer': 'SGD',
    'activation': 'ReLU',
    'epochs': 5
}

# 1. Learning Rate
run_experiment_set(
    "Learning Rate",
    "lr", 
    [1e-2, 1e-3, 1e-4], 
    BASELINE_CONFIG
)

# 2. Batch Size
run_experiment_set(
    "Batch Size",
    "batch_size", 
    [32, 64, 128], 
    BASELINE_CONFIG
)

# 3. Hidden Size
run_experiment_set(
    "Hidden Size",
    "hidden_size", 
    [128, 256, 512], 
    BASELINE_CONFIG
)

# 4. Optimizer
run_experiment_set(
    "Optimizer",
    "optimizer", 
    ['SGD', 'Adam', 'RMSprop'], 
    BASELINE_CONFIG
)

# 5. Activation
run_experiment_set(
    "Activation Experiment",
    "activation", 
    ['ReLU', 'Tanh', 'Sigmoid'], 
    BASELINE_CONFIG
)

print("\nAll experiments completed.")
