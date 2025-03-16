import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['lines.linewidth'] = 2.0
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Computer Modern Roman']
# rcParams['text.usetex'] = False
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': False,
    # 'text.latex.preamble': r'\usepackage{amsfonts}'
})

def visualize_decision_boundary(models, titles, X, y, device, padding=5,epoch=None):
    """
    Visualize decision boundaries and predictive uncertainties
    
    Args:
        models: List of models to visualize
        titles: List of titles for each model
        X: Input data (numpy array)
        y: Target labels (numpy array)
        device: PyTorch device (e.g., "cpu" or "cuda")
    """
    num_models = len(models)
    plt.figure(figsize=(7*num_models, 6))  # Create a single figure
    plt.subplots_adjust(wspace=0.05)

    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    contour_ref = None  # Store reference for colorbar

    for i, (model, title) in enumerate(zip(models, titles)):
        plt.subplot(1, num_models, i + 1)  # Create subplots manually
        model.eval()
        
        # Make predictions on the mesh grid
        with torch.no_grad():
            Z = torch.softmax(model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)), dim=1)
            Z = Z.cpu().numpy()
        
        # Extract class probabilities
        probs = Z[:, 1].reshape(xx.shape)
        
        # Plot decision boundary
        contour_ref = plt.contourf(xx, yy, probs, levels=20, cmap=cm.coolwarm)

        # Scatter plot for training data
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="cornflowerblue", edgecolors="black", s=60, label="Class 0", alpha=0.8)
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="tomato", edgecolors="black", s=60, label="Class 1", alpha=0.8)

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(title, fontsize=18)
        plt.xticks([])
        plt.yticks([])

    # Add a single colorbar to the right of all subplots
    # cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust for positioning
    cbar = plt.colorbar(contour_ref)
    cbar.ax.set_ylabel(
        "$\mathbb{E}_{\Theta}[p(y_{*} | x_{*}, \Theta; f) | \mathcal{D}]$", 
        rotation=270, 
        labelpad=40, 
        fontsize=20
    )
    cbar.ax.tick_params(labelsize=16)

    plt.savefig(f"images/decision_boundary_{epoch}.png", dpi=300)
    

def get_context_dist(X, y, context_set_size=100, x_context_lim = 0.5):
    """
    Get a distribution of context points
    
    Args:
        X: Input data (numpy array)
        y: Target labels (numpy array)
        num_context: Number of context points to sample
    """
    x_context_lim = 0.5  # Context grid range

    # Define the range for the context grid based on training data
    x_context_min, x_context_max = (
        X[:, 0].min() - x_context_lim,
        X[:, 0].max() + x_context_lim,
    )
    y_context_min, y_context_max = (
        X[:, 1].min() - x_context_lim,
        X[:, 1].max() + x_context_lim,
    )

    # Calculate grid resolution based on the context set size
    h_context_x = (x_context_max - x_context_min) / context_set_size ** 0.5
    h_context_y = (y_context_max - y_context_min) / context_set_size ** 0.5

    xx_context, yy_context = np.meshgrid(
        np.arange(x_context_min, x_context_max, h_context_x),
        np.arange(y_context_min, y_context_max, h_context_y)
    )
    x_context = np.vstack((xx_context.reshape(-1), yy_context.reshape(-1))).T
    
    return torch.FloatTensor(x_context)

def evaluate(model, test_loader, device):
    """
    Evaluate a model on test data
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        device: Device to use for evaluation
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_out_sample(model, out_sample_loader, device):
    """
    Evaluate a model on out-of-sample data
    Args:
        model: The model to evaluate
        out_sample_loader: DataLoader for out-of-sample data
        device: Device to use for evaluation
    """
    model.eval()
    res = 0
    total = 0
    with torch.no_grad():
        for data, target in out_sample_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.softmax(output, dim=1)
            values, _ = torch.max(output, 1)
            res += values.sum().item()
            total += target.size(0)
    return res / total

def get_out_sample_predictions(model, out_sample_loader, device):
    """
    Get predictions on out-of-sample data
    Args:
        model: The model to evaluate
        out_sample_loader: DataLoader for out-of-sample data
        device: Device to use for evaluation
    """
    model.eval()
    predictions = []
    all_labels = []
    with torch.no_grad():
        for data in out_sample_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            values, _ = torch.max(outputs, 1)
            predictions.extend((values).cpu().numpy())
            all_labels.extend((1 - labels).cpu().numpy())
    
    predictions = np.array(predictions)
    all_labels = np.array(all_labels)

    return predictions, all_labels

def train_standard(model, train_loader, optimizer, device, weight_decay=1e-4):
    """
    Train a model using standard parameter-space MAP estimation
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to use for training
        weight_decay: Weight decay parameter for L2 regularization
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    return model