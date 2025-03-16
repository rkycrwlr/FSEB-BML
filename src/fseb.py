import torch
import torch.nn as nn


class FSEB:
    def __init__(self, model, prior_model, tau_f=1.0, tau_theta=1e-4, sigma=1e-20, context_batch_size=32, rbf_kernel=False):
        """
        Function-Space Empirical Bayes regularizer
        
        Args:
            model: The model being trained
            prior_model: The model used for the prior distribution over functions
            tau_f: Precision parameter for function-space regularization
            tau_theta: Precision parameter for parameter-space regularization
        """
        self.model = model
        self.prior_model = prior_model
        self.tau_f = tau_f
        self.tau_theta = tau_theta
        self.sigma = sigma
        self.context_batch_size = context_batch_size
        self.rbf_kernel = rbf_kernel
        
        # Freeze the prior model
        for param in self.prior_model.parameters():
            param.requires_grad = False

    def kernel_rbf(self, h, sigma=1.0):
        # compute rbf kernel matrix K(h,h)
        K = torch.exp(-torch.cdist(h, h, p=2.0) ** 2 / (2 * sigma ** 2))
        return K

    def compute_kernel(self, x):
        """Compute the kernel matrix K(x, x; phi_0)"""
        h_x = self.prior_model(x)

        if self.rbf_kernel:
            return self.kernel_rbf(h_x) + torch.eye(x.size(0), device=x.device)
        
        return torch.matmul(h_x, h_x.t()) + torch.eye(x.size(0), device=x.device)
    
    def compute_mahalanobis_distance(self, f_x, K):
        """Compute the squared Mahalanobis distance"""
        # Numerically stable inverse using SVD
        # U, S, V = torch.svd(K)
        # K_inv = torch.matmul(V, torch.matmul(torch.diag(1.0 / S), U.t()))
        K_inv = torch.inverse(K)
        # Compute squared Mahalanobis distance for each output dimension
        dist = 0
        for k in range(f_x.size(1)):
            f_k = f_x[:, k].unsqueeze(1)
            dist += torch.matmul(f_k.t(), torch.matmul(K_inv, f_k))
        
        return dist.squeeze()
    
    def noise_model(self):
        """Add noise to the model parameters"""
        for param in self.model.parameters():
            param.data.add_(torch.randn_like(param) * self.sigma)

    def reset_model(self, original_params):
        """Reset the model parameters"""
        for param, orig in zip(self.model.parameters(), original_params):
            param.data.copy_(orig)
    
    def compute_regularizer(self, context_subset_loader, device, mc_samples=10):
        """Compute the FS-EB regularizer F(θ)"""

        original_params = [p.clone() for p in self.model.parameters()]
        regularizer = 0.0
        for _ in range(mc_samples):
            self.noise_model()

            # Sample a random context set
            x_hat = next(iter(context_subset_loader)).to(device)

            f_x_hat = self.model(x_hat)
            
            # Compute kernel matrix
            K = self.compute_kernel(x_hat)
            
            # Compute Mahalanobis distance
            mahalanobis_dist = self.compute_mahalanobis_distance(f_x_hat, K) / self.context_batch_size

            reg_output = torch.sum(f_x_hat**2) / self.context_batch_size
            
            # Compute L2 norm of parameters
            l2_norm = 0
            for param in self.model.parameters():
                l2_norm += torch.sum(param ** 2)
            
            # Compute the regularizer
            regularizer += -0.5 * self.tau_f * mahalanobis_dist - 0.5 * self.tau_theta * l2_norm #- reg_output

            self.reset_model(original_params)
        
        return regularizer / mc_samples
    
    def compute_nll(self, x, y, criterion, mc_samples=10):
        """Compute the negative log-likelihood"""
        original_params = [p.clone() for p in self.model.parameters()]
        loss = 0.0
        for _ in range(mc_samples):
            self.noise_model()
            loss += criterion(self.model(x), y)
            self.reset_model(original_params)
        
        return loss / mc_samples


def train_fseb(model, prior_model, train_loader, context_subset_loader, optimizer, device, 
               tau_f=1.0, tau_theta=1e-4, context_batch_size=32):
    """
    Train a model using Function-Space Empirical Bayes
    
    Args:
        model: The model to train
        prior_model: The model used for the prior distribution over functions
        train_loader: DataLoader for training data
        context_dist: Distribution to sample context points from
        optimizer: Optimizer for training
        device: Device to use for training
        tau_f: Precision parameter for function-space regularization
        tau_theta: Precision parameter for parameter-space regularization
        context_batch_size: Number of context points to sample per batch
    """
    model.train()
    fseb = FSEB(model, prior_model, tau_f, tau_theta, context_batch_size=context_batch_size)
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Compute cross-entropy loss
        ce_loss = fseb.compute_nll(data, target, criterion)
        
        # Compute FS-EB regularizer
        reg_loss = -fseb.compute_regularizer(context_subset_loader, device)
        
        # Total loss
        loss = ce_loss + reg_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    return model