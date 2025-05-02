import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Sentence Embeddings using SBERT
def embed_sentences(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    model = model.to(device)
    return model.encode(texts, convert_to_tensor=True, device=device)

# 2. Cost Matrix (Squared Euclidean)
def compute_cost_matrix(X, Y):
    return torch.cdist(X, Y, p=2).pow(2)

# 3. Sinkhorn Gradient (Stable Dual)
def stable_sinkhorn_gradient(M, epsilon, a, b, grad_clip_value=1e5):
    M = M.to(torch.float32)
    a = a.to(torch.float32)
    b = b.to(torch.float32)

    B, b_size = M.shape

    def logsumexp(x, axis):
        x_max = torch.max(x, dim=axis, keepdim=True).values
        return x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True))

    def f_obj(beta_flat):
        beta = torch.tensor(beta_flat, dtype=torch.float32, device=M.device).reshape(b_size, 1)
        log_a = torch.log(torch.clamp(a, min=1e-10))
        M_diff = (beta.T - M) / epsilon
        alpha = (1 / epsilon) * (log_a - logsumexp(M_diff, axis=1))
        f = -torch.sum(alpha * a) - torch.sum(beta * b)
        return -f.item()

    beta0 = torch.zeros(b_size, dtype=torch.float32).cpu()
    result = minimize(f_obj, beta0.numpy(), method='L-BFGS-B')
    beta_opt = torch.tensor(result.x.reshape(b_size, 1), dtype=M.dtype, device=M.device)

    alpha_opt = (1 / epsilon) * (
        torch.log(torch.clamp(a, min=1e-10)) - torch.logsumexp((beta_opt.T - M) / epsilon, dim=1)
    ).view(-1, 1)
    T_star = torch.exp(torch.clamp(epsilon * (alpha_opt + beta_opt.T - M), max=1e10))
    grad_M = -T_star
    grad_M = torch.where(torch.isnan(grad_M) | torch.isinf(grad_M), torch.zeros_like(grad_M), grad_M)
    grad_M = torch.clamp(grad_M, min=-grad_clip_value, max=grad_clip_value)
    return T_star, grad_M

# 4. Gradient w.r.t. coreset embeddings
def gradient_step(D, Y, grad_M):
    B = D.shape[0]
    grad = torch.zeros_like(D)
    for i in range(B):
        diff = D[i].unsqueeze(0) - Y
        grad[i] = (2 * grad_M[i].unsqueeze(1) * diff).sum(dim=0)
    return grad

# 5. Adam Optimizer
class AdamOptimizer:
    def __init__(self, shape, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, device='cpu'):
        self.m = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (v_hat.sqrt() + self.eps)

# 6. Main WCSL Algorithm
def wasserstein_coreset(texts, B=20, b=32, T=100, lambda_reg=100.0,
                        delta_stop=1e-3, max_epochs=20, lr=1e-2, grad_clip_value=1e5, device='cuda'):
    embeddings = embed_sentences(texts).to(device)
    N, d = embeddings.shape

    idx = random.sample(range(N), B)
    D = embeddings[idx].clone().detach().to(device)
    D_prev = D.clone()
    adam = AdamOptimizer(D.shape, lr=lr, device=device)
    a = torch.full((B,), 1.0 / B, device=device)

    for epoch in range(max_epochs):
        for t in range(T):
            y_idx = torch.randint(0, N, (b,))
            Y = embeddings[y_idx].to(device)
            b_vec = torch.full((b,), 1.0 / b, device=device)

            M = compute_cost_matrix(D, Y)
            _, grad_M = stable_sinkhorn_gradient(M, epsilon=1.0 / lambda_reg, a=a, b=b_vec, grad_clip_value=grad_clip_value)
            grad_D = gradient_step(D, Y, grad_M)
            D = adam.step(D, grad_D)

        # Convergence check
        M_conv = compute_cost_matrix(D_prev, D)
        _, loss_conv = stable_sinkhorn_gradient(M_conv, epsilon=1.0 / lambda_reg, a=a, b=a, grad_clip_value=grad_clip_value)
        diff = loss_conv.abs().mean().item()
        print(f"Epoch {epoch}: Sinkhorn diff = {diff:.6f}")
        if diff < delta_stop:
            print("Converged.")
            break
        D_prev = D.clone().detach()

    sim = cosine_similarity(D.cpu().numpy(), embeddings.cpu().numpy())
    selected = [texts[row.argmax()] for row in sim]
    return selected