from scipy.stats import wasserstein_distance
import numpy as np

import torch
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def embed_sentences(texts, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    model = model.to(device)
    return model.encode(texts, convert_to_tensor=True, device=device)


def average_wasserstein_distance(set1, set2):
    """
    Approximate Wasserstein-1 distance between two sets of d-dimensional vectors
    by averaging 1D Wasserstein distances across dimensions.
    Args:
        set1, set2: numpy arrays of shape (N1, d), (N2, d)
    Returns:
        float: average W1 distance over all dimensions
    """
    assert set1.shape[1] == set2.shape[1], "Dimension mismatch"
    d = set1.shape[1]
    distances = [
        wasserstein_distance(set1[:, i].cpu().numpy(), set2[:, i].cpu().numpy())
        for i in range(d)
    ]
    return np.mean(distances)