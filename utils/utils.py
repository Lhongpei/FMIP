import torch
import numpy as np
import os
def generate_unique_samples(pred_Ivars, K, max_attempts=1000):
    """
    pred_Ivars: (N, 2) Array of probabilities for each variable.
    K: Number of unique samples to generate (Can be higher than K).
    max_attempts: Maximum attempts to generate unique samples.
    """
    unique_sample = set()
    with torch.no_grad():
        pred_Ivars = torch.tensor(pred_Ivars, device='cuda' if torch.cuda.is_available() else 'cpu')
        sampled_Ivars = torch.multinomial(pred_Ivars, K, replacement=True).T
        unique_sample.update(set(map(tuple, sampled_Ivars.cpu().tolist())))
        attempts = 0
        while len(unique_sample) < K and attempts < max_attempts:
            sampled_Ivars = torch.multinomial(pred_Ivars, K, replacement=True).T
            unique_sample.update(set(map(tuple, sampled_Ivars.cpu().tolist())))
            attempts += 1
        if len(unique_sample) < K:
            print(f"Warning: Only {len(unique_sample)} unique samples generated out of {K} requested.")
    return np.array(list(unique_sample))

def get_file_extension(filename):
    base, ext = os.path.splitext(filename)
    while ext.lower() == '.gz' and '.' in base:
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    if ext.lower() == '.lp':
        base, ext2 = os.path.splitext(base)
        if ext2.lower() == '.proto':
            ext = ext2 + ext
    return ext.lower() if ext else ''

def find_respect_solution(solution_root, problem_name):
    file_type = get_file_extension(problem_name)
    solution_path = os.path.join(solution_root, problem_name.replace(file_type, '_sol.pkl'))
    return solution_path

def at(a, t, x):
    """
    Extract coefficients at specified timesteps t and conditioning data x.

    Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, shape = (num_timesteps, num_pixel_vals, num_pixel_vals).
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (batch_size, ...), int32 or int64 type.

    Returns:
        torch.Tensor: Extracted coefficients, shape = (batch_size, ..., num_pixel_vals).
    """
    a = a.to(dtype=torch.float32)  # Convert to float32 if necessary
    t_broadcast = t.view(t.shape + (1,) * (x.dim() - 1))  # Broadcast t to match the leading dimensions of x

    # Advanced indexing for selecting values
    return a[t_broadcast, x]

def at_onehot(a, t, x):
    """
    Extract coefficients at specified timesteps t and conditioning data x (one-hot representation).

    Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, shape = (num_timesteps, num_pixel_vals, num_pixel_vals).
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (batch_size, ..., num_pixel_vals), float32 type.

    Returns:
        torch.Tensor: Result of the dot product of x with a[t], shape = (batch_size, ..., num_pixel_vals).
    """
    a = a.to(dtype=torch.float32)  # Convert to float32 if necessary

    # Gather `a` along the 0th dimension using `t` as indices
    a_t = a[t]  # a_t.shape = (batch_size, num_pixel_vals, num_pixel_vals)

    # Perform the matrix multiplication
    # x.shape = (batch_size, ..., num_pixel_vals)
    # a_t.shape = (batch_size, num_pixel_vals, num_pixel_vals)
    # out.shape = (batch_size, ..., num_pixel_vals)
    out = torch.einsum('...i,bij->...j', x, a_t)  # Batch matrix multiplication
    return out
