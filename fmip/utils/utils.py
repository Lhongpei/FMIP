import torch
import numpy as np
import os
def generate_unique_samples(pred_Ivars, K, max_attempts=1000):
    """
    pred_Ivars: (N, 2) Array of probabilities for each variable.
    K: Number of unique samples to generate (Can be higher than K).
    max_attempts: Maximum attempts to generate unique samples.
    """
    if not isinstance(K, int) or K < 0 or not isinstance(max_attempts, int) or max_attempts < 0:
        raise ValueError("K and max_attempts must be nonnegative integers")
    if K == 0:
        return np.empty((0, len(pred_Ivars)), dtype=np.int64)
    unique_sample = {}
    with torch.no_grad():
        pred_Ivars = torch.as_tensor(pred_Ivars)
        if not pred_Ivars.is_floating_point():
            pred_Ivars = pred_Ivars.float()
        sampled_Ivars = torch.multinomial(pred_Ivars, K, replacement=True).T
        unique_sample.update(dict.fromkeys(map(tuple, sampled_Ivars.cpu().tolist())))
        attempts = 0
        while len(unique_sample) < K and attempts < max_attempts:
            sampled_Ivars = torch.multinomial(pred_Ivars, K, replacement=True).T
            unique_sample.update(dict.fromkeys(map(tuple, sampled_Ivars.cpu().tolist())))
            attempts += 1
        if len(unique_sample) < K:
            print(f"Warning: Only {len(unique_sample)} unique samples generated out of {K} requested.")
    return np.array(list(unique_sample)[:K], dtype=np.int64)

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
    stem = problem_name[:-len(file_type)] if file_type else problem_name
    solution_path = os.path.join(solution_root, stem + '_sol.pkl')
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
    a = a.to(device=x.device, dtype=torch.float32)
    t = t.to(device=x.device, dtype=torch.long)
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
    a = a.to(device=x.device, dtype=x.dtype)
    t = t.to(device=x.device, dtype=torch.long)

    # Gather `a` along the 0th dimension using `t` as indices
    a_t = a[t]  # a_t.shape = (batch_size, num_pixel_vals, num_pixel_vals)

    # Perform the matrix multiplication
    # x.shape = (batch_size, ..., num_pixel_vals)
    # a_t.shape = (batch_size, num_pixel_vals, num_pixel_vals)
    # out.shape = (batch_size, ..., num_pixel_vals)
    out = torch.einsum('b...i,bij->b...j', x, a_t)
    return out
