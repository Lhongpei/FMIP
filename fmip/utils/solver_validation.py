"""Input contracts and instance selection for downstream solvers."""

import numpy as np
from pathlib import Path
import pickle


def load_test_instances(problem_root, split_path=None, dataset_cache=None, limit=None):
    """Load a complete test split, with an optional explicit instance limit.

    An explicit split takes precedence. Otherwise, look beside the problem
    directory, then in the checkpoint's dataset cache for the same dataset.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('instance_limit must be a positive integer or None')
    problem_root = Path(problem_root)
    if split_path is None:
        candidates = [problem_root.parent / 'split.pkl']
        if dataset_cache is not None:
            candidates.append(Path(dataset_cache) / problem_root.parent.name / 'split.pkl')
        split_path = next((path for path in candidates if path.is_file()), None)
        if split_path is None:
            raise FileNotFoundError('No test split found; pass split_path to the downstream optimizer')
    with open(split_path, 'rb') as stream:
        split = pickle.load(stream)
    names = split.get('test') if isinstance(split, dict) else None
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError('The split must contain a nonempty test list of problem filenames')
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError('Test instances must be filenames within the problem directory')
    if len(names) != len(set(names)):
        raise ValueError('The test split contains duplicate problem filenames')
    # Validate the full split before starting any expensive inference or solve.
    for name in names:
        if not (problem_root / name).is_file():
            raise FileNotFoundError(f'Test instance does not exist: {problem_root / name}')
    return sorted(names)[:limit]


def history_path(output_file):
    """Use a distinct sidecar name regardless of the CSV filename's suffix."""
    output_file = Path(output_file)
    return output_file.with_name(output_file.stem + '_history.pkl')


def require_binary_domains(variables):
    if any(variable.VType not in ('B', 'I') or variable.LB < 0 or variable.UB > 1 for variable in variables):
        raise ValueError("This downstream heuristic only supports integer domains within {0, 1}")


def binary_probabilities(probabilities, variables):
    require_binary_domains(variables)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(variables) or probabilities.shape[1] < 1:
        raise ValueError("Expected one categorical probability row per registered integer variable")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Predicted probabilities must be finite and nonnegative")
    # A reserved mask column (or unused higher category) may be retained by FMIP.
    # It is safe to remove only when it carries no probability mass.
    if probabilities.shape[1] > 2 and np.any(probabilities[:, 2:] != 0):
        raise ValueError("Binary heuristics cannot use predictions with mass outside classes 0 and 1")
    probabilities = probabilities[:, :2]
    if probabilities.shape[1] == 1:
        probabilities = np.pad(probabilities, ((0, 0), (0, 1)))
    total = probabilities.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("Each variable needs positive probability mass on classes 0 or 1")
    return probabilities / total


def validate_trust_region(k_zero, k_one, delta_rate):
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in (k_zero, k_one, delta_rate)):
        raise ValueError("Trust-region fractions must lie in [0, 1]")
    if k_zero + k_one > 1:
        raise ValueError("Zero-fixed and one-fixed fractions must not overlap")
