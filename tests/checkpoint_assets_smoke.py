"""Check structural compatibility of a directory of trusted local checkpoints."""
from pathlib import Path
import sys

import numpy as np

from fmip.utils.gp_utils import load_policy_checkpoint
from test_data_network import make_graph


def main():
    root = Path(sys.argv[1])
    checkpoints = sorted(root.glob('*/diff.ckpt'))
    if not checkpoints:
        raise ValueError(f'No checkpoints found under {root}')
    for checkpoint in checkpoints:
        model = load_policy_checkpoint(checkpoint, device='cpu')
        model.inference_steps = 2
        model.inference_scheduler = type(model.inference_scheduler)(2, 'linear')
        result = model.inference(make_graph(continuous=not model.only_discrete))
        assert result['Ivars'].shape == (2, model.max_integer_num)
        assert np.isfinite(result['Ivars']).all()
        if not model.only_discrete:
            assert np.isfinite(result['Cvars']).all()
        print(f'CHECKPOINT_OK {checkpoint.parent.name}: layers={model.args.n_layers}, '
              f'hidden={model.args.hidden_dim}, classes={model.max_integer_num}, '
              f'only_discrete={model.only_discrete}')
    print(f'CHECKPOINT_ASSETS_OK: {len(checkpoints)}')


if __name__ == '__main__':
    main()
