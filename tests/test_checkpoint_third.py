"""Checkpoint entry points must restore both weight shapes and flow semantics."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from fmip import train
from fmip.pl_gmip_model import GMIPModel
from fmip.utils.gp_utils import checkpoint_model_args
from test_training_inference import make_model


class CheckpointEntryTests(unittest.TestCase):
    def checkpoint(self, model, metadata=True):
        result = {'state_dict': model.state_dict()}
        if metadata:
            result['hyper_parameters'] = {'param_args': vars(model.args)}
        return result

    def test_runtime_paths_and_sampling_survive_model_configuration_restore(self):
        original = make_model(True, 'mask', 3)
        original.args.flow_schedule = 'cosine'
        runtime = train.arg_parser(['--dataset_root', '/new/data', '--batch_size', '7',
                                    '--inference_steps', '19', '--use_guidance'])
        resolved = checkpoint_model_args(self.checkpoint(original), runtime)
        self.assertEqual((resolved.hidden_dim, resolved.n_layers), (64, 1))
        self.assertEqual((resolved.discrete_flow_mode, resolved.flow_schedule), ('mask', 'cosine'))
        self.assertEqual((resolved.max_integer_num, resolved.only_discrete), (3, True))
        self.assertEqual((resolved.dataset_root, resolved.batch_size, resolved.inference_steps),
                         ('/new/data', 7, 19))
        self.assertTrue(resolved.use_guidance)
        self.assertEqual(runtime.discrete_flow_mode, 'uniform')
        restored = GMIPModel(resolved, load_datasets=False)
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(restored.DFM.mode, 'mask')
        self.assertEqual(restored.args.flow_schedule, 'cosine')

    def run_entry(self, checkpoint, action):
        trainer = MagicMock()
        logger = SimpleNamespace(save_dir='/tmp/test', name='run', version='1', _id='1')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.ckpt'
            torch.save(checkpoint, path)
            args = train.arg_parser([*action, '--ckpt_path', str(path),
                                     '--dataset_root', '/new/data'])
            with patch.object(train, 'GMIPModel') as model_class, \
                    patch.object(train, 'Trainer', return_value=trainer), \
                    patch.object(train, 'WandbLogger', return_value=logger), \
                    patch.object(train, 'ModelCheckpoint'), patch.object(train, 'rank_zero_info'):
                train.main(args)
            if '--resume_weight_only' in action:
                resolved = model_class.load_from_checkpoint.call_args.kwargs['param_args']
                trainer.fit.assert_called_once_with(model_class.load_from_checkpoint.return_value)
            else:
                resolved = model_class.call_args.kwargs['param_args']
                if '--do_train' in action:
                    trainer.fit.assert_called_once_with(model_class.return_value, ckpt_path=str(path))
                else:
                    trainer.validate.assert_called_once_with(model_class.return_value, ckpt_path=str(path))
        self.assertEqual(resolved.dataset_root, '/new/data')
        return resolved

    def test_cli_validation_restores_mask_and_continuous_path(self):
        original = make_model(False, 'mask', 3)
        original.args.flow_schedule = 'cosine'
        args = self.run_entry(self.checkpoint(original), ['--do_valid_only'])
        self.assertEqual((args.discrete_flow_mode, args.flow_schedule), ('mask', 'cosine'))
        self.assertEqual((args.n_layers, args.max_integer_num), (1, 3))

    def test_cli_full_and_weights_only_resume_restore_training_semantics(self):
        original = make_model(True, 'mask', 3)
        original.args.flow_schedule = 'linearVP'
        for options in (['--do_train'], ['--do_train', '--resume_weight_only']):
            with self.subTest(options=options):
                args = self.run_entry(self.checkpoint(original), options)
                self.assertEqual((args.discrete_flow_mode, args.flow_schedule), ('mask', 'linearVP'))
                self.assertTrue(args.only_discrete)

    def test_cli_legacy_checkpoint_infers_architecture(self):
        original = make_model(True, classes=4)
        args = self.run_entry(self.checkpoint(original, metadata=False), ['--do_test'])
        self.assertEqual((args.n_layers, args.hidden_dim, args.max_integer_num), (1, 64, 4))
        self.assertTrue(args.only_discrete)


if __name__ == '__main__':
    unittest.main()
