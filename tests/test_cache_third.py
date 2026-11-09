"""Cache provenance, concurrent first use, and unsupported objective regressions."""

import json
import multiprocessing
import os
import pickle
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gurobipy as gp
import torch

from fmip.dataset import GraphMILPDataset, SplitSchedule, _atomic_write
from fmip.utils.milp_reader import MIPmodel
from test_data_network import make_graph


def _fake_components():
    graph = make_graph()
    result = {'x_' + kind: graph[kind].x for kind in ('cons', 'Cvars', 'Ivars')}
    for prefix, kind in (('C', 'Cvars'), ('I', 'Ivars')):
        for field in ('edge_attr', 'edge_index'):
            result[prefix + '2cons_' + field] = graph[kind, 'to', 'cons'][field]
    return result


def _cache_worker(kind, root, cache, entered, processing, release, result, calls):
    """Delay the first writer while a second process attempts initialization."""
    torch.set_num_threads(1)

    def record_processing():
        with open(calls, 'a') as stream:
            stream.write(str(os.getpid()) + '\n')
        processing.set()
        if not release.wait(20):
            raise TimeoutError('Writer was not released')

    try:
        entered.set()
        if kind == 'dataset':
            class Reader:
                def __init__(self, filename):
                    pass

                def generGraphCom(self):
                    record_processing()
                    return _fake_components()

            with patch('fmip.dataset.MIPmodel', Reader):
                data = GraphMILPDataset(root, cache, given_solution=False)
                result.put(('ok', len(data)))
        else:
            original = SplitSchedule.split_data

            def split_data(schedule):
                record_processing()
                return original(schedule)

            with patch.object(SplitSchedule, 'split_data', split_data):
                result.put(('ok', SplitSchedule(root, cache).get_split()))
    except Exception as error:
        result.put(('error', repr(error)))


class CacheProvenanceTests(unittest.TestCase):
    def files(self, root, count=3):
        (root / 'problem').mkdir(parents=True)
        (root / 'solution').mkdir()
        for index in range(count):
            (root / 'problem' / f'p{index}.lp').touch()
            (root / 'solution' / f'p{index}_sol.pkl').touch()

    def test_automatic_dataset_selection_tracks_added_and_removed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.files(root, count=1)
            with patch('fmip.dataset.MIPmodel') as reader:
                reader.return_value.generGraphCom.side_effect = lambda: _fake_components()
                first = GraphMILPDataset(str(root), str(root / 'cache'), given_solution=False)
                (root / 'problem' / 'p1.lp').touch()
                second = GraphMILPDataset(str(root), str(root / 'cache'), given_solution=False)
                (root / 'problem' / 'p0.lp').unlink()
                third = GraphMILPDataset(str(root), str(root / 'cache'), given_solution=False)
            self.assertEqual([len(first), len(second), len(third)], [1, 2, 1])
            self.assertEqual(third[0]['name'], 'p1.lp')
            self.assertEqual(len({first.processed_dir, second.processed_dir, third.processed_dir}), 3)
            self.assertTrue(Path(first.processed_paths[0]).exists())

    def test_saved_split_rejects_changed_source_ratio_or_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'first'
            other = Path(temporary) / 'second'
            self.files(root)
            self.files(other)
            cache = str(Path(temporary) / 'cache')
            original = SplitSchedule(str(root), cache).get_split()
            for source, ratio in ((other, [0.9, 0.05, 0.05]), (root, [0.8, 0.1, 0.1])):
                with self.subTest(source=source, ratio=ratio):
                    with self.assertRaisesRegex(ValueError, 'source, inventory, or configuration'):
                        SplitSchedule(str(source), cache, split_ratio=ratio)
            self.assertEqual(SplitSchedule(str(root), cache).get_split(), original)
            (root / 'problem' / 'extra.lp').touch()
            (root / 'solution' / 'extra_sol.pkl').touch()
            with self.assertRaisesRegex(ValueError, 'source, inventory, or configuration'):
                SplitSchedule(str(root), cache)

    def test_legacy_split_is_preserved_and_receives_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.files(root)
            cache = root / 'cache'
            cache.mkdir()
            partition = {'train': ['p2.lp'], 'valid': ['p0.lp'], 'test': ['p1.lp']}
            with (cache / 'split.pkl').open('wb') as stream:
                pickle.dump(partition, stream)
            self.assertEqual(SplitSchedule(str(root), str(cache)).get_split(), partition)
            metadata = json.loads((cache / 'split.meta.json').read_text())
            self.assertEqual(metadata['root'], str(root.resolve()))
            self.assertEqual(metadata['problems'], ['p0.lp', 'p1.lp', 'p2.lp'])

    def test_legacy_split_cannot_silently_omit_new_instances(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.files(root, count=4)
            cache = root / 'cache'
            cache.mkdir()
            with (cache / 'split.pkl').open('wb') as stream:
                pickle.dump({'train': ['p0.lp'], 'valid': ['p1.lp'], 'test': ['p2.lp']}, stream)
            with self.assertRaisesRegex(ValueError, 'missing problems'):
                SplitSchedule(str(root), str(cache))
            self.assertFalse((cache / 'split.meta.json').exists())

    def test_interrupted_serialization_keeps_the_previous_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'cache.pkl'
            target.write_bytes(b'previous complete cache')

            def failed_writer(stream):
                stream.write(b'incomplete replacement')
                raise RuntimeError('serialization failed')

            with self.assertRaisesRegex(RuntimeError, 'serialization failed'):
                _atomic_write(str(target), failed_writer)
            self.assertEqual(target.read_bytes(), b'previous complete cache')
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def concurrent_initialization(self, kind):
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.files(root, count=1 if kind == 'dataset' else 3)
            result = context.Queue()
            processing, release = context.Event(), context.Event()
            entries = [context.Event(), context.Event()]
            calls = root / 'process_calls.txt'
            workers = [context.Process(target=_cache_worker,
                args=(kind, str(root), str(root / 'cache'), entry, processing,
                      release, result, str(calls))) for entry in entries]
            try:
                workers[0].start()
                self.assertTrue(processing.wait(20), 'First writer did not start')
                workers[1].start()
                self.assertTrue(entries[1].wait(20), 'Second process did not start')
                with self.assertRaises(queue.Empty):
                    result.get(timeout=0.3)
                release.set()
                outputs = [result.get(timeout=30) for _ in workers]
                self.assertTrue(all(item[0] == 'ok' for item in outputs), outputs)
                self.assertEqual(outputs[0], outputs[1])
                for worker in workers:
                    worker.join(timeout=20)
                    self.assertEqual(worker.exitcode, 0)
                self.assertEqual(len(calls.read_text().splitlines()), 1)
            finally:
                release.set()
                for worker in workers:
                    if worker.pid is not None:
                        if worker.is_alive():
                            worker.terminate()
                        worker.join(timeout=10)
                result.close()
                result.join_thread()

    def test_concurrent_dataset_initialization_processes_once(self):
        self.concurrent_initialization('dataset')

    def test_concurrent_split_initialization_uses_one_partition(self):
        self.concurrent_initialization('split')


class PiecewiseObjectiveTests(unittest.TestCase):
    def test_real_piecewise_linear_objective_is_rejected(self):
        with gp.Env(empty=True) as environment:
            environment.setParam('OutputFlag', 0)
            environment.start()
            with gp.Model(env=environment) as model:
                model.Params.Threads = 1
                variable = model.addVar(lb=0, ub=2, vtype=gp.GRB.INTEGER)
                model.setPWLObj(variable, [0., 1., 2.], [1., 0., 1.])
                model.update()
                self.assertEqual(model.NumPWLObjVars, 1)
                self.assertEqual(model.NumQNZs, 0)
                self.assertEqual(variable.Obj, 0.)
                reader = MIPmodel.__new__(MIPmodel)
                reader.model = model
                with self.assertRaisesRegex(ValueError, 'NumPWLObjVars'):
                    reader.generGraphCom()


if __name__ == '__main__':
    unittest.main()
