"""Regression tests for training entry points, MILP encoding and solver helpers."""
import ast
from argparse import ArgumentParser
from pathlib import Path
import shlex
from types import SimpleNamespace
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
from fmip.utils.arg import default_args
from fmip.utils.focalloss import FocalLoss
from fmip.utils.milp_reader import MIPmodel
from fmip.utils.milp_utils import cal_objective_graph, cal_feasibility, cal_feasibility_batch, rescale_grad
from fmip.utils import lp_utils
from fmip.utils.mvb import MVB
from fmip.utils.solver_validation import binary_probabilities, validate_trust_region


def load_definitions(relative_path, names, namespace=None):
    """Load isolated entry points without starting loggers or loading models."""
    tree = ast.parse((ROOT / relative_path).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=selected, type_ignores=[]), relative_path, "exec"), namespace)
    return namespace


class EntryPointTests(unittest.TestCase):
    def test_training_scripts_use_supported_arguments(self):
        parser = load_definitions("fmip/train.py", {"arg_parser"}, {"ArgumentParser": ArgumentParser, "time": time})["arg_parser"]
        for script in (ROOT / "train_scripts").glob("*.sh"):
            with self.subTest(script=script.name):
                args = shlex.split(script.read_text().replace("\\\n", " "), comments=True)
                self.assertTrue(parser(args[2:]).do_train)

    def test_default_args_ignore_callers_cli(self):
        with patch.object(sys, "argv", ["gurobi_mvb.py", "--fixratio", "0.5"]):
            self.assertEqual(default_args().dataset_name, "load_balancing_tiny")
        self.assertEqual(default_args(["--ce_weight", "0.3", "1"]).ce_weight, [0.3, 1.0])

    def test_validation_only_does_not_train(self):
        args = default_args(["--do_valid_only"])
        trainer = MagicMock()
        trainer_constructor = MagicMock(return_value=trainer)
        logger = SimpleNamespace(save_dir="/tmp/test", name="run", version="1", _id="1")
        namespace = {
            "os": os, "torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 0, is_available=lambda: False)),
            "GMIPModel": MagicMock(), "wandb": SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "id")),
            "WandbLogger": MagicMock(return_value=logger), "rank_zero_info": MagicMock(),
            "ModelCheckpoint": MagicMock(), "LearningRateMonitor": MagicMock(),
            "TQDMProgressBar": MagicMock(), "DDPStrategy": MagicMock(), "Trainer": trainer_constructor,
        }
        main = load_definitions("fmip/train.py", {"main"}, namespace)["main"]
        main(args)
        trainer.fit.assert_not_called()
        trainer.validate.assert_called_once()
        trainer.test.assert_not_called()
        self.assertEqual(trainer_constructor.call_args.kwargs["devices"], 1)
        self.assertEqual(trainer_constructor.call_args.kwargs["strategy"], "auto")


class LossAndGuidanceTests(unittest.TestCase):
    def test_focal_loss_has_correct_gradient(self):
        logits = torch.tensor([[0.4, -0.1], [-0.2, 0.7]], dtype=torch.double, requires_grad=True)
        targets = torch.tensor([0, 1])
        loss = FocalLoss(gamma=2, alpha=0.5)
        self.assertTrue(torch.autograd.gradcheck(lambda x: loss(x, targets), (logits,)))

    def test_gradient_clipping_preserves_vector_shape(self):
        grad = torch.tensor([2.0, -2.0])
        clipped = rescale_grad(grad)
        self.assertEqual(clipped.shape, grad.shape)
        torch.testing.assert_close(clipped, torch.tensor([0.5, -0.5]))

    def test_gradient_clipping_scales_each_sample(self):
        grad = torch.tensor([[0.5, 0.5], [2.0, -2.0]])
        clipped = rescale_grad(grad)
        torch.testing.assert_close(clipped, torch.tensor([[0.5, 0.5], [0.5, -0.5]]))

    def test_discrete_objective_needs_no_continuous_nodes(self):
        class Graph(dict):
            only_discrete = True
        graph = Graph(Ivars=SimpleNamespace(x=torch.tensor([[0., 1., 2., 1.], [0., 1., 3., 1.]]), batch=torch.tensor([0, 1])), name=["a", "b"])
        self.assertEqual(cal_objective_graph(graph).item(), 5.0)
        torch.testing.assert_close(cal_objective_graph(graph, reduction=False), torch.tensor([2., 3.]))


class ReaderTests(unittest.TestCase):
    def reader(self, types, matrix, senses):
        variables = [SimpleNamespace(LB=0., UB=1., Obj=1., VType=kind, VarName=f"x{i}") for i, kind in enumerate(types)]
        constraints = [SimpleNamespace(RHS=float(i + 1), Sense=sense) for i, sense in enumerate(senses)]
        reader = MIPmodel.__new__(MIPmodel)
        reader.model = SimpleNamespace(getA=lambda: sp.csr_matrix(matrix), getVars=lambda: variables, getConstrs=lambda: constraints)
        return reader

    def test_interleaved_integer_indices_keep_coefficients(self):
        reader = self.reader([gp.GRB.CONTINUOUS, gp.GRB.BINARY, gp.GRB.INTEGER, gp.GRB.CONTINUOUS, gp.GRB.BINARY], [[1, 2, 3, 4, 5]], [gp.GRB.LESS_EQUAL])
        graph = reader.generGraphCom()
        torch.testing.assert_close(graph["I2cons_edge_index"], torch.tensor([[0, 1, 2], [0, 0, 0]]))
        torch.testing.assert_close(graph["I2cons_edge_attr"].flatten(), torch.tensor([2., 3., 5.]))

    def test_constraint_senses_become_equivalent_upper_bounds(self):
        reader = self.reader([gp.GRB.BINARY], [[2], [3], [4]], [gp.GRB.LESS_EQUAL, gp.GRB.GREATER_EQUAL, gp.GRB.EQUAL])
        graph = reader.generGraphCom()
        torch.testing.assert_close(graph["x_cons"].flatten(), torch.tensor([1., -2., 3., -3.]))
        torch.testing.assert_close(graph["I2cons_edge_attr"].flatten(), torch.tensor([2., -3., 4., -4.]))
        torch.testing.assert_close(graph["I2cons_edge_index"][1], torch.arange(4))

    def test_empty_continuous_features_have_feature_dimension(self):
        graph = self.reader([gp.GRB.BINARY], [[1]], [gp.GRB.LESS_EQUAL]).generGraphCom()
        self.assertEqual(graph["x_Cvars"].shape, (0, 6))
        self.assertEqual(graph["C2cons_edge_attr"].shape, (0, 1))
        self.assertEqual(graph["C2cons_edge_index"].shape, (2, 0))

    def test_gurobi_infinity_bounds_do_not_overflow_features(self):
        reader = self.reader([gp.GRB.CONTINUOUS], [[1]], [gp.GRB.LESS_EQUAL])
        reader.model.getVars()[0].LB = -gp.GRB.INFINITY
        reader.model.getVars()[0].UB = gp.GRB.INFINITY
        features = reader.generGraphCom()["x_Cvars"]
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[0, [0, 1, 3, 4]], torch.zeros(4))


class MathematicalUtilityTests(unittest.TestCase):
    def test_graph_feasibility_accepts_vectors_and_batched_samples(self):
        graph = {
            "Ivars": SimpleNamespace(x=torch.tensor([[0., 0.], [0., 1.]])),
            "Cvars": SimpleNamespace(x=torch.tensor([[0., 0.], [0., 1.]])),
            "cons": SimpleNamespace(x=torch.ones(2, 1), batch=torch.tensor([0, 1])),
            "name": ["a", "b"],
        }
        for kind in ["Ivars", "Cvars"]:
            graph[kind, "to", "cons"] = SimpleNamespace(edge_index=torch.tensor([[0, 1], [0, 1]]), edge_attr=torch.ones(2, 1))
        self.assertEqual(cal_feasibility(graph).item(), 1.)
        torch.testing.assert_close(cal_feasibility_batch(graph, torch.tensor([0., 1.]), torch.tensor([0., 1.]), reduction=False), torch.tensor([0., 1.]))
        torch.testing.assert_close(cal_feasibility_batch(graph, torch.tensor([[0., 1.], [2., 1.]]), torch.tensor([0., 1.]), reduction=False), torch.tensor([[0., 1.], [1., 1.]]))

    def test_lp_feasibility_detects_constraint_and_bound_violations(self):
        A = torch.tensor([[1., 1.]])
        b = torch.tensor([1.])
        lb, ub = torch.zeros(2), torch.ones(2)
        x = torch.tensor([[0., 1.], [-2., 0.], [2., 1.]])
        expected = torch.tensor([0., 2., 2.])
        torch.testing.assert_close(lp_utils.feasibility(x, A, b, lb, ub), expected)
        for row, violation in zip(x, expected):
            torch.testing.assert_close(lp_utils.feasibility(row, A.to_sparse(), b, lb, ub), violation)

    def test_basis_recovery_preserves_upper_bound_nonbasic_variables(self):
        result = lp_utils.recover_solution(torch.tensor([2, 1, 0]), torch.ones(1, 3), torch.tensor([7.]),
                                           torch.ones(3), torch.zeros(3), torch.tensor([5., 10., 10.]))
        torch.testing.assert_close(result, torch.tensor([5., 2., 0.]))

    def test_basis_selection_keeps_requested_number_of_basic_variables(self):
        np.testing.assert_array_equal(lp_utils.recover_basis_np(np.array([0.3, 0.7]), 2), [1, 1])
        np.testing.assert_array_equal(lp_utils.recover_basis_np(np.array([0.3, 0.7]), 0), [0, 0])
        logits = torch.tensor([[0., 0.9, 0.95], [0.8, 0.1, 0.1]])
        torch.testing.assert_close(lp_utils.recover_basis_logits(logits, 1), torch.tensor([1., 0.]))
        torch.testing.assert_close(lp_utils.recover_basis(logits, 1, logits=True), torch.tensor([1., 0.]))

    def test_sparse_identity_handles_uncoalesced_matrix(self):
        matrix = torch.sparse_coo_tensor(torch.tensor([[0, 0], [0, 0]]), torch.tensor([1., 2.], dtype=torch.double), (1, 2))
        result = lp_utils.cat_sparse_identity(matrix)
        torch.testing.assert_close(result.to_dense(), torch.tensor([[3., 0., 1.]], dtype=torch.double))

    def test_objective_returns_one_value_per_solution(self):
        coefficient = torch.tensor([1., 2.])
        solutions = torch.tensor([[0., 1.], [2., 3.]])
        torch.testing.assert_close(lp_utils.objective(solutions, coefficient), torch.tensor([2., 8.]))

    def test_equality_conversion_does_not_relax_existing_equalities(self):
        class Variable:
            lb, ub, vtype, varName = 0, 1, gp.GRB.CONTINUOUS, "x"
        variable = Variable()
        row = SimpleNamespace(size=lambda: 1, getCoeff=lambda i: 1, getVar=lambda i: variable, getConstant=lambda: 7)
        constraint = SimpleNamespace(Sense=gp.GRB.EQUAL, getAttr=lambda attr: 0, ConstrName="eq")
        original = SimpleNamespace(getVars=lambda: [variable], getConstrs=lambda: [constraint], getRow=lambda c: row,
                                   getObjective=lambda: row, ModelSense=gp.GRB.MINIMIZE)
        transformed = MagicMock()
        transformed.addVar.return_value = 0
        with patch.object(lp_utils, "Model", return_value=transformed):
            lp_utils.transform_to_equalities(original)
        self.assertEqual(transformed.addVar.call_count, 1)
        transformed.setObjective.assert_called_once_with(7, gp.GRB.MINIMIZE)

    def test_mvb_cardinality_ranks_variables_by_probability_of_one(self):
        optimizer = MVB(1, 4)
        probabilities = np.array([[0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        up, up_prob, low, low_prob = optimizer._getMVBIdxFromCardinality(probabilities, 1, 1)
        np.testing.assert_array_equal(up, [0])
        np.testing.assert_array_equal(low, [3])
        np.testing.assert_allclose(up_prob, [0.8])
        np.testing.assert_allclose(low_prob, [0.1])


class SolverHelperTests(unittest.TestCase):
    def test_prediction_accepts_model_with_inference_method(self):
        graph = MagicMock()
        graph.to.return_value = graph
        components = dict.fromkeys(["x_cons", "x_Cvars", "x_Ivars", "C2cons_edge_attr", "I2cons_edge_attr", "C2cons_edge_index", "I2cons_edge_index"])
        reader = SimpleNamespace(generGraphCom=lambda: components)
        predictor = SimpleNamespace(inference=MagicMock(return_value={"Ivars": "prediction"}))
        namespace = {"MIPmodel": object, "toHeteroData": MagicMock(return_value=graph)}
        predict = load_definitions("fmip/utils/gp_utils.py", {"pred_by_model"}, namespace)["pred_by_model"]
        self.assertEqual(predict(reader, predictor, device="cpu"), {"Ivars": "prediction"})
        predictor.inference.assert_called_once_with(graph, guidance=False, batchsize=1)

    def test_predsearch_budget_uses_variables_and_handles_no_incumbent(self):
        class Expression:
            def __init__(self, values):
                self.values = values
            def __le__(self, rhs):
                return self.values, rhs
        variables = [SimpleNamespace(VType=gp.GRB.BINARY, LB=0, UB=1) for _ in range(2)]
        model = MagicMock(SolCount=0)
        model.getVars.return_value = variables
        model.addVars.side_effect = lambda n, **kwargs: {i: Expression([i]) for i in range(n)}
        fake_gp = SimpleNamespace(Model=object, GRB=gp.GRB, read=lambda path: model, quicksum=lambda values: Expression(list(values)))
        namespace = {"gp": fake_gp, "np": np, "os": os, "time": time, "MIPmodel": MagicMock(), "save_history": MagicMock(),
                     "binary_probabilities": binary_probabilities, "validate_trust_region": validate_trust_region}
        cls = load_definitions("downstream/gurobi_predsearch.py", {"PredsearchOptimizer"}, namespace)["PredsearchOptimizer"]
        optimizer = cls.__new__(cls)
        optimizer.problem_root = "/unused"
        optimizer.set_threads = None
        optimizer.timelimit = 1
        optimizer.get_prediction = lambda: {"Ivars": np.array([[0.8, 0.2], [0.3, 0.7]])}
        # Both rounded counts are zero: no variable may be selected via [-0:].
        best, _ = optimizer.apply_predsearch_strategy("x.mps", [0, 0.1, 0.2])
        self.assertIsNone(best)
        self.assertEqual(model.addVars.call_args.args[0], 0)
        model.optimize.assert_called_once_with(namespace["save_history"])
        # A selected slack must appear in the budget, rather than its dict key.
        variables[0] = Expression([])
        variables[0].VType = gp.GRB.BINARY
        variables[0].LB, variables[0].UB = 0, 1
        optimizer.apply_predsearch_strategy("x.mps", [0.5, 0, 0.2])
        budget, _ = model.addConstr.call_args.args[0]
        self.assertEqual(len(budget), 1)
        self.assertIsInstance(budget[0], Expression)

    def test_saved_history_survives_clearing_work_buffers(self):
        namespace = {"gp": gp, "time_points": [1.], "iter_points": [2.], "obj_values": [3.]}
        cls = load_definitions("downstream/gurobi_predsearch.py", {"PredsearchOptimizer"}, namespace)["PredsearchOptimizer"]
        optimizer = cls.__new__(cls)
        params = [0.1, 0.2, 0.3]
        optimizer.result_dict = {str(params): {}}
        optimizer.solving_history_dict = {str(params): {}}
        optimizer.apply_predsearch_strategy = lambda *args: (None, 1.)
        optimizer.process_instance_predsearch("x", [params])
        self.assertEqual(optimizer.solving_history_dict[str(params)]["x"]["obj"], [3.])
        self.assertEqual(namespace["obj_values"], [])

    def test_apollo_fixes_integer_positions_only(self):
        cls = load_definitions("downstream/gurobi_apollo.py", {"ApolloOptimizer"}, {"gp": gp})["ApolloOptimizer"]
        optimizer = cls.__new__(cls)
        variables = [SimpleNamespace(VType=kind, LB=0, UB=1) for kind in [gp.GRB.CONTINUOUS, gp.GRB.BINARY, gp.GRB.BINARY]]
        model = SimpleNamespace(getVars=lambda: variables, update=lambda: None)
        optimizer.fix_by_ind(model, [0], [1, 0])
        self.assertEqual((variables[0].LB, variables[0].UB), (0, 1))
        self.assertEqual((variables[1].LB, variables[1].UB), (1, 1))

    def test_neural_diving_uses_integer_indices_and_original_sample_id(self):
        class Executor:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, fn, args):
                result = fn(args)
                return SimpleNamespace(result=lambda: result)
        models = []
        def read_model(path, env=None):
            variables = [SimpleNamespace(LB=0, UB=1, VType=kind)
                         for kind in (gp.GRB.CONTINUOUS, gp.GRB.BINARY, gp.GRB.BINARY)]
            model = SimpleNamespace(getVars=lambda: variables, update=lambda: None, optimize=lambda callback: None, setParam=lambda *args: None, dispose=lambda: None,
                                    SolCount=1, ObjVal=float(len(models)))
            models.append(model)
            return model
        namespace = {
            "gp": SimpleNamespace(read=read_model, GRB=gp.GRB, Env=lambda **kwargs: SimpleNamespace(dispose=lambda: None)), "np": np, "torch": torch, "time": time, "os": os,
            "concurrent": SimpleNamespace(futures=SimpleNamespace(ThreadPoolExecutor=Executor, as_completed=lambda futures: reversed(futures))),
            "time_points_dict": {}, "iter_points_dict": {}, "obj_values_dict": {},
        }
        cls = load_definitions("downstream/gurobi_neuraldiving.py", {"NeuralDivingOptimizer"}, namespace)["NeuralDivingOptimizer"]
        optimizer = cls.__new__(cls)
        optimizer.set_threads = None
        optimizer.timelimit = None
        optimizer.sense = "min"
        best, _, sample_id = optimizer.apply_sampled_strategy("x", [[1., 0.], [0., 1.]], [1, 2], sample_times=2)
        self.assertIs(best, models[0])
        self.assertEqual(sample_id, 0)
        for model in models:
            variables = model.getVars()
            self.assertEqual((variables[0].LB, variables[0].UB), (0, 1))
            self.assertEqual((variables[1].LB, variables[1].UB), (0, 0))
            self.assertEqual((variables[2].LB, variables[2].UB), (1, 1))


if __name__ == "__main__":
    unittest.main()
