"""
Implementation of PMVB framework.
© Wenzhi Gao, gwz@stanford.edu
"""

import numpy as np
from gurobipy import *
import gurobipy as gp
# import coptpy as cp

## utils

def computeObjLoss(mvbObj, originalObj, ModelSense = -1):
    if ModelSense == -1:
        return (originalObj - mvbObj) / originalObj * 100
    elif ModelSense == 1:
        return (mvbObj - originalObj) / originalObj * 100

def get_results(m,n,ori_time,ori_warm_time,mvb_time,ori_best_time,ori_warm_best_time,TimeDominance,objLoss,objLoss_warm):

    # print(f"{'='*20} Primal Heuristic Time {instance_name} {'='*20}")
    # print(f"{'Original Time':<25}{'Original (Warm) Time':<30}{'MVB Time':<20}")
    # print(f"{ori_best_time:<25.3f}{ori_warm_best_time:<30.3f}{TimeDominance:<20.3f}")
    # print(f"{'='*63}")

    results = {
            'rows': m,
            'cols': n,
            'ori_time': ori_time,
            'ori_warm_time': ori_warm_time,
            'mvb_time': mvb_time,
            'ori_best_time': ori_best_time,
            'ori_warm_best_time': ori_warm_best_time,
            'TimeDominance': TimeDominance,
            'objLoss': objLoss,
            'objLoss_warm': objLoss_warm
        }
    return results

## MVB class

class MVB(object):
    """
    A class for multi-variable branching framework

    """
    # Learner status
    MVB_LEARNER_UNITIALIZED = 100
    MVB_LEARNER_INITIALZED = 101
    MVB_LEARNER_TRAINED = 102

    # Model status
    MVB_MODEL_UNINITIALIZED = 103
    MVB_MODEL_INITIALIZED = 104

    # Solver used
    MVB_SOLVER_GUROBI = 105
    # MVB_SOLVER_CPLEX = 106
    # MVB_SOLVER_SCIP = 107
    # MVB_SOLVER_COPT = 108

    def __init__(self, mdim, ndim):

        self._mdim = mdim
        self._ndim = ndim
        self._learner = None
        self._model = None
        self._mvbmodel = None
        self._learnerStatus = self.MVB_LEARNER_UNITIALIZED
        self._modelStatus = self.MVB_MODEL_UNINITIALIZED
        self._solver = self.MVB_SOLVER_GUROBI
        self._varIdx = list(range(ndim))
        self._vars = None
        self._getFeature = None
        self._getLearner = None
        self._predictor = None
        self._threshold = 1.0
        self._fixratio = 0.0
        self._fixLowIdx = []
        self._fixUpIdx = []
        self._mvbUpIdx = []
        self._mvbLowIdx = []
        self._ksiUp = 0
        self._ksiLow = 0
        self._tmvb = [self._threshold, 0.9]
        self._pSuccessLow = [0.95]
        self._pSuccessUp  = [0.95]
        self._nRegion = len(self._tmvb) - 1

        return

    def registerLearner(self, trainer, predictor, learner=None):

        """
        Note that for adaptivity the learner here is an abstract instance
        that supports the

        ******
        Training Method (including cross validation)

        Given training set (X, Y), the learner should support

        - learner = self._getLearner(X, y)

        and this gets the model initialized

        ******
        Predicting Method

        Given a trained learner, the learner should support

        - Xpred = self._predict(learner, feature)

        to output the probabilistic estimation of the (binary) variables
        """

        self._getLearner = trainer
        self._predictor = predictor
        self._learnerStatus = self.MVB_LEARNER_INITIALZED

        if learner is not None:
            self._learner = learner
            self._learnerStatus = self.MVB_LEARNER_TRAINED

        print("Learner has been registered")

        return

    def registerModel(self, model, solver="gurobi"):

        if solver == "gurobi":
            self._registerGurobiModel(model)
        # elif solver == "scip":
        #     self._registerSCIPModel(model)
        # elif solver == "copt":
        #     self._registerCOPTModel(model)
        else:
            self._registerSCIPModel(model)

        self._mvbmodel = self._cloneModel()
        print("Model has been registered")

        return

    def registerVars(self, varIdx):

        if self._modelStatus != self.MVB_MODEL_INITIALIZED:
            raise RuntimeError("Model is not initialized")

        if len(varIdx) > self._ndim:
            raise RuntimeError("Variable index cannot exceed the "
                               " total number of variables")
        self._varIdx = varIdx

        if self._solver == self.MVB_SOLVER_GUROBI:
            self._vars = [self._mvbmodel.getVars()[i] for i in self._varIdx]
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     self._vars = [self._mvbmodel.getVars()[i] for i in self._varIdx]
        else:   
            raise NotImplementedError("Support not yet added")

        return

    def registerFeature(self, getFeatures):

        """
        Register the extraction utility of the feature given a model

        Function getFeatures should support

        feature = getFeatures(model)

        """
        self._getFeature = getFeatures

        return

    def train(self, X, Y):

        """
        Train the registered learner

        Note that X is of shape (nSample, mdim)
         and that Y is of shape (nSample, ndim)

        """
        # Check dimension
        (nSample1, mDim) = X.shape
        (nSample2, nDim) = Y.shape

        assert (nSample1 == nSample2)
        assert (mDim == self._mdim)
        assert (nDim == self._ndim)

        if self._learnerStatus == self.MVB_LEARNER_UNITIALIZED:
            raise RuntimeError("Learner is not registered")
        elif self._modelStatus == self.MVB_MODEL_UNINITIALIZED:
            raise RuntimeError("Model is not initialized")

        self._learner = self._getLearner(X, Y)
        self._learnerStatus = self.MVB_LEARNER_TRAINED

        return
    
    def getMultiVarBranch(self, warm=False, Xpred=None, upCut=True, lowCut=True, ratio_involve=False, ratio=None):

        """
        Implement the MVB framework to solve the embedded model
        """

        if Xpred is None:
            if self._learnerStatus != self.MVB_LEARNER_TRAINED:
                raise RuntimeError("Learner is not trained")
            if self._modelStatus == self.MVB_MODEL_UNINITIALIZED:
                raise RuntimeError("Model is not initialized")

            feature = self._getFeature(self._model).reshape(-1, 1)
            Xpred = self._predictor(self._learner, feature)

        (fixUpIdx, fixLowIdx) = self._getFixIdx(Xpred, self._fixratio)
        self._fixLowIdx = fixLowIdx
        self._fixUpIdx = fixUpIdx
        print(f"- {len(fixLowIdx)} variables are fixed w.r.t. the fix ratio {self._fixratio}")

        self._fixVars(self._mvbmodel, fixUpIdx, isUpper=True)
        self._fixVars(self._mvbmodel, fixLowIdx, isUpper=False)

        for i in range(self._nRegion):
            if ratio_involve:
                assert ratio is not None
                nLow = int(len(Xpred) * ratio[0])
                nUp = int(len(Xpred) * ratio[1])
                (mvbUpIdx, mvbUpProb, mvbLowIdx, mvbLowProb) = self._getMVBIdxFromCardinality(Xpred, nLow, nUp)
            else:
                (mvbUpIdx, mvbUpProb, mvbLowIdx, mvbLowProb) = self._getMVBIdx(Xpred, self._tmvb[i + 1], self._tmvb[i], upCut)
            (ksiUp, ksiLow) = self._getHoeffdingBound(mvbUpProb, mvbLowProb, self._pSuccessLow[i], self._pSuccessUp[i])
            self._mvbUpIdx = mvbUpIdx
            self._mvbLowIdx = mvbLowIdx
            self._ksiLow = ksiLow
            self._ksiUp = ksiUp

            # Run MVB
            if upCut:
                self._addCut(mvbUpIdx, ksiUp, isGeq=True)
            if lowCut:
                self._addCut(mvbLowIdx, ksiLow, isGeq=False)

            # Warm-start
            if i == 0 and warm:
                start = np.zeros(self._ndim)
                start[mvbUpIdx] = 1.0
                start[mvbLowIdx] = 0.0
                self._setWarmStart(start)

            print("- {0} variables are involved in the MVB".format(len(mvbUpIdx) + len(mvbLowIdx)))

        print("MVB model is generated")

        return self._mvbmodel
    
    def generate_subproblem(self, model: Model, up_id, low_id, k_up, k_low, up=True, low=True, obj = None, isGeq=True, upCut=True, lowCut=True,
                            gap=0.005):

        if self._solver == self.MVB_SOLVER_GUROBI:
            vars = model.getVars()
            if self._fixratio > 0:
                self._fixVars(model, self._fixLowIdx, isUpper=False)

            if upCut and len(up_id) > 0:
                if up:
                    model.addConstr(quicksum(vars[up_id[i]] for i in range(len(up_id)))
                                    >= np.ceil(k_up), name="mvb_up")
                else:
                    model.addConstr(quicksum(vars[up_id[i]] for i in range(len(up_id)))
                                    <= np.ceil(k_up)-1, name="mvb_up_comp")
            if lowCut and len(low_id) > 0:
                if low:
                    model.addConstr(quicksum(vars[low_id[i]] for i in range(len(low_id)))
                                    <= np.floor(k_low), name="mvb_low")
                else:
                    model.addConstr(quicksum(vars[low_id[i]] for i in range(len(low_id)))
                                    >= np.floor(k_low)+1, name="mvb_low_comp")
                
            if obj is not None:
                if isGeq:
                    model.addConstr(model.getObjective() >= obj + abs(obj)*gap)
                else:
                    model.addConstr(model.getObjective() <= obj - abs(obj)*gap)

        return model

    
    def get_model_list(self, Xpred=None, obj = None, isGeq=True, upCut=True, lowCut=True, gap=0.005):
        if Xpred is None:
            if self._learnerStatus != self.MVB_LEARNER_TRAINED:
                raise RuntimeError("Learner is not trained")
            if self._modelStatus == self.MVB_MODEL_UNINITIALIZED:
                raise RuntimeError("Model is not initialized")

            feature = self._getFeature(self._model).reshape(-1, 1)
            Xpred = self._predictor(self._learner, feature)

        assert self._nRegion == 1 # ensure total possible model number is 4

        for i in range(self._nRegion):
            mvbUpIdx = self._mvbUpIdx
            mvbLowIdx = self._mvbLowIdx
            ksiLow = self._ksiLow
            ksiUp = self._ksiUp

            if upCut and lowCut:
                model_cup_low = self._model.copy()
                model_up_clow = self._model.copy()
                model_cup_clow = self._model.copy()
                model_list = [self.generate_subproblem(model_cup_low, mvbUpIdx, mvbLowIdx,
                                                ksiUp, ksiLow, up=False, low=True, obj = obj, isGeq=isGeq, upCut=upCut, lowCut=lowCut,gap=gap),
                            self.generate_subproblem(model_up_clow, mvbUpIdx, mvbLowIdx,
                                                ksiUp, ksiLow, up=True, low=False, obj = obj, isGeq=isGeq, upCut=upCut, lowCut=lowCut, gap=gap),
                            self.generate_subproblem(model_cup_clow, mvbUpIdx, mvbLowIdx,
                                                ksiUp, ksiLow, up=False, low=False, obj = obj, isGeq=isGeq, upCut=upCut, lowCut=lowCut, gap=gap)]
            elif not upCut and not lowCut:
                model_list = []
            elif upCut:
                model_list = [self.generate_subproblem(self._model.copy(), mvbUpIdx, mvbLowIdx,
                                ksiUp, ksiLow, up=False, low=True, obj = obj, isGeq=isGeq, upCut=upCut, lowCut=lowCut, gap=gap)]
            elif lowCut:
                model_list = [self.generate_subproblem(self._model.copy(), mvbUpIdx, mvbLowIdx,
                                ksiUp, ksiLow, up=True, low=False, obj = obj, isGeq=isGeq, upCut=upCut, lowCut=lowCut, gap=gap)]

        return model_list

    def getLearner(self):

        if self._learnerStatus != self.MVB_LEARNER_TRAINED:
            raise RuntimeError("Learner is not trained")

        return self._learner

    def setParam(self, fixratio, threshold, tmvb, pSuccessLow, pSuccessUp):

        # Recall that tmvb = [threshold, t1, t2, ..., tn]
        nRegion = len(tmvb) - 1
        assert (threshold == tmvb[0])
        assert (nRegion == len(pSuccessLow))
        assert (nRegion == len(pSuccessUp))

        for i in range(nRegion - 1):
            if tmvb[i] < tmvb[i + 1]:
                raise RuntimeError("tmvb must be a descending array")

        self._threshold = threshold
        self._fixratio = fixratio
        self._tmvb = tmvb
        self._pSuccessLow = pSuccessLow
        self._pSuccessUp = pSuccessUp
        self._nRegion = nRegion

        return

    def _cloneModel(self):

        if self._modelStatus != self.MVB_MODEL_INITIALIZED:
            raise RuntimeError("Model is not initialized")

        if self._solver == self.MVB_SOLVER_GUROBI:
            return self._model.copy()
        # elif self._solver == self.MVB_SOLVER_SCIP:
        #     return self._model.clone()
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     return self._model.clone()
        else:
            raise NotImplementedError("Support for SCIP solver is not yet added")

    @staticmethod
    def _getFixIdx(Xpred, threshold):

        """
        Get Fixed variable indices

        """
        nLow = int(len(Xpred) * threshold)
        nUp = int(len(Xpred) * threshold)
        nonzero_prob = Xpred[:, 1] - Xpred[:, 0]
        sorted_indices = np.argsort(nonzero_prob)
        if nLow == 0:
            fixLowIdx = np.array([]).astype(int)
        else:
            fixLowIdx = sorted_indices[:min(nLow, len(nonzero_prob))]
        if nUp == 0:
            fixUpIdx = np.array([]).astype(int)
        else:
            fixUpIdx = sorted_indices[-min(nUp, len(nonzero_prob)):]

        return fixUpIdx, fixLowIdx

    def _getMVBIdx(self, Xpred, tLow, tUp, upCut = True):

        """
        Get MVB indices

        """
        mvbUpIdx = np.where(((Xpred >= tLow) & (Xpred < tUp))[:,1])[0]
        # mvbLowIdx = np.where(((Xpred <= 1 - tLow) & (Xpred > 1 - tUp))[:,1])[0]
        # mvbLowProb = Xpred[mvbLowIdx,1]
        mvbLowIdx = np.where(((Xpred >= tLow) & (Xpred < tUp))[:,0])[0]

        diff_mvbUpIdx = np.setdiff1d(mvbUpIdx, self._fixUpIdx, assume_unique=True)
        diff_mvbLowIdx = np.setdiff1d(mvbLowIdx, self._fixLowIdx, assume_unique=True)

        mvbUpProb = Xpred[diff_mvbUpIdx,1]
        mvbLowProb = 1 - Xpred[diff_mvbLowIdx,0]

        return diff_mvbUpIdx, mvbUpProb, diff_mvbLowIdx, mvbLowProb
    
    def _getMVBIdxFromCardinality(self, Xpred, nLow, nUp):

        """
        Get MVB indices

        """
        sorted_indices = np.argsort(Xpred)
        if nUp == 0:
            mvbUpIdx = np.array([]).astype(int)
        else:
            mvbUpIdx = sorted_indices[-min(nUp, len(sorted_indices)):]
        if nLow == 0:
            mvbLowIdx = np.array([]).astype(int)
        else:
            mvbLowIdx = sorted_indices[:min(nLow, len(sorted_indices))]
        
        mvbUpIdx = np.setdiff1d(mvbUpIdx, self._fixUpIdx, assume_unique=True)
        mvbLowIdx = np.setdiff1d(mvbLowIdx, self._fixLowIdx, assume_unique=True)

        mvbUpProb = Xpred[mvbUpIdx]
        mvbLowProb = Xpred[mvbLowIdx]

        return mvbUpIdx, mvbUpProb, mvbLowIdx, mvbLowProb
    
    @staticmethod
    def _getHoeffdingBound(mvbUpProb, mvbLowProb, pSuccessLow, pSuccessUp):

        """
        Compute the tail bound based on Hoeffding's inequality

        """
        mvbUpProb = mvbUpProb.reshape(-1)
        mvbLowProb = mvbLowProb.reshape(-1)
        nUpVars = len(mvbUpProb)
        nLowVars = len(mvbLowProb)
        pFailLow = 1 - pSuccessLow
        pFailUp  = 1 - pSuccessUp
        ksiUp = np.sum(mvbUpProb) - np.sqrt(nUpVars * np.log(1 / np.maximum(pFailUp, 1e-20)) / 2)
        ksiUp = np.maximum(0,np.minimum(ksiUp, nUpVars))
        if pFailUp <= 1e-10:
            ksiUp = np.minimum(ksiUp, int(nUpVars/5))
        ksiLow = np.sum(mvbLowProb) + np.sqrt(nLowVars * np.log(1 / np.maximum(pFailLow, 1e-20)) / 2)
        ksiLow = np.maximum(ksiLow, 0)
        if pFailLow >= 0.5:
            ksiLow = 0

        return ksiUp, ksiLow

    def _addCut(self, varIdx, bound, isGeq):

        nCutVars = len(varIdx)
        
        if self._solver == self.MVB_SOLVER_GUROBI:
            if isGeq:
                self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
                                         >= np.ceil(bound), name = "mvb_up")
            else:
                self._mvbmodel.addConstr(quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
                                         <= np.floor(bound), name = "mvb_low")
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     if isGeq:
        #         self._mvbmodel.addConstr(cp.quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
        #                                  >= np.ceil(bound), name = "mvb_up")
        #     else:
        #         self._mvbmodel.addConstr(cp.quicksum(self._vars[varIdx[i]] for i in range(nCutVars))
        #                                  <= np.floor(bound), name = "mvb_low")
        else:
            raise NotImplementedError("Support not added ")
                
        return
    
    def _fixVars(self, model, varIdx, isUpper):

        nFixVars = len(varIdx)
        if self._solver == self.MVB_SOLVER_GUROBI:
            vars = [model.getVars()[i] for i in self._varIdx]
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     vars = [model.getVars()[i] for i in self._varIdx]
        else:   
            raise NotImplementedError("Support not yet added")
        
        if self._solver == self.MVB_SOLVER_GUROBI:
            if isUpper:
                model.addConstr(quicksum(vars[varIdx[i]] for i in range(nFixVars))
                                         >= nFixVars)
            else:
                model.addConstr(quicksum(vars[varIdx[i]] for i in range(nFixVars))
                                         <= 0)
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     if isUpper:
        #         model.addConstr(cp.quicksum(vars[varIdx[i]] for i in range(nFixVars))
        #                                  >= nFixVars)
        #     else:
        #         model.addConstr(cp.quicksum(vars[varIdx[i]] for i in range(nFixVars))
        #                                  <= 0)
        else:
            raise NotImplementedError("Support not added ")
        
        return
    
    def _setWarmStart(self, start):
        
        if self._solver == self.MVB_SOLVER_GUROBI:
            self._mvbmodel.setAttr(GRB.Attr.Start, self._vars, start)
        # elif self._solver == self.MVB_SOLVER_COPT:
        #     self._mvbmodel.setMipStart(self._vars, start)
        else:
            raise NotImplementedError("Support not added ")
        
        pass

    def _registerCplexModel(self, cplexModel):

        self._solver = self.MVB_SOLVER_CPLEX
        self._model = cplexModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return

    def _registerGurobiModel(self, gurobiModel):

        self._solver = self.MVB_SOLVER_GUROBI
        self._model = gurobiModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return

    def _registerSCIPModel(self, scipModel):

        self._solver = self.MVB_SOLVER_SCIP
        self._model = scipModel
        self._modelStatus = self.MVB_MODEL_INITIALIZED

        return

    # def _registerMCOPTodel(self, coptModel):

    #     self._solver = self.MVB_SOLVER_COPT
    #     self._model = coptModel
    #     self._modelStatus = self.MVB_MODEL_INITIALIZED

    #     return
