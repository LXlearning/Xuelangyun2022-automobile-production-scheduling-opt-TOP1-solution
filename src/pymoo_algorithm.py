import numpy as np
# from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.operators.crossover.ox import OrderCrossover, ox
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.util.ref_dirs import get_reference_directions

from my_mutation import MyInversionMutation, random_sequence
from obj_strategy import init_case_f1, init_case_f2, init_case_f3
from utils import df_encode


def find_f1_wrong(df_tmp):
    f1_wrong_inx = df_tmp[abs(df_tmp['车型'].diff()) == 1].index.tolist()
    return f1_wrong_inx


def find_f2_wrong(df_tmp):
    bad_point1 = df_tmp[abs(df_tmp['车身颜色'].diff()) != 0].index.tolist()
    bad_point2 = df_tmp[(abs(df_tmp['车顶颜色'].diff()) != 0)].index.tolist()
    bad_point = list(set(bad_point1) | set(bad_point2))
    return bad_point


class MyOrderCrossover(Crossover):

    def __init__(self, data_encode, shift=False, prob=1, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift
        self.prob = prob
        self.data = data_encode

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)
        prob = np.random.random()
        for i in range(n_matings):
            if prob < 0.3: # self.prob:
                a, b = X[:, i, :]

                # define the sequence to be used for crossover
                df_tmp = self.data.iloc[b, :].reset_index()
                bad_point = find_f1_wrong(df_tmp)
                if len(bad_point) > 1:
                    idx1 = np.random.randint(len(bad_point)-1)
                    start, end = bad_point[idx1:idx1+2]
                    end -= 1
                else:
                    start, end = random_sequence(len(a))
                Y[0, i, :] = ox(a, b, seq=(start, end), shift=self.shift)

                df_tmp2 = self.data.iloc[a, :].reset_index()
                bad_point = find_f1_wrong(df_tmp2)
                if len(bad_point) > 1:
                    idx1 = np.random.randint(len(bad_point)-1)
                    start, end = bad_point[idx1:idx1+2]
                    end -= 1
                else:
                    start, end = random_sequence(len(a))
                Y[1, i, :] = ox(b, a, seq=(start, end), shift=self.shift)
            elif prob > 0.6:
                a, b = X[:, i, :]
                df_tmp = self.data.iloc[b, :].reset_index()
                bad_point = find_f2_wrong(df_tmp)
                start, end = random_sequence(bad_point)
                end -= 1
                Y[0, i, :] = ox(a, b, seq=(start, end), shift=self.shift)

                df_tmp = self.data.iloc[a, :].reset_index()
                bad_point = find_f2_wrong(df_tmp)
                start, end = random_sequence(bad_point)
                end -= 1
                Y[1, i, :] = ox(b, a, seq=(start, end), shift=self.shift)
            else:
                Y[0, i, :] = X[0, i, :]
                Y[1, i, :] = X[1, i, :]
        return Y


class MyOrderCrossover2(Crossover):

    def __init__(self, prob=0.5, shift=False, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.shift = shift
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for i in range(n_matings):
            if np.random.random() < self.prob:
                a, b = X[:, i, :]
                n = len(a)

                # define the sequence to be used for crossover
                start, end = random_sequence(n)

                Y[0, i, :] = ox(a, b, seq=(start, end), shift=self.shift)
                Y[1, i, :] = ox(b, a, seq=(start, end), shift=self.shift)
            else:
                Y[0, i, :] = X[0, i, :]
                Y[1, i, :] = X[1, i, :]

        return Y


def algorithm_choose(X, data, case='NSGA3', pop_size=50):
    """算法选择"""
    data_encode = df_encode(data)
    # create the reference directions to be used for the optimization
    # ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=12)
    pop = Population.new("X", X)
    if case == 'NSGA3':
        # An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting
        # approach, part I: solving problems with box constraints, 2014
        ref_dirs = get_reference_directions("energy", 4, pop_size, seed=1)
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            eliminate_duplicates=True,
            sampling=pop,
            mutation=MyInversionMutation(data_encode),
            crossover=MyOrderCrossover(data_encode),
        )
    elif case == 'SMSEMOA':
        # Sms-emoa: multiobjective selection based on dominated hypervolume, 2007
        algorithm = SMSEMOA(pop_size=pop_size,
                            eliminate_duplicates=True,
                            sampling=pop,
                            mutation=MyInversionMutation(data_encode, opt1=True),
                            crossover=MyOrderCrossover(data_encode),
                            survival=RankAndCrowdingSurvival())
    # elif case == 'AGEMOEA':
    #     # An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective optimization, 2019
    #     algorithm = AGEMOEA(pop_size=pop_size,
    #                         eliminate_duplicates=True,
    #                         sampling=pop,
    #                         mutation=InversionMutation(),
    #                         crossover=OrderCrossover())
    elif case == 'UNSGA3':
        # RNSGA3: reference point based NSGA-III for preferred solutions, 2018
        # A unified evolutionary optimization procedure for single, multiple, and many objectives
        ref_dirs = get_reference_directions("energy", 4, pop_size, seed=1)
        algorithm = UNSGA3(ref_dirs=ref_dirs,
                           pop_size=pop_size,
                           eliminate_duplicates=True,
                           sampling=pop,
                           mutation=MyInversionMutation(prob=0.4),
                           crossover=OrderCrossover())
    elif case == 'CTAEA':
        ref_dirs = get_reference_directions("energy", 4, pop_size, seed=1)
        algorithm = CTAEA(ref_dirs=ref_dirs,
                          eliminate_duplicates=True,
                          sampling=pop,
                          mutation=InversionMutation(),
                          crossover=OrderCrossover())
    print('优化策略:', case)
    return algorithm
