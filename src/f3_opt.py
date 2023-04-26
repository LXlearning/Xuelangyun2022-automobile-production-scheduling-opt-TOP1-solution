import numpy as np
from tqdm import tqdm

from my_mutation import MyInversionMutation
from obj_func import ObjFunc
from utils import df_encode


class CarOpt_f3():

    def __init__(self, data, x_base):
        self.data = df_encode(data)
        self.obj = ObjFunc()
        data_base = self.data.iloc[x_base, :].reset_index().rename(columns={'index': 'x'})
        self.f1_base, self.f2_base, _, self.f4_base, _ = self.obj.cal_obj(data_base)

    def _evaluate(self, x):
        df_x = self.data.iloc[x, :].reset_index().rename(columns={'index': 'x'})
        f1, f2, f3, f4, bad_point = self.obj.cal_obj(df_x)
        F = f3
        G = [f1 - self.f1_base, f2 - self.f2_base, f4 - self.f4_base]
        return F, G


def f3_opt_main(x, data, max_iter=30, verbose=False):
    """f3单独优化"""
    mutation = MyInversionMutation(df_encode(data))
    x_base = x.copy()
    problem_f3 = CarOpt_f3(data, x_base)
    F_base, _ = problem_f3._evaluate(x_base)
    X, F_best = [x_base], [F_base]

    iter_i = tqdm(range(max_iter)) if verbose else range(max_iter)
    for i in iter_i:
        x_tmp = mutation._do('', [X[-1]])[0]
        F, G = problem_f3._evaluate(x_tmp)
        if (F <= np.min(F_best)) & (np.max(G) <= 0):
            # 优化有进展
            F_best.append(F)
            X.append(x_tmp)
        else:
            F_best.append(F_best[-1])
            X.append(X[-1])
    return X[-1], F_best
