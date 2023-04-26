import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from tqdm import tqdm

from f3_opt import f3_opt_main
from obj_func import ObjFunc
from obj_strategy import init_x
from pymoo_algorithm import algorithm_choose
from utils import csv_find, df_encode


class CarOpt(ElementwiseProblem):

    def __init__(self, data, print_flag=0, **kwargs):

        n_var = data.shape[0]
        self.df = df_encode(data)
        self.print_flag = print_flag
        self.obj = ObjFunc(print_flag=print_flag)
        self.f_baseline = self.obj.cal_min_max(self.df)

        super(CarOpt, self).__init__(n_var=n_var,
                                     n_obj=4,
                                     xl=0,
                                     xu=n_var,
                                     vtype=int,
                                     **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        df_x = self.df.iloc[x, :].reset_index().rename(columns={'index': 'x'})
        f1, f2, f3, f4, bad_point = self.obj.cal_obj(df_x)

        if self.print_flag == 1:
            print('原始响应:', f1, f2, f3, f4)
        f1 = (f1 - self.f_baseline[0][0]) / (self.f_baseline[0][1] - self.f_baseline[0][0])
        f2 = (f2 - self.f_baseline[1][0]) / (self.f_baseline[1][1] - self.f_baseline[1][0])
        f3 = (f3 - self.f_baseline[2][0]) / (self.f_baseline[2][1] - self.f_baseline[2][0])
        f4 = (f4 - self.f_baseline[3][0]) / (self.f_baseline[3][1] - self.f_baseline[3][0])
        out["F"] = [f1, f2, f3, f4]
        out['bad_point'] = bad_point
        return out


def get_batch_obj(output, data):
    """获取结果序列的目标值"""
    problem = CarOpt(data)
    obj_cols = ['Objective 1', 'Objective 2', 'Objective 3', 'Objective 4']
    df1 = pd.DataFrame(columns=obj_cols)
    cols = [f'Variable {i}' for i in range(1, data.shape[0]+1)]
    for i, row in output[cols].iterrows():
        df1.loc[i, obj_cols] = problem._evaluate(row, {})['F']
    df1 = df1.reset_index().drop_duplicates(obj_cols, keep='first')
    best_idx = df1['index'].values
    return best_idx, df1


def opt_f3_result(output, problem, data):
    cols = [f'Variable {i}' for i in range(1, data.shape[0]+1)]
    output_f3opt = pd.DataFrame(columns=cols)
    for i, row in output[cols].iterrows():
        if problem._evaluate(row, {})['F'][2] > 0:
            output_f3opt.loc[i, cols] = f3_opt_main(row.values, data)[0]
        else:
            output_f3opt.loc[i, cols] = row.values
    return output_f3opt


class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.f1 = Column("f1", width=13)
        self.f2 = Column("f2", width=13)
        self.f3 = Column("f3", width=13)
        self.f4 = Column("f4", width=13)
        self.hv = Column("hv", width=13)
        self.columns += [self.f1, self.f2, self.f3, self.f4, self.hv]
        self.ind = HV(ref_point=np.array([1, 1, 1, 1]))

    def update(self, algorithm):
        super().update(algorithm)
        self.f1.set(np.round(np.mean(algorithm.pop.get("F")[:, 0].mean()), 4))
        self.f2.set(np.round(np.mean(algorithm.pop.get("F")[:, 1].mean()), 4))
        self.f3.set(np.round(np.mean(algorithm.pop.get("F")[:, 2].mean()), 4))
        self.f4.set(np.round(np.mean(algorithm.pop.get("F")[:, 3].mean()), 4))
        self.hv.set(self.ind(algorithm.result().F))


def opt_data(data, name, baseline=1, baseline_path=None, verbose=False):
    start = time.time()
    print('='*20, name, '='*20)
    problem = CarOpt(data)

    # 自定义初始种群
    pop_size = 40
    n_gen    = 200
    seed     = 48

    if baseline == 1:
        X = init_x(data, pop_size=pop_size)
    elif baseline == 2:
        df_res = pd.read_csv(baseline_path)
        cols = [f'Variable {i}' for i in range(1, data.shape[0]+1)]
        X = df_res[cols].values

    algorithm = algorithm_choose(X, data, case='SMSEMOA', pop_size=pop_size)

    res = minimize(problem,
                   algorithm,
                   termination=("n_gen", n_gen),  # termination,
                   output=MyOutput(),
                   seed=seed,
                   verbose=verbose)

    # print('-----优化完成-----')
    print('帕累托解数量:', res.X.shape)
    df_output = pd.DataFrame(res.X, columns=[f'Variable {i}' for i in range(1, res.X.shape[1]+1)])
    df_output[['Objective 1', 'Objective 2', 'Objective 3', 'Objective 4']] = np.round(res.F, 4)
    ind = HV(ref_point=np.array([1, 1, 1, 1]))
    hv = ind(res.F)
    print('优化结果:', df_output[['Objective 1', 'Objective 2', 'Objective 3', 'Objective 4']])
    print('HV:', np.round(hv, 7))

    # # f3第二轮优化
    best_idx, _ = get_batch_obj(df_output, data)
    output_best = df_output.iloc[best_idx, :]
    output_f3opt = opt_f3_result(output_best, problem, data)
    output_f3opt = output_f3opt.astype(int)

    # df_output = df_output.head(50)
    code_time = int(time.time() - start)
    print(f'times: {(time.time() - start):.0f}s')
    return output_f3opt, name, hv, code_time


def opt_main(opt_path, baseline=1, baseline_path=None, verbose=False):
    name = opt_path.split(r'/')[-1].split('.')[0]
    data = pd.read_csv(opt_path, index_col=0).reset_index(drop=True)
    df_output, name, hv, code_time = opt_data(data, name, baseline=baseline, baseline_path=baseline_path, verbose=verbose)

    # if verbose is False:
    abs_path = opt_path.split('/raw')[0]
    df_output.to_csv(f'{abs_path}/interim/res_{name}.csv', index=False)
    print('csv更新完成')
    return df_output, name, hv, code_time


def applyParallel(paths, func):
    ret = Parallel(n_jobs=8)(delayed(func)(opt_path) for opt_path in tqdm(paths))
    return ret


if __name__ == '__main__':
    start_all = time.time()
    abs_path = './data'
    paths = csv_find(abs_path + r'/raw')
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    output = applyParallel(paths[:4], opt_main)

    print('opt finish!!!')
    print(f'总时长: {(time.time() - start_all):.0f}s')

    # save result
    currentTime = datetime.now().strftime(r"%m-%d_%H-%M")
    output_path = abs_path + rf'/output/output_baseline_batch_{currentTime}.xlsx'

    df_res = pd.DataFrame()
    for df, name, hv, code_time in output:
        df_res.loc[name, 'time'] = code_time
        df_res.loc[name, 'HV'] = hv
    df_res.to_csv(abs_path + rf'/output/output_hv_{currentTime}.csv')

    if os.path.isfile(output_path):
        with pd.ExcelWriter(output_path, mode='a') as writer:
            for df, name, hv, _ in output:
                df.to_excel(writer, sheet_name=name, index=False)
    else:
        with pd.ExcelWriter(output_path) as writer:
            for df, name, hv, _ in output:
                df.to_excel(writer, sheet_name=name, index=False)
