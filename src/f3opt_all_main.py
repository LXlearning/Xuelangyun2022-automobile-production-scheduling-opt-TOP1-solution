import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pymoo_main import CarOpt, get_batch_obj, opt_data, opt_f3_result
from utils import csv_find


def opt_main_f3(opt_path):
    """f3单独优化main"""
    start = time.time()
    name = opt_path.split(r'/')[-1].split('.')[0]
    abs_path = opt_path.split('/raw')[0]
    data = pd.read_csv(opt_path, index_col=0).reset_index(drop=True)
    df_output = pd.read_csv(f'{abs_path}/interim/res_{name}.csv')

    # f3 opt
    problem = CarOpt(data)
    best_idx, _ = get_batch_obj(df_output, data)
    output_best = df_output.iloc[best_idx, :]
    output_f3opt = opt_f3_result(output_best, problem, data)
    output_f3opt = output_f3opt.astype(int)

    # if verbose is False:
    output_f3opt.to_csv(f'{abs_path}/interim/res_{name}.csv', index=False)
    print('csv更新完成')
    print(f'times: {(time.time() - start):.0f}s')
    return output_f3opt


def applyParallel(paths, func):
    ret = Parallel(n_jobs=8)(delayed(func)(opt_path) for opt_path in tqdm(paths))
    return ret


if __name__ == '__main__':
    paths = csv_find(r'./data/raw')
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    output = applyParallel(paths, opt_main_f3)
    # output = opt_main_f3(paths[0])
