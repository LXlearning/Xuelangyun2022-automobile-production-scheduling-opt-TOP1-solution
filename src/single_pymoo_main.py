import time

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymoo_main import opt_main
from utils import csv_find


@click.command()
@click.option('--name', default='data_103', help='数据集名')


def single_data_main(name, abs_path='./data'):
    """单数据集运行"""
    start_all = time.time()
    df_time = pd.DataFrame()

    paths = csv_find(f'{abs_path}/raw')
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    raw_path = f'{abs_path}/raw/{name}.csv'

    df_output1, name, hv, name_time = opt_main(raw_path, verbose=True)
    df_time.loc[name, 'time'] = name_time
    df_time.loc[name, 'hv'] = hv
    # df_time.to_csv(f'{abs_path}/output/output_hv.csv')
    print('opt finish!!!')
    print(f'times: {(time.time() - start_all):.0f}s')


if __name__ == '__main__':
    single_data_main()
