import random
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import df_encode, insert_same_color, insert_same_color_tail

warnings.filterwarnings('ignore')


def init_case_f2(data, color_best, seed=1, f3_flag=True, return_index=True, reverse=False, n1=1,
                 f4_flag=False, f2_flag=False, color_best_flag=True):
    """f2最优解初始策略: 颜色相同的车连续排列(按最优车型&四驱方式排列), 颜色不同的车按最优车型&四驱方式排列

    Parameters
    ----------
    data : DataFrame
    color_best : list
        焊接切换位置的最佳切换颜色
    seed : int, optional
        随机种子, by default 1
    f3_flag : bool, optional
        插入单台车时为四驱车时抽两台车, by default True
    return_index : bool, optional
        返回DataFrame or index, by default True
    reverse : bool, optional
        无对比颜色是否排在序列前, by default False
    n1 : int, optional
        f3排序时随机插入其他车辆数量, by default 1
    f4_flag : bool, optional
        是否启用f4最优策略, by default False
    f2_flag : bool, optional
        末尾不足5台车是否允许插入新车辆到头部, by default False
    color_best_flag : bool, optional
        没什么用的参数,懒得删了, by default True
    """
    data_encode = df_encode(data)
    data1 = data[data_encode.apply(lambda x: (x['车身颜色'] == x['车顶颜色']), axis=1)]
    data2 = data[data_encode.apply(lambda x: (x['车身颜色'] != x['车顶颜色']), axis=1)]
    if color_best_flag:
        color_best_0 = color_best[0]
    else:
        color_best_0 = color_best[-1]
    # 判断是否需要排列
    if len(data1) > 0:
        data_encode1 = df_encode(data1)
        colors = data_encode1['车身颜色'].unique().tolist()
        # color_best放在开头or结尾
        if reverse:
            colors.insert(0, colors.pop(colors.index(color_best_0)))
        else:
            colors.append(colors.pop(colors.index(color_best_0)))
    else:
        data_encode1 = pd.DataFrame()
        colors = []

    if len(data2) > 0:
        data_encode2 = df_encode(data2)
        data2_2wd = data_encode2.sort_values(['变速器']).copy()
    else:
        data_encode2 = pd.DataFrame()
        data2_2wd = pd.DataFrame()

    # 颜色一样的车连续5台排列
    data_f2 = pd.DataFrame()
    for c in colors:
        data1_color = data1[data_encode1['车身颜色'] == c].sort_values(['车型', '变速器'], ascending=False)

        # 车型切换位置trick: color_best车辆提出来单独插入
        if (c == color_best_0) & (reverse):
            num = (len(data1_color) // 5) * 5
            data1_color_other = data1_color.iloc[num:, :]
            data1_color = data1_color.head(num)
        else:
            data1_color_other = pd.DataFrame()

        data1_color_all = pd.DataFrame()
        for i, j in zip(['A', 'B'], [0, 1]):
            data1_color_f1 = data1_color[data1_color['车型'] == i]
            data1_color_new = init_case_f3(data1_color_f1, seed=seed, return_index=False, n1=n1)

            data2_2wd_f1 = data2_2wd[data2_2wd['车型'] == j]
            if len(data2_2wd) > 0:
                data2_4wd = data2_2wd_f1[(data2_2wd_f1['车身颜色'] != c) & ((data2_2wd_f1['变速器'] == 1))]
                data2_4wd = data2_4wd[~data2_4wd.index.isin(data_f2.index)]

                # 每5台颜色一致的车前插入1台有对比颜色的车，车身颜色和前者一致(节省油漆时间)
                data2_color2_1 = data2_2wd_f1[(data2_2wd_f1['车身颜色'] == c)]
                if (len(data2_color2_1) > 0):
                    data1_color_new = insert_same_color(data2_color2_1, data1_color_new, data2_4wd, type='head',
                                                        f3_flag=f3_flag, f2_flag=f2_flag)

                # 插入1台有对比颜色的车，车顶颜色和前者一致(节省油漆时间)
                data2_color2_2 = data2_2wd_f1[data2_2wd_f1['车顶颜色'] == c]
                data2_4wd_2 = data2_4wd[(~data2_4wd.index.isin(data1_color_new.index)) & (~data2_4wd.index.isin(data2_color2_2.index))]

                # case四驱车很少
                if data1_color_new['变速器'].sum() == 1:
                    data1_color_new = data1_color_new.sort_values('变速器')
                    data1_color_new = pd.concat([data1_color_new, data2_4wd_2.head(2)])
                    data2_4wd_2 = data2_4wd_2.iloc[2:, :]

                # A车型剩余数量 & B车型需要数量
                num_b = 5 - (len(data1_color) - len(data1_color_f1)) % 5
                num_a = len(data1_color_f1) % 5
                if f4_flag:
                    # 末尾可插入
                    if num_a > 0:
                        df_tmp = data2_color2_2.head(1)
                        data1_color_new = pd.concat([data1_color_new, df_tmp])
                        data2_color2_2.drop(index=df_tmp.index, axis=0, inplace=True)
                    if (len(data2_color2_2) > 0):
                        data1_color_new = insert_same_color(data2_color2_2, data1_color_new, data2_4wd_2, type='tail',
                                                            f3_flag=f3_flag, num=4)
                else:
                    if (len(data2_color2_2) > 0) & (len(data1_color_new) % 5 != 0) & (j == 0) & (num_a > num_b):
                        # 只有A车型剩余数量 > B车型需要数量时才插入车辆
                        data1_color_new = insert_same_color_tail(data2_color2_2, data1_color_new, data2_4wd_2, num_b)
            data1_color_all = pd.concat([data1_color_all, data1_color_new])

            # 插入至23辆车 0213update
            data1_count = data1_color_all.groupby('车型')['车身颜色'].count().reset_index()
            data2 = data2[~data2.index.isin(data1_color_all.index)]
            data_encode2 = data_encode2[~data_encode2.index.isin(data1_color_all.index)]
            data2_2wd = data2_2wd[~data2_2wd.index.isin(data1_color_all.index)]

            if len(data1_count) > 1:
                data1_count_f1 = data1_count[data1_count['车身颜色'] < 23]
                for i, df_row in data1_count_f1.iterrows():
                    data2_tmp = data2[data_encode2['车型'] == df_row['车型']]
                    data_encode2_tmp = init_case_f3(data2_tmp.head(23-df_row['车身颜色']), seed=seed, return_index=False, n1=n1)
                    if df_row['车型'] == 0:
                        data1_color_all = pd.concat([data_encode2_tmp, data1_color_all])
                    else:
                        data1_color_all = pd.concat([data1_color_all, data_encode2_tmp])
                    data2.drop(index=data_encode2_tmp.index, axis=0, inplace=True)
                    data_encode2.drop(index=data_encode2_tmp.index, axis=0, inplace=True)
                    data2_2wd.drop(index=data_encode2_tmp.index, axis=0, inplace=True)

        # 拼接成新序列
        data_f2 = pd.concat([data1_color_other, data_f2, data1_color_all])

    # 按最优四驱方式插入剩下不同颜色的车
    data2 = data2[~data2.index.isin(data1_color_all.index)]
    if len(data2) > 0:
        for i in range(2):
            data2_tmp = data2[data_encode2['车型'] == i]
            if len(data2_tmp) > 0:
                data_encode2_new = init_case_f3(data2_tmp, seed=seed, return_index=False, n1=n1)
                if i == 0:
                    data_f2 = pd.concat([data_encode2_new, data_f2])
                else:
                    data_f2 = pd.concat([data_f2, data_encode2_new])
    assert np.sum(data_f2.index) == np.sum(data.index)
    if return_index:
        return list(data_f2.index)
    else:
        return data_f2


def init_case_f3(data, seed=1, return_index=True, n1=1):
    """f3最优解初始策略(随机2-3辆插入4驱车)"""
    if len(data) == 0:
        if return_index:
            return list(data.index)
        else:
            return data
    data_encode = df_encode(data)
    data_encode1 = data_encode[data_encode['变速器'] == 0]
    data_encode2 = data_encode[data_encode['变速器'] == 1]

    # 变速器0无数据时直接不用排序了
    if len(data_encode1) == 0:
        if return_index:
            return list(data_encode.index)
        else:
            return data_encode

    data_f3 = pd.DataFrame()
    i = 0
    while (len(data_encode2) > 1) & (len(data_encode1) > 0):
        n = n1 if len(data_encode1) >= n1+1 else len(data_encode1)
        data_tmp1 = data_encode1.head(n)
        data_encode1.drop(index=data_tmp1.index, axis=0, inplace=True)

        n0 = 2
        data_tmp2 = data_encode2.sample(n0, random_state=seed+i)
        data_encode2.drop(index=data_tmp2.index, axis=0, inplace=True)
        # 四驱车2-3辆插入
        data_f3 = pd.concat([data_f3, data_tmp2, data_tmp1])
        i += 1
    # 拼接剩余的车辆
    if (data_encode['车型'].nunique() == 1) & (data_encode['车型'].max() == 1):
        data_f3 = pd.concat([data_encode2, data_encode1, data_f3])
    else:
        data_f3 = pd.concat([data_f3, data_encode1, data_encode2])
    if return_index:
        return list(data_f3.index)
    else:
        return data_f3


def init_case_f1(data, min_func='f2', f3_flag=True, f4_flag=False, n1=1, seed=1, color_best_flag=True):
    """f1+f2最小化策略, f1+f3最小化策略, f3_flag True时只优化二驱车时间
    """
    data_encode = df_encode(data)
    data1 = data[data_encode['车型'] == 0]
    data2 = data[data_encode['车型'] == 1]

    data_new = pd.DataFrame()
    color_best = best_color_for_f1(data_encode)
    if min_func == 'f2':
        # 先按车型排, 再把车型相同的颜色一样的尽量5台排放, 最后将车型相同颜色不一致的四驱车按要求排
        if len(data1) > 0:
            data_tmp1 = init_case_f2(data1, color_best=color_best,
                                     seed=seed, f3_flag=f3_flag, return_index=False, reverse=False, n1=n1, f4_flag=f4_flag,
                                     color_best_flag=color_best_flag)
            assert np.sum(data1.index) == np.sum(data_tmp1.index)
        else:
            data_tmp1 = pd.DataFrame()
        if len(data2) > 0:
            data_tmp2 = init_case_f2(data2, color_best=color_best,
                                     seed=seed, f3_flag=f3_flag, return_index=False, reverse=True, n1=n1, f4_flag=f4_flag,
                                     color_best_flag=color_best_flag)
            assert np.sum(data2.index) == np.sum(data_tmp2.index)
        else:
            data_tmp2 = pd.DataFrame()

    elif min_func == 'f3':
        # 先按车型排, 再把车型相同的四驱车按要求排
        if len(data1) > 0:
            data_tmp1 = init_case_f3(data1, seed=seed, return_index=False)
        else:
            data_tmp1 = pd.DataFrame()
        if len(data2) > 0:
            data_tmp2 = init_case_f3(data2, seed=seed, return_index=False)
        else:
            data_tmp2 = pd.DataFrame()

    data_new = pd.concat([data_tmp1, data_tmp2])
    return list(data_new.index)


def best_color_for_f1(data_encode, f2_flag=False):
    """获取车型切换时的最佳车身颜色"""
    data_encode1 = data_encode[data_encode.apply(lambda x: (x['车身颜色'] == x['车顶颜色']), axis=1)]
    data_encode1_0 = data_encode1[data_encode1['车型'] == 0]
    data_encode1_1 = data_encode1[data_encode1['车型'] == 1]
    df_color0 = data_encode1_0['车身颜色'].value_counts().reset_index()
    df_color1 = data_encode1_1['车身颜色'].value_counts().reset_index()
    df_color0['车型0'] = df_color0['车身颜色'] % 5
    df_color1['车型1'] = df_color1['车身颜色'] % 5
    df_color = df_color0.merge(df_color1, on='index', how='inner')

    df_color_best = df_color[(df_color['车型0'] > 0) & (df_color['车型1'] > 0) & (df_color[['车型0', '车型1']].sum(axis=1) <= 5)]
    color_best = df_color_best['index'].tolist() if len(df_color_best) > 0 else [df_color['index'].iloc[0]]
    return color_best


def init_x(data, pop_size=50, seed=1):
    """初始策略"""
    X = []
    n = 2
    data_encode = df_encode(data)
    color_best = best_color_for_f1(data_encode)
    for i in range(n):
        for j in range(1, 4):
            X.append(np.array(init_case_f2(data, color_best, n1=j, seed=seed+i, f3_flag=True, f2_flag=True, f4_flag=False)))
            X.append(np.array(init_case_f2(data, color_best, n1=j, seed=seed+i, f3_flag=True, f2_flag=True, f4_flag=True)))
            X.append(np.array(init_case_f1(data, min_func='f2', n1=j, seed=seed+i, f4_flag=False)))
            X.append(np.array(init_case_f1(data, min_func='f2', n1=j, seed=seed+i, f4_flag=True)))
            X.append(np.array(init_case_f3(data, seed=seed+i, n1=j)))

    X = np.array(X)

    if X.min() < 0 or X.max() >= len(data):
        print('初始策略错误')
        sys.exit(0)
    return X


if __name__ == '__main__':
    name = 'data_184'
    data = pd.read_csv(f'./data/raw/{name}.csv', index_col=0)
    X = init_x(data, pop_size=50, seed=1)
    print(X)
    # from pymoo_main import CarOpt
    # problem = CarOpt(data)
    # df_output = pd.DataFrame()
    # df1 = pd.DataFrame(columns=['Objective 1', 'Objective 2', 'Objective 3', 'Objective 4'])

    # for i, x in tqdm(enumerate(X)):
    #     df1.loc[i, :] = problem._evaluate(x, {})['F']
    # print(df1)
