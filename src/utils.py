import os
import sys

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV


def csv_find(parent_path, file_flag='.csv', all_dirs=False):
    '''批处理读csv文件位置'''
    df_paths = []
    if all_dirs:
        for root, dirs, files in os.walk(parent_path):  # os.walk输出[目录路径，子文件夹，子文件]
            for file in files:
                if "".join(file).find(file_flag) != -1:  # 判断是否csv文件(files是list格式，需要转成str)
                    df_paths.append(root + '/' + file)  # 所有csv文件
    else:
        for file in os.listdir(parent_path):
            if "".join(file).find(file_flag) != -1:
                df_paths.append(parent_path + '/' + file)
    return df_paths


# def cate_encode(paths):
#     df_all = pd.DataFrame()
#     for path in paths:
#         data0 = pd.read_csv(path, index_col=0)
#         df_all = pd.concat([df_all, data0])

#     df_all['车顶颜色'] = df_all.apply(lambda x: x['车身颜色'] if x['车顶颜色']=='无对比颜色' else x['车顶颜色'], axis=1)
#     print('df_all:', df_all.shape)
#     # 类别特征编码
#     df_code_dict = {col: {cate: code for code,cate in enumerate(df_all[col].astype('category').cat.categories)} 
#                             for col in df_all.columns}
#     return df_code_dict


def df_encode(data):
    df_code_dict = {'车型': {'A': 0, 'B': 1},
                    '车身颜色': {'天空灰': 0,
                                '探索绿': 1,
                                '明亮红': 2,
                                '水晶珍珠白': 3,
                                '水晶紫': 4,
                                '液态灰': 5,
                                '薄雾灰': 6,
                                '闪耀黑': 7,
                                '飞行蓝': 8},
                    '车顶颜色': {'天空灰': 0,
                                '探索绿': 1,
                                '明亮红': 2,
                                '水晶珍珠白': 3,
                                '水晶紫': 4,
                                '液态灰': 5,
                                '薄雾灰': 6,
                                '闪耀黑': 7,
                                '飞行蓝': 8,
                                '石黑': 99},
                    '变速器': {'两驱': 0, '四驱': 1}}
    df = data.copy()
    df['车顶颜色'] = df.apply(lambda x: x['车身颜色'] if x['车顶颜色']=='无对比颜色' else x['车顶颜色'], axis=1)
    for col in df_code_dict.keys():
        df[col] = df[col].apply(lambda x: df_code_dict[col][x])
    return df


def hv_cal(data, df_output, problem):
    """单个data序列的hv计算"""
    cols = [f'Variable {i}' for i in range(1, data.shape[0]+1)]
    pareto_front = []
    for i, x in df_output[cols].iterrows():
        out = {}
        out = problem._evaluate(x.values, out)
        pareto_front.append(out['F'])
    pareto_front = np.array(pareto_front)
    ind = HV(ref_point=np.array([1, 1, 1, 1]))
    hv = ind(pareto_front)
    return hv


def insert_same_color(df1, df2, df3, type='head', f3_flag=False, num=5, f2_flag=False):
    """头部or末端随机插入一台有对比颜色车辆"""
    df_new = pd.DataFrame()
    if f2_flag:
        n = 4 if df2['车型'].max() == 1 else 0
    else:
        n = 0
    while (len(df1) > 0) & (len(df2) > n):
        # 随机抽1台颜色不同的车
        data_tmp1 = df1.head(1)
        df1.drop(index=data_tmp1.index, axis=0, inplace=True)

        # 四驱车时抽两台车
        if type == 'head':
            if len(data_tmp1[data_tmp1['变速器'] == 1]) > 0:
                # f3_flag==True时, df3不足从df1中抽四驱车
                if (f3_flag) & (len(df3) == 0):
                    data_tmp3 = df1.head(1)
                    df1.drop(index=data_tmp3.index, axis=0, inplace=True)
                else:
                    data_tmp3 = df3.head(1)
                    df3.drop(index=data_tmp3.index, axis=0, inplace=True)
                data_tmp1 = pd.concat([data_tmp3, data_tmp1])

        # 随机抽5台不同的车
        data_tmp2 = df2.head(num)
        df2.drop(index=data_tmp2.index, axis=0, inplace=True)
        if type == 'head':
            df_new = pd.concat([df_new, data_tmp1, data_tmp2])
        elif type == 'tail':
            df_new = pd.concat([df_new, data_tmp2, data_tmp1])
        else:
            print('error insert_same_color')
            sys.exit(0)

    # 如果df1不足，插入剩余的df2
    if df2['车型'].max() == 1:
        df_new = pd.concat([df2, df_new])
    else:
        df_new = pd.concat([df_new, df2])
    return df_new


def insert_same_color_tail(df1, df2, df3, num=0):
    """插入一台有对比颜色车辆,num为插入位置"""
    df_new = pd.DataFrame()

    # 随机抽1台颜色不同的车
    data_tmp1 = df1.head(1)
    df1.drop(index=data_tmp1.index, axis=0, inplace=True)

    # 四驱车时抽两台车
    if (len(data_tmp1[data_tmp1['变速器'] == 1]) > 0) & (df2.iloc[-1]['变速器'] != 1):
        # f3_flag==True时, df3不足从df1中抽四驱车
        if (len(df3) == 0):
            data_tmp3 = df1.head(1)
            df1.drop(index=data_tmp3.index, axis=0, inplace=True)
        else:
            data_tmp3 = df3.head(1)
            df3.drop(index=data_tmp3.index, axis=0, inplace=True)
        data_tmp1 = pd.concat([data_tmp1, data_tmp3])
    df2_1 = df2.head(len(df2) - num)
    df2_2 = df2.tail(num)
    df_new = pd.concat([df2_1, data_tmp1, df2_2])
    return df_new
