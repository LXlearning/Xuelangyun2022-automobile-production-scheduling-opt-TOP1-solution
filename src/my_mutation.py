import sys

import numpy as np
from pymoo.core.mutation import Mutation


def find_f3_wrong(df_tmp):
    """f3 错误位置"""
    df_tmp['roll3'] = df_tmp['变速器'].rolling(window=3, center=True).mean()

    # 找到可交换的最佳的二驱车位置
    df_tmp['roll3'] = df_tmp['变速器'].rolling(window=3, center=True).mean()
    df_tmp['roll2_u'] = df_tmp['变速器'].rolling(window=2, center=True).mean()
    df_tmp['roll2_d'] = df_tmp['变速器'].rolling(window=2, center=True).mean().shift(-1)
    df_tmp['roll3_u'] = df_tmp['变速器'].rolling(window=3).mean()
    df_tmp['roll3_d'] = df_tmp['变速器'].rolling(window=3).mean().shift(-2)
    df_tmp['roll5'] = df_tmp['变速器'].rolling(window=5, center=True).mean()
    df_tmp1 = df_tmp[(df_tmp['变速器'] == 1)]
    bad_point1 = df_tmp1[(df_tmp1['roll3'] < 0.5)].index.tolist()
    bad_point2 = (df_tmp1[(df_tmp1['roll3'] == 1)].index - 1).tolist()
    bad_point3 = (df_tmp1[(df_tmp1['roll3'] == 1)].index + 1).tolist()
    # 当前排序的序列号
    bad_point = bad_point1 + bad_point2 + bad_point3
    # 真实idx车号
    bad_point_idx = df_tmp.iloc[bad_point, :]['index'].tolist()

    # 可交换的最优二驱车车号
    df_tmp0 = df_tmp[(df_tmp['变速器'] == 0)]
    idx1 = df_tmp0[(abs(df_tmp0['roll2_u']-0.5)<0.1) & (abs(df_tmp0['roll3_u']-0.3)<0.1) &
                   (abs(df_tmp0['roll3']-0.3)<0.1)].index.tolist()
    idx2 = df_tmp0[(abs(df_tmp0['roll2_d']-0.5)<0.1) & (abs(df_tmp0['roll3_d']-0.3)<0.1) &
                   (abs(df_tmp0['roll3']-0.3)<0.1)].index.tolist()
    idx_best = list(set(idx1+idx2))
    idx_best.sort()
    best_point_idx = df_tmp.iloc[idx_best, :]['index'].tolist()

    solo_point_idx = df_tmp0[(abs(df_tmp['roll5']==0))]['index'].tolist()
    # print('f3错误序列号:', bad_point)
    # print('f3best0序列号:', idx_best)
    return bad_point_idx, best_point_idx, solo_point_idx


def find_near_point(p, n_max, up=False):
    if up:
        s = np.random.randint(2, 6)
    else:
        s = np.random.randint(1, 5)
    prob = 0.2 if up else np.random.random()
    if prob < 0.3:
        start = p
        end = min(start+s, n_max)
    elif prob > 0.7:
        end = p
        start = max(end-s, 0)
    else:
        start = min(p+1, n_max)
        end = min(start+3, n_max)
    return tuple([start, end])


class MyInversionMutation(Mutation):

    def __init__(self, data_encode, opt1=False, prob=0.5):
        super().__init__()
        self.prob = prob
        self.data, self.df_type = get_data_type(data_encode, opt1=opt1)

    def _do(self, problem, X, **kwargs):
        for i, y in enumerate(X):
            Y = X.copy()
            df_tmp = self.data.iloc[y, :].reset_index(drop=True)
            bad_point_idx, best_point_idx, solo_point_idx = find_f3_wrong(df_tmp)

            # max_iter = 1
            max_iter = max(len(bad_point_idx)//5, 1)
            max_iter = min(max_iter, 3)
            if len(bad_point_idx) > 0:
                for j in range(max_iter):
                    p = np.random.choice(bad_point_idx)
                    bad_point_idx.remove(p)
                    # 四驱车变异策略
                    q = find_same_type_point(self.data, self.df_type, p, best_point_idx, solo_point_idx)
                    if q is not None:
                        y = switch_two_point(list(y), p, q)
                Y[i] = y
            else:
                Y[i] = y
        return Y


def find_same_type_point(data_encode, df_type, p, idx_best, solo_point_idx):
    """从同类型数据中随机抽1条, data_encode必须是原始data序列"""

    for i, row in df_type.iterrows():
        if p in row['index']:
            df_type1 = data_encode.iloc[row['index'], :]
            idx0 = list(df_type1[df_type1['变速器'] == 0].index)
            idx0_best = list(set(idx0) & set(idx_best))
            idx0_solo = list(set(idx0) & set(solo_point_idx))
            # 查找是否有best idx
            if len(idx0) > 0:
                if len(idx0_best) > 0:
                    q = np.random.choice(idx0_best)
                    idx_best.remove(q)
                else:
                    if len(idx0_solo) > 0:
                        q = np.random.choice(idx0_solo)
                    else:
                        q = np.random.choice(idx0)
                return q
            else:
                return None
    if data_encode.iloc[[p, q], :]['type'].unique() > 1:
        print('error find_same_type_point')
        sys.exit()


def get_data_type(data_encode, opt1):
    """数据类别编码"""
    data_encode['color_flag'] = data_encode.apply(lambda x: 1 if (x['车身颜色'] != x['车顶颜色']) else 0, axis=1)
    # 分类
    df_type = data_encode.groupby(['color_flag', '车型', '车顶颜色'])['发动机'].count().reset_index().reset_index()
    df_type = df_type.drop('发动机', axis=1).rename(columns={'index': 'type'})

    if opt1:
        type1 = (df_type['车型']==0)&(df_type['车顶颜色']==7)&(df_type['color_flag']==1)
        type2 = (df_type['车型']==0)&(df_type['车顶颜色']==99)&(df_type['color_flag']==1)
        df_type.loc[type1, 'type'] = df_type.loc[type2, 'type'].values

    # 获取类别list
    data_encode_t = data_encode.merge(df_type, on=['color_flag', '车型', '车顶颜色'], how='left').reset_index()
    df_type2 = data_encode_t.groupby('type')['index'].apply(list).reset_index()
    df_type = df_type.merge(df_type2, on='type')
    data_encode_t = data_encode_t.drop(['物料编号', '发动机'], axis=1)
    return data_encode_t, df_type


def inversion_mutation(y, seq, inplace=True):
    y = y if inplace else np.copy(y)

    if seq is None:
        seq = random_sequence(len(y))
    start, end = seq

    y[start:end + 1] = np.flip(y[start:end + 1])
    return y


def insert_mutation(y, seq):
    start, end = seq
    a = list(y[start:end+1])
    a.append(a.pop(0))
    y[start:end + 1] = a
    return y


def switch_two_point(x, a, b):
    """交换list的两个元素
    eg: a = [1,2,3,4,5,6,7,8]
        switch_two_point(a, 2, 8)
    """
    c, d = x.index(a), x.index(b)
    x[c], x[d] = x[d], x[c]
    return np.array(x)


def random_sequence2(n):
    start = np.random.choice(n, replace=False)
    s = np.random.randint(3, n // 5)
    # s = 5
    end = min(start+s, n-1)
    return tuple([start, end])


def random_sequence(n):
    start, end = np.sort(np.random.choice(n, 2, replace=False))
    return tuple([start, end])
