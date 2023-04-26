import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class ObjFunc():
    def __init__(self, print_flag=0):
        self.print_flag = print_flag

    def weld_time(self, df):
        """焊接时间"""

        df_tmp = df.copy()
        df_tmp['车型切换'] = abs(df_tmp['车型'].diff())

        weld_time_all = 0  # 焊接总时间
        series_weld_time = 0  # 连续焊接时间
        switching_num = 0  # 切换次数
        switching_point = {}  # 切换位置&时间
        switching_num_better = 0

        for i, df_row in df_tmp.iterrows():
            weld_time_all += 80  # 焊接时间
            if (df_row['车型切换'] > 0) and (series_weld_time <= 1800):
                switching_point[i] = 1800 - series_weld_time  # 切换时间
                weld_time_all += 1800 - series_weld_time
                series_weld_time = 80
                switching_num += 1
            else:
                series_weld_time += 80
                if df_row['车型切换'] > 0:
                    switching_point[i] = 0
                    # 重置连续焊接时间
                    series_weld_time = 80
                    switching_num += 1
                    switching_num_better += 1

        weld_time_all = weld_time_all / len(df)
        bad_point = list(switching_point.keys())
        if self.print_flag == 1:
            print('车辆数:', len(df))
            print('切换位置:', switching_point)
            print('切换次数:', switching_num)
            print('>30min的切换次数:', switching_num_better)
            print('焊装平均耗时:', weld_time_all)
        return weld_time_all, switching_num, bad_point

    def paint_switching_num(self, df):
        """涂装喷头切换次数"""
        df_tmp = df.copy()
        df_tmp['车身颜色_old'] = df_tmp['车身颜色'].shift(1)
        df_tmp['车身颜色_old'].iloc[0] = df_tmp['车顶颜色'].iloc[0]

        switching_num = 0  # 切换次数
        series_num = 0  # 连续车数量
        series_num_dict = {}
        switching_point = []

        for i, df_row in df_tmp.iterrows():
            # 从下辆车开始统计之前有多少连续车辆
            case1 = (df_row['车顶颜色'] != df_row['车身颜色_old']) or (series_num >= 5)
            case2 = df_row['车身颜色'] != df_row['车顶颜色']
            if case1 or case2:
                if (series_num > 0):
                    series_num_dict[i] = series_num
                if case1 & (not case2):  # 当前车与上一辆需要切换 & 车顶车身不需要切换
                    switching_num += 1
                    switching_point.append(i)
                    series_num = 1   # 当前车是连续的第一辆车
                elif (not case1) & case2:  # 当前车与上一辆不需要切换 & 车顶车身需要切换
                    switching_num += 1
                    switching_point.append(i)
                    series_num = 0
                else:  # 当前车与上一辆需要切换 & 车顶车身需要切换
                    switching_num += 2
                    switching_point.append(i)
                    switching_point.append(i)
                    series_num = 0
            else:  # 车辆连续
                series_num += 1

        # 结尾的序列
        if series_num > 0:
            series_num_dict[i+1] = series_num
        series_num_dict_bad = dict((k, v) for (k, v) in series_num_dict.items() if v < 5)

        series_num_good = [v for (k, v) in series_num_dict.items() if v == 5]
        series_nums = list(series_num_dict_bad.values())
        series_idx = list(series_num_dict_bad.keys())
        series_score = np.round(np.sqrt((5 - np.array(np.array(series_nums)))).sum(), 4)

        bad_point = list(set(switching_point))
        if self.print_flag == 1:
            # print('切换位置:', switching_point)
            # print('good连续车数量:', series_num_good)
            print('切换次数:', switching_num)
            print('bad连续车数量:', series_nums)
            print('bad连续车位置', series_idx)
        return switching_num, series_score, bad_point

    def paint_time(self, df):
        """涂装耗时"""
        switching_num, series_score, bad_f2_point = self.paint_switching_num(df)
        t2_2 = 80 * switching_num
        t2 = t2_2 / len(df) + 80

        if self.print_flag == 1:
            print('涂装平均耗时:', t2)
        return t2, series_score, bad_f2_point

    def cal_obj(self, df):
        """总耗时"""
        if self.print_flag == 1:
            print('='*20, '焊装计算', '='*20)
        weld_time_all, f1, bad_f1_point = self.weld_time(df)

        if self.print_flag == 1:
            print('='*20, '涂装计算', '='*20)
        paint_time_all, f2, bad_f2_point = self.paint_time(df)

        f4 = (weld_time_all + paint_time_all)
        if self.print_flag == 1:
            print('='*20, '总装计算', '='*20)
        f3, bad_f3_point = self.cal_4wd(df)

        # bad_point = list(set(bad_f1_point + bad_f2_point + bad_f3_point))
        bad_point = bad_f3_point
        # if self.print_flag == 1:
        #     print('bad_point:', bad_point)
        return f1, f2, f3, f4, bad_point

    def cal_4wd(self, df):
        df_tmp = df.copy()
        df_tmp['变速器_old'] = df_tmp['变速器'].shift(1)

        switching_dict = {}  # 连续四驱末尾位置&数量
        series_num = 1
        for i, df_row in df_tmp.iterrows():
            if (df_row['变速器_old'] == 1):
                if (df_row['变速器'] == 1):
                    series_num += 1
                else:
                    switching_dict[i-1] = series_num
            else:
                series_num = 1
            if (i == len(df_tmp)-1) & (df_row['变速器'] == 1):
                switching_dict[i] = series_num

        switching_point = np.array(list(switching_dict.keys()))
        switching_value = np.array(list(switching_dict.values()))
        # bad1
        bad_end_point = (switching_value <= 1) | (switching_value >= 4)
        bad_car_end = switching_point[bad_end_point]
        bad_car_num = switching_value[bad_end_point]

        # bad2
        bad_end_point2 = switching_value == 3
        bad_car_num2 = switching_point[bad_end_point2]
        bad_car_end2 = switching_point[bad_end_point2]
        # 不满足f3的车位置
        bad_switching_point = []
        for i, p in zip(bad_car_num, bad_car_end):
            if i == 1:
                bad_switching_point.append(p)
            else:
                for j in range(i):
                    bad_switching_point.append(p-j)
        bad_switching_point = list(set(bad_switching_point + list(bad_car_end2)))
        bad_switching_point.sort()

        k_f3 = 1 / df_tmp['变速器'].sum()
        if df_tmp['变速器'].sum() % 2 == 0:
            score = bad_car_num.sum() / df_tmp['变速器'].sum() + k_f3 * len(bad_car_num2)
        else:
            score = bad_car_num.sum() / df_tmp['变速器'].sum() + k_f3 * (len(bad_car_num2)-1)

        if self.print_flag == 1:
            print('bad四驱末尾位置:', list(bad_switching_point))
            print('bad四驱连续数量:', list(bad_car_num))
            print('不满足要求的序列数:', bad_car_num.sum())
            print('四驱数量:', df_tmp['变速器'].sum())
            print('四驱3连放位置:', bad_car_end2)
        return score, bad_switching_point

    def cal_min_max(self, df):
        """上下限计算"""
        # 焊装计算
        f1_baseline = [1, df['车型'].value_counts().min()*2]

        df_color1 = df[df['车身颜色'] - df['车顶颜色'] == 0]
        df_color2 = df[df['车身颜色'] - df['车顶颜色'] != 0]

        # 涂装计算: 只考虑颜色相同车的排列
        df_tmp = df_color1['车身颜色'].value_counts() % 5
        f2_min = np.round(np.sqrt((5 - df_tmp[df_tmp > 0])).sum(), 4)
        f2_max = len(df_color1) * np.sqrt(4)
        f2_baseline = [f2_min, f2_max]
        # 涂装切换次数计算(min): 车身车顶相同颜色切换value_count//5+1次
        # 不同颜色时切换2次, 由于不同颜色的车顶可以放在相同颜色的前面, 需要-可排序的数量
        f2_min1 = (df_color1['车身颜色'].value_counts() // 5 + 1).sum()

        color_series = df_color1['车身颜色'].value_counts().reset_index()  # 相同颜色的可连续车颜色
        color_series['车身颜色'] = np.ceil(color_series['车身颜色']/5)
        color_not_series = df_color2['车身颜色'].value_counts().reset_index()  # 不同颜色可排在前面的车颜色
        f2_min2 = len(df_color2) * 2 - color_not_series.merge(color_series, on='index').set_index('index').min(axis=1).sum()
        f2_min_switching = f2_min1 + f2_min2 - 1

        # 车身车顶相同颜色切换1次, 不同颜色切换2次
        f2_max_switching = len(df_color1) + 2 * (len(df_color2))

        # 总装计算
        f3_baseline = [0, 1]

        # 时间计算(min), 焊装80s, 涂装80+80*切换次数/车数量
        f4_min = 80 + 80 + 80 * f2_min_switching / len(df)
        # 时间计算(max), 焊装80+f1_max*1720/车数量, 涂装80+80*切换次数/车数量
        f4_max = 80 + f1_baseline[1] * 1720 / len(df) + 80 + 80 * f2_max_switching / len(df)
        f4_baseline = [f4_min, f4_max]
        return f1_baseline, f2_baseline, f3_baseline, f4_baseline
