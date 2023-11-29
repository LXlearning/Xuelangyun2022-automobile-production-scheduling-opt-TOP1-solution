# 2022雪浪云算法大赛-汽车全厂排产优化问题-冠军方案
## 赛题背景
汽车生产工艺复杂，一辆汽车的制造需要完成冲压、焊装、涂装、总装四大工艺，经过车身车间、涂装车间、总装车间。各车间存在上下游关联关系，每个车间有自己的优化排序目标，需要综合考虑多种复杂的排序规则及工艺约束，制定合理的混流装配排序计划，通过对车身序列进行排序优化，从而保证生产物料消耗的均衡性以及各个生产工位的负荷均匀化等。目前使用进化算法和大多数其他元启发式算法对于该问题进行优化可得到一组满足生产目标的方案。

## 方案说明
* [方案解析](https://mp.weixin.qq.com/s/nORSnpYJp1kcIrdzg8lEDA)
* [赛题地址](https://www.xuelangyun.com/#/cdc)


## 代码运行

```
python -m pip install --upgrade pip
pip install -r requirements.txt

# 全量运行所有数据,默认batch size=8(总时长5-6h)
python ./src/pymoo_main.py

# 单独运行某个数据集
python ./src/single_pymoo_main.py --name data_103
```


## 文件目录
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- 过程数据,包含每个数据集的优化结果
    │   ├── output         <- 结果数据,包含全量数据集优化的.xlsx文件
    │   └── raw            <- 原始数据
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    └── src                <- Source code for use in this project.


## 公众号

欢迎关注公众号

<img src="./docs/pics/code.jpg" width = "25%" />

