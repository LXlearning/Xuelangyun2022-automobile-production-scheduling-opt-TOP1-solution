# 2022-Xuelangyun-automobile-production-scheduling-opt-TOP1-solution
## 方案说明
* 方案解析: 
* 赛题地址:  [雪浪算力开发者大赛](https://www.xuelangyun.com/#/cdc)

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


