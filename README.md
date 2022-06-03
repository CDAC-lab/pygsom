# `pygsom` - python GSOM algorithm

`pygsom` is an **open source** python-based implementation of **GSOM** algorithm. GSOM is unsupervised  dimensionality reduction and clustering  algorithm 


## Table of Contents

* [Installation](#installation)
* [Minimal example](#minimal-example)
* [Getting started](#getting-started)
* [Citing gsom](#citing pygsom)

## Installation

To pip install `pygsom` from github:

```bash
pip install pygsom
```


pygsom  supports Python 3.6+.

## Minimal example


```python
import numpy as np
import pandas as pd
import gsom

data_filename = "data/zoo.txt".replace('\\', '/')


if __name__ == '__main__':
    np.random.seed(1)
    df = pd.read_csv(data_filename)
    print(df.shape)
    data_training = df.iloc[:, 1:17]
    gsom_map = gsom.GSOM(.83, 16, max_radius=4)
    gsom_map.fit(data_training.to_numpy(), 100, 50)
    map_points = gsom_map.predict(df,"Name","label")
    gsom.plot(map_points, "Name", gsom_map=gsom_map)
    map_points.to_csv("gsom.csv", index=False)
```

## Getting started
Train the GSOM algorithm : need to give input data in numpy array with training iterations and smoothing iterations
```python
gsom_map.fit(data_training.to_numpy(), <training iterations>, <smooth iterations>)
```
Predict cluster nodes : need to give input data in pandas dataframe with names and labels 
```python
map_points = gsom_map.predict(df,<name column name>,<label column name>)
```
Plot the 2D map: need to give the output of predict function with label column (name column or label column)
```python
gsom.plot(map_points, <name column name/label column name>, gsom_map=<gsom_map>)
```

## Citing pygsom

If you use `pygsom`, please cite the following paper:

```
@article{alahakoon2000dynamic,
  title={Dynamic self-organizing maps with controlled growth for knowledge discovery},
  author={Alahakoon, Damminda and Halgamuge, Saman K and Srinivasan, Bala},
  journal={IEEE Transactions on neural networks},
  volume={11},
  number={3},
  pages={601--614},
  year={2000},
  publisher={IEEE}
}
```
