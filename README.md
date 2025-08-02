# ILM


## Installation

To install required dependencies run:
```
pip install requirements.txt
```


## Download Data

All datasets can be downloaded here: https://limewire.com/d/cvi6c#kkv7rAnyIq

Please extract the dataset.zip into the root directory.


## Reproduce Results

The commands needed to reproduce all the results with the appropriate hyperparameters can be found in the **`scripts/hyparameters/cold_start`** directory. We include a file for each table in the paper which includes the commands to reproduce the table.

For example, to reproduce the overall results, the command for each method can be found in the `overall_comparison.sh` file located in the `scripts/hyperparameter/cold_start/` directory.

The script file can be run itself (not recommended due to serial execution). Example:
```
./scripts/hyparameters/cold_start/overall_comparison.sh
```

Or the commands can be run individually:
```
cd benchmarking/cold_start/
CUBLAS_WORKSPACE_CONFIG=:4096:8 python main_cold_ilm.py --data_name starcraft --input_size 25 --gnn_model GCN --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256 --num_layers_predictor 2 --epochs 9999 --kill_cnt 3 --eval_steps 5 --runs 10 --iterations 3 --weights meta --cold_perc 0.0
```


## References

In order to run these experiments we modified the HeaRT framework to run in the Zero-Edge Cold-Start fully inductive setting, which includes their implementations of GNN, GNN+Pairswise, and MLP models.

For all other cold-start competitor models we used the official implementation, also modified to fit our experimental settings. All references can be seen below.

HeaRT:
https://github.com/Juanhui28/HeaRT

```
@inproceedings{
  li2023evaluating,
  title={Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking},
  author={Li, Juanhui and Shomer, Harry and Mao, Haitao and Zeng, Shenglai and Ma, Yao and Shah, Neil and Tang, Jiliang and Yin, Dawei},
  booktitle={Neural Information Processing Systems {NeurIPS}, Datasets and Benchmarks Track},
  year={2023}
}
```

DEAL:
https://github.com/working-yuhao/DEAL

```
@inproceedings{ijcai2020-168,
  title     = {Inductive Link Prediction for Nodes Having Only Attribute Information},
  author    = {Hao, Yu and Cao, Xin and Fang, Yixiang and Xie, Xike and Wang, Sibo},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {1209--1215},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/168},
  url       = {https://doi.org/10.24963/ijcai.2020/168},
}
```


LEAP:
https://github.com/AhmedESamy/LEAP

```
@inproceedings{10.1007/978-3-031-82481-4_31,
author = {Samy, Ahmed E. and Kefato, Zekarias T. and Girdzijauskas, \v{S}ar\={u}nas},
title = {Leap: Inductive Link Prediction via Learnable Topology Augmentation},
year = {2025},
isbn = {978-3-031-82480-7},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-82481-4_31},
doi = {10.1007/978-3-031-82481-4_31},
abstract = {Link prediction is a crucial task in many downstream applications of graph machine learning. To this end, Graph Neural Network (GNN) is a widely used technique for link prediction, mainly in transductive settings, where the goal is to predict missing links between existing nodes. However, many real-life applications require an inductive setting that accommodates for new nodes, coming into an existing graph. Thus, recently inductive link prediction has attracted considerable attention, and a multi-layer perceptron (MLP) is the popular choice of most studies to learn node representations. However, these approaches have limited expressivity and do not fully capture the graph’s structural signal. Therefore, in this work we propose LEAP, an inductive link prediction method based on LEArnable toPology augmentation. Unlike previous methods, LEAP models the inductive bias from both the structure and node features, and hence is more expressive. To the best of our knowledge, this is the first attempt to provide structural contexts for new nodes via learnable augmentation in inductive settings. Extensive experiments on seven real-world homogeneous and heterogeneous graphs demonstrates that LEAP significantly surpasses SOTA methods. The improvements are up to 22\% and 17\% in terms of AUC and average precision, respectively. The code and datasets are available on GitHub (1).},
booktitle = {Machine Learning, Optimization, and Data Science: 10th International Conference, LOD 2024, Castiglione Della Pescaia, Italy, September 22–25, 2024, Revised Selected Papers, Part I},
pages = {448–463},
numpages = {16},
keywords = {Inductive link prediction, Graph Neural Networks, Learnable augmentation, Heterogeneous graphs},
location = {Castiglione della Pescaia, Italy}
}
```

CSMDDI:
https://github.com/itsosy/csmddi

```
@article{article,
author = {Liu, Zun and Wang, Xing-Nan and Yu, Hui and Shi, Jian-Yu and Dong, Wen-Min},
year = {2022},
month = {02},
pages = {},
title = {Predict multi-type drug–drug interactions in cold start scenario},
volume = {23},
journal = {BMC Bioinformatics},
doi = {10.1186/s12859-022-04610-4}
}
```

