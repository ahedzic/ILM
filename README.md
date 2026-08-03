# ILM


## Installation

To install required dependencies run:
```
pip install requirements.txt
```


## Download Data

Due to file size, missing Yelp and Flickr datasets (file includes all datasets) can be downloaded here: https://arizonastateu-my.sharepoint.com/:u:/g/personal/ahedzic_sundevils_asu_edu/IQCR4MBx8OTaRJypeiyL-v9EAV0_bVbCf1M3tPRZi_VGMfc?e=EDws7d
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


