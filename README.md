# ILM
Datasets:
Note, data sets contain pickled graphs that are in the following format which is a python dictionary:
{
	"expected": A DGL graph containing the true graph.
	"empty": A DGL graph without any edges.
	"full": A DGL graph that is fully connected. The model uses this graphs to get all possible node pairs. Additionally, this graph contains the edge labels in its edata['label']. Labels are one-hot encoded to the label if the edge exists, and all zeros if the edge does not exist.
}

DGL version must be compatible to load the pickle file for a graph.

Setup:
Running the models requires a cuda environment with at least one GPU present.

*_configs.txt contains the configuration for each dataset for the model.

To run a specific model/dataset copy the configuration to the run_config.json file. An example is provided.

Set available gpus in run_config.json to the ids of the GPUs available on the system. You can run one model per available GPU.

To run:
After correct setup run with the following command:
python main.py