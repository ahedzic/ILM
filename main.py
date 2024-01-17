import json
from model_runner import ModelRunner
from utils import dataset_load, MetricsProcessor

CONFIG_PATH = "run_config.json"

def main():
    config_file = open(CONFIG_PATH, 'r')
    config = json.load(config_file)
    gpus = config['available_gpus']
    metrics = MetricsProcessor("")
    runners = []

    for model_config in config['models']:
        train_graphs, test_graphs = dataset_load(model_config['dataset_name'], model_config['dataset_path'], model_config['train_ratio'])
        dataset = { 'train': train_graphs, 'test': test_graphs }
        gpu = gpus.pop()
        runner = ModelRunner(model_config, dataset, metrics, gpu) # Update to distribute across gpus
        runners.append(runner)
        runner.daemon = True
        runner.start()

    for runner in runners:
        runner.join()
        
    metrics.export_metrics()
    print(config)

if __name__ == '__main__':
    main()