class BaseGraphModel():
    def __init__(self, model_params, dataset, results, gpu_id):
        self.model_params = model_params
        self.dataset = dataset
        self.results = results
        self.gpu_id = gpu_id

    def initialize_model(self):
        pass

    def prepare_graphs(self, dataset):
        return dataset

    def train_model(self, train_dataset):
        pass

    def test_model(self, test_dataset):
        pass