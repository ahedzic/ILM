from models.mlp import MLP
from models.ilm import ILM
from models.IGAT import IGAT
from comparison_models.gran import GRAN
from comparison_models.accslp import ACCSLP
from comparison_models.leroy import Leroy
from threading import Thread

class ModelRunner(Thread):
    def __init__(self, model_config, dataset, results, gpu_id):
        super(ModelRunner, self).__init__()
        self.model_config = model_config
        self.dataset = dataset
        self.results = results
        self.gpu_id = gpu_id

    def create_model(self):
        model_name = self.model_config['model']

        if model_name == 'MLP':
            self.model = MLP(self.model_config, self.dataset, self.results, self.gpu_id)
        if model_name == 'GRAN':
            self.model = GRAN(self.model_config, self.dataset, self.results, self.gpu_id)
        if model_name == 'ACCSLP':
            self.model = ACCSLP(self.model_config, self.dataset, self.results, self.gpu_id)
        if model_name == 'IGAT':
            self.model = IGAT(self.model_config, self.dataset, self.results, self.gpu_id)
        if model_name == 'Leroy':
            self.model = Leroy(self.model_config, self.dataset, self.results, self.gpu_id)
        if model_name == 'ILM':
            self.model = ILM(self.model_config, self.dataset, self.results, self.gpu_id)

    def run(self):
        self.create_model()
        self.model.initialize_model()

        if not self.model_config["inference_only"]:
            self.model.train_model(self.dataset['train'])

        if self.model_config['model'] == 'Leroy':
            self.model.test_model((self.dataset['train'] + self.dataset['test']))
        else:
            self.model.test_model(self.dataset['test'])