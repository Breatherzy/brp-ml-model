import os
from time import sleep

from models.BNModel import BNModel
from models.Conv1DModel import Conv1DModel
from models.GRUModel import GRUModel
from models.LRModel import LRModel
from models.LSTMModel import LSTMModel
from models.OneClassSVMModel import OneClassSVMModel
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel

models = [BNModel, Conv1DModel, GRUModel, LRModel, LSTMModel, OneClassSVMModel, RandomForestModel, SVMModel]

for model in models:
    print('Testing:', model.__name__)

    _model = model()
    _model.load_data('data/iris.txt')
    _model.compile()
    _model.fit()
    pre_save = _model.evaluate()
    _model.save('models/saves/' + _model.__class__.__name__ + '.keras')
    print('Model saved:', _model.__class__.__name__ + '.keras')
    sleep(5)
    _model = model()
    _model.load_data('data/iris.txt')
    _model.load('models/saves/' + _model.__class__.__name__ + '.keras')
    after_save = _model.evaluate()
    _model.fit()
    after_fit = _model.evaluate()

    print('Pre-save:', pre_save)
    print('After-save:', after_save)
    print('After-fit:', after_fit)
    sleep(5)
