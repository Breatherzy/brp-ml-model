from models.BatchNormalizationModel import BNModel
from models.Conv1DModel import Conv1DModel

models = [BNModel, Conv1DModel]

for model in models:
    _model = model()
    _model.load_data('data/iris.txt')
    _model.compile()
    _model.fit()
    _model.evaluate()
    _model.save('models/saves/' + _model.__class__.__name__ + '.keras')

    print('Model saved:', _model.__class__.__name__ + '.keras')

    _model = model()
    _model.load_data('data/iris.txt')
    _model.load('models/saves/' + _model.__class__.__name__ + '.keras')
    _model.evaluate()
