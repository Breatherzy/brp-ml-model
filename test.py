from time import sleep

from models.BNModel import BNModel
from models.Conv1DModel import Conv1DModel
from models.GRUModel import GRUModel
from models.LRModel import LRModel
from models.LSTMModel import LSTMModel
from models.OneClassSVMModel import OneClassSVMModel
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel
from scripts.load_data import save_sequences


def models_test():
    models = [
        # BNModel,
        # Conv1DModel,
        GRUModel,
        # LRModel,
        # LSTMModel,
        # OneClassSVMModel,
        # RandomForestModel,
        # SVMModel,
    ]

    for model in models:
        print("Testing:", model.__name__)

        _model = model()
        _model.load_data("data/pretrained/tens_sequence/tens_normal.txt")
        _model.compile()
        _model.fit(epochs=10)
        pre_save = _model.evaluate()
        _model.save("models/saves/" + _model.__class__.__name__ + ".keras")
        print("Model saved:", _model.__class__.__name__ + ".keras")

        _model.plot_prediction()
        # sleep(5)
        # _model = model()
        # _model.load_data("data/pretrained/tens_sequence/tens_test.txt")
        # _model.load("models/saves/" + _model.__class__.__name__ + ".keras")
        # after_save = _model.evaluate()
        # _model.fit()
        # after_fit = _model.evaluate()

        print("Pre-save:", pre_save)
        # print("After-save:", after_save)
        # print("After-fit:", after_fit)
        sleep(5)


def plot_raw_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt("data/raw/tens/tens_test.txt")

    features = data
    labels = np.zeros(len(data))

    interactive_plot(features, labels)


def plot_tagged_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt("data/pretrained/tens_point/tens_test.txt", delimiter=",")

    features = data[:, :-1]
    labels = data[:, -1]

    interactive_plot(features, labels)

# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

if __name__ == "__main__":
    # save_sequences("data/pretrained/tens_point/tens_test.txt", "data/pretrained/tens_sequence/tens_test.txt", 5)
    models_test()
