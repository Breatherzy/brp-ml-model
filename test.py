from time import sleep

from models.BNModel import BNModel
from models.Conv1DModel import Conv1DModel
from models.GRUModel import GRUModel
from models.LRModel import LRModel
from models.LSTMModel import LSTMModel
from models.OneClassSVMModel import OneClassSVMModel
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel
from scripts.load_data import save_sequences, save_sequences_to_concatened, empty_file


def models_test():
    models = [
        # BNModel,
        # Conv1DModel,
        # GRUModel,
        # LRModel,
        LSTMModel,
        # OneClassSVMModel,
        # RandomForestModel,
        # SVMModel,
    ]

    for model in models:
        print("Testing:", model.__name__)

        _model = model()
        _model.load_data("data/pretrained/tens_sequence/tens_concatened.txt")
        _model.compile()
        _model.fit()
        _model.plot_prediction(
            _model.X_test, title=f"{_model.__class__.__name__} - tens_normal - {_model.evaluate():.2f}%")
        print("Model evaluated:", _model.evaluate())

        # pre_save = _model.evaluate()
        # _model.save("models/saves/" + _model.__class__.__name__ + ".keras")
        # print("Model saved:", _model.__class__.__name__ + ".keras")

        # for data in ["tens_bezdech_wdech.txt", "tens_bezdech_wydech.txt", "tens_hiper.txt", "tens_wydech_wstrzym.txt", "tens_wdech_wstrzym.txt"]:
        #     _model.load_data("data/pretrained/tens_sequence/" + data)
        #     _model.fit()
        #     # _model.plot_prediction(
        #     #     _model.X_test, title=f"{_model.__class__.__name__} - {data} - {_model.evaluate():.2f}%")
        #
        # _model.plot_prediction(
        #     _model.X_test, title=f"{_model.__class__.__name__} - tens_normal - {_model.evaluate():.2f}%")

        # _model.load("models/saves/" + _model.__class__.__name__ + ".keras")
        # after_save = _model.evaluate()
        # _model.fit()
        # after_fit = _model.evaluate()

        # print("After-save:", after_save)
        # print("After-fit:", after_fit)
        # sleep(5)


def plot_raw_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt("data/raw/tens/tens_test.txt")

    features = data
    labels = np.zeros(len(data))

    interactive_plot(features, labels, labels)


def plot_tagged_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt(
        "data/pretrained/tens_point/tens_test.txt", delimiter=",")

    features = data[:, :-1]
    labels = data[:, -1]

    interactive_plot(features, labels, labels)


# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

if __name__ == "__main__":
    empty_file("data/pretrained/tens_sequence/tens_concatened.txt")
    for data in ["tens_normal.txt", "tens_bezdech_wdech.txt", "tens_bezdech_wydech.txt", "tens_hiper.txt", "tens_wydech_wstrzym.txt",
                 "tens_wdech_wstrzym.txt", "tens_bezdech.txt", "tens_test.txt",]:
        save_sequences("data/pretrained/tens_point/" + data,
                       "data/pretrained/tens_sequence/" + data, 5)

        if data != "tens_test.txt":
            save_sequences_to_concatened(
                "data/pretrained/tens_sequence/" + data, "data/pretrained/tens_sequence/tens_concatened.txt"
            )

    models_test()
