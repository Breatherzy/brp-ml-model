from models.AbstractModel import SensorType
from models.BNModel import BNModel
from models.Conv1DModel import Conv1DModel
from models.GRUModel import GRUModel
from models.LRModel import LRModel
from models.LSTMModel import LSTMModel
from models.OneClassSVMModel import OneClassSVMModel
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel
from scripts.load_data import save_sequences, save_sequences_to_concatened, empty_file

BASE_SENSOR = SensorType.TENSOMETER.value
SENSOR_SIZE = 5


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
    with open(f"models/saves/{BASE_SENSOR}_evaluation.txt", "w") as file:
        for model in models:
            print("Testing:", model.__name__)

            _model = model()
            _model.load_data(
                filename=f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatened.txt",
                sensor_type=BASE_SENSOR,
            )
            _model.compile()
            _model.fit(epochs=183)
            # _model.plot_prediction(
            #     _model.X_test, title=f"{_model.__class__.__name__} - {BASE_SENSOR}_concatened - {_model.evaluate():.2f}%")
            # file.write(f"{model.__name__} - {BASE_SENSOR}_concatened - {_model.evaluate()*100:.2f}%\n")
            print("Model evaluated:", _model.evaluate())

            _model.save(
                f"models/saves/{BASE_SENSOR}/" + _model.__class__.__name__ + ".keras"
            )
            # print("Model saved:", _model.__class__.__name__ + ".keras")

            # for data in ["_normal.txt", "_bezdech_wdech.txt", "_bezdech_wydech.txt", "_hiper.txt", "_wydech_wstrzym.txt",
            #      "_wdech_wstrzym.txt", "_bezdech.txt",]:
            #     _model.load_data(filename=f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}" + data, sensor_type=SensorType.ACCELEROMETER.value)
            #
            #     _model.fit()

            # print("Model evaluated:", _model.evaluate())
            # _model.plot_prediction(
            #     _model.X_test, title=f"{_model.__class__.__name__} - tens_normal - {_model.evaluate():.2f}%")


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

    data = np.loadtxt("data/pretrained/tens_point/tens_test.txt", delimiter=",")

    features = data[:, :-1]
    labels = data[:, -1]

    interactive_plot(features, labels, labels)


# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

if __name__ == "__main__":
    empty_file(f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatened.txt")
    for data in [
        "_normal.txt",
        "_bezdech_wdech.txt",
        "_bezdech_wydech.txt",
        "_hiper.txt",
        "_wydech_wstrzym.txt",
        "_wdech_wstrzym.txt",
        "_bezdech.txt",
        "_test.txt",
    ]:
        save_sequences(
            f"data/pretrained/{BASE_SENSOR}_point/{BASE_SENSOR}" + data,
            f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}" + data,
            SENSOR_SIZE,
        )

        if data != "_test.txt":
            save_sequences_to_concatened(
                f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}" + data,
                f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatened.txt",
            )

    models_test()
