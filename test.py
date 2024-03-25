from models.AbstractModel import SensorType
from models.GRUModel import GRUModel
from scripts.load_data import save_sequences, save_sequences_to_concatenated, empty_file

BASE_SENSOR = SensorType.ACCELEROMETER.value
SENSOR_SIZE = 11


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
                filename=f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatenated.txt",
                sensor_type=BASE_SENSOR,
            )
            _model.compile()
            _model.fit(epochs=183)
            # _model.plot_prediction(
            #     _model.X_test, title=f"{_model.__class__.__name__} - {BASE_SENSOR}_concatenated -
            #     {_model.evaluate()*100:.2f}%")
            print("Model evaluated:", _model.evaluate())
            _model.save(
                f"models/saves/{BASE_SENSOR}/{_model.__class__.__name__}_{BASE_SENSOR}"
            )
            file.write(
                f"{_model.__class__.__name__} - {BASE_SENSOR}_concatenated - {_model.evaluate()*100:.2f}%\n"
            )
            # for data in ["_normal.txt", "_bezdech_wdech.txt", "_bezdech_wydech.txt", "_hiper.txt",
            #              "_wydech_wstrzym.txt", "_wdech_wstrzym.txt", "_bezdech.txt", ]:
            #     _model.load_data(filename=f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}" + data,
            #                      sensor_type=SensorType.ACCELEROMETER.value)
            #
            #     _model.fit()

            # print("Model evaluated:", _model.evaluate())
            # _model.plot_prediction(
            #     _model.X_test, title=f"{_model.__class__.__name__} - {BASE_SENSOR}_normal -
            #     {_model.evaluate()*100:.2f}%")


def plot_raw_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt(f"data/raw/{BASE_SENSOR}/{BASE_SENSOR}_test.txt")

    features = data
    labels = np.zeros(len(data))

    interactive_plot(features, labels, labels)


def plot_tagged_data():
    import numpy as np

    from scripts.plot import interactive_plot

    data = np.loadtxt(f"data/pretrained/{BASE_SENSOR}_point/{BASE_SENSOR}_test.txt", delimiter=",")

    features = data[:, :-1]
    labels = data[:, -1]

    interactive_plot(features, labels, labels)


# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

if __name__ == "__main__":
    empty_file(f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatenated.txt")
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
            save_sequences_to_concatenated(
                f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}" + data,
                f"data/pretrained/{BASE_SENSOR}_sequence/{BASE_SENSOR}_concatenated.txt",
            )

    models_test()
