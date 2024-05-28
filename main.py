from models.AbstractModel import SensorType
from models.GRUModel import GRUModel
from models.LSTMModel import LSTMModel
from scripts.load_data import prepare_data_for_training
from scripts.plot import plot_evaluation_history, plot_history, plot_test_data

# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

SENSOR = SensorType.TENSOMETER
SENSOR_NAME = SENSOR.value["name"]


def evaluate_epochs():
    prepare_data_for_training(sensor=SENSOR)
    history = []

    for i in range(10, 101, 10):
        _model = GRUModel()
        _model.load_data(
            filename=f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
            sensor_type=f"{SENSOR_NAME}",
        )
        _model.compile()
        _model.fit(epochs=i, sensor_type=f"{SENSOR_NAME}")
        history.append((i, _model.evaluate()))

    print(*history)

    plot_evaluation_history(history)


if __name__ == "__main__":
    prepare_data_for_training(sensor=SENSOR)
    model = GRUModel()
    model.load_data(
        filename=f"data/pretrained/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
        sensor_type=f"{SENSOR_NAME}",
    )
    model.compile()
    model.fit(sensor_type=f"{SENSOR_NAME}", epochs=50)

    # model.load(f"models/saves/{SENSOR_NAME}/GRUModel_{SENSOR_NAME}")
    model.save(f"models/saves/{SENSOR_NAME}/GRUModel_{SENSOR_NAME}")
    model.confusion_matrix(model.X_test, model.y_test, name=f"{SENSOR_NAME}_test")
    model.plot_prediction(model.X_test, name=f"{SENSOR_NAME}_test")
    plot_history(f"models/saves/{SENSOR_NAME}/GRUModel.history")
    plot_test_data(SENSOR_NAME, normalize_data=True)
