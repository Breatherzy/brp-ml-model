from models.AbstractModel import SensorType
from models.GRUModel import GRUModel
from models.LSTMModel import LSTMModel
from scripts.load_data import prepare_data_for_training
from scripts.plot import plot_history, plot_evaluation_history

# TODO: Change logic in labelling or any different
#  plot so it can be used in this test using predicted
#  data from model and X_test, y_test fields

if __name__ == "__main__":
    SENSOR = SensorType.TENSOMETER
    SENSOR_NAME = SENSOR.value["name"]

    prepare_data_for_training(sensor=SENSOR)

    model = LSTMModel()
    model.load_data(
        filename=f"data/labelled/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
        sensor_type=f"{SENSOR_NAME}",
    )
    model.compile()
    model.fit(sensor_type=f"{SENSOR_NAME}")

    model.save(f"models/saves/{SENSOR_NAME}/LSTMModel_{SENSOR_NAME}")
    model.plot_prediction(model.X_test, name=f"{SENSOR_NAME}_concatenated")
    plot_history(f"models/saves/{SENSOR_NAME}/LSTMModel.history")


def evaluate_epochs():
    prepare_data_for_training(sensor=SENSOR)
    history = []

    for i in range(10, 101, 10):
        _model = LSTMModel()
        _model.load_data(
            filename=f"data/labelled/{SENSOR_NAME}_sequence/{SENSOR_NAME}_concatenated.txt",
            sensor_type=f"{SENSOR_NAME}",
        )
        _model.compile()
        _model.fit(epochs=i)
        history.append((i, _model.evaluate()))

    print(*history)

    plot_evaluation_history(history)
