import pandas as pd
class CustomTrackingMetric:
    """
    Custom metric by
    Sasha, Ilia and Renat ;)
    """

    def __init__(self):
        self.data = []

    def add_row(self, predict):
        row = {'frame_id': predict['frame_id']}
        for el in predict['data']:
            row[f"cb_id_{el['cb_id']}"] = el['track_id']

        self.data.append(row)

    def calculate_metric(self):
        data = pd.DataFrame(self.data)



# Костыль с importom здесь для проверки модели
# Чтобы не поднимать сервер по over раз
# Потом удалить
if __name__ == '__main__':
    from models import EasyModel

    easy_model = EasyModel()
    metric_model = CustomTrackingMetric()
    from track_22_04_21_40 import track_data

    for el in track_data:
        predict = easy_model.predict(el)

        metric_model.add_row(predict)

    metric_model.calculate_metric()