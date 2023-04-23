from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.tools import generate_detections as gdet

from track_3 import track_data


# DeepSORT -> Initializing tracker.
max_cosine_distance = 0.4
nn_budget = None
model_filename = './model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

for frame in track_data:
    detections = [(el['bounding_box'], 1.0) for el in frame['data']]  # Set the confidence score to 1.0 for all detections
    # DeepSORT -> Predicting Tracks.
    tracker.predict()
    tracks = tracker.update(detections)
