from fitting_graphs.datasets.stanford_scanning_repository import StanfordDataset
import numpy as np

BASE_PATH = "../datasets/data/stanford_dataset"

dataset = StanfordDataset(BASE_PATH)
dataset.exclude_transform = ['s']

while dataset.current_model_name != 'bunny_model':
    print("Next")
    dataset.next_object()

samp = dataset.get_sample()

samp.transform_target(np.identity(3), np.array([[1, 1, 1]]))

samp.display()