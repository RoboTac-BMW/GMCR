from fitting_graphs.datasets.stanford_scanning_repository import StanfordDataset
from src.fg_reg import FittingGraphRegistration
import numpy as np

BASE_PATH = 'datasets/data/stanford_dataset'

PARAMETERS = {'FG_REG': None}
O_R = 0.9

# Setup Dataset and get one sample
dataset = StanfordDataset(BASE_PATH)
samp = dataset.get_sample()
inlier_c, outlier_c = samp.sample_correspondences(O_R, 50)
correspondences = np.concatenate([inlier_c, outlier_c], axis=0)

# Create FittingGraphMethod
fg_reg = FittingGraphRegistration(PARAMETERS)
fg_reg(samp.source.points, samp.target.points, correspondences)
print(samp.s)
