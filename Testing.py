from create_dataset import create_dataset
from utils import distrib

train, val = create_dataset('semantic_segmentation_dataset.pt')

classcount, rgbmean = distrib(val)

