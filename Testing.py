from create_dataset import create_dataset
from utils import distrib
from semantic_segmentation_improved import SemanticSegmentationImproved
from torch import load
train, val = create_dataset('semantic_segmentation_dataset.pt')

classcount, rgbmean = distrib(val)

print(load('improved_state_dict_CNN.pt'))
#print(model_state['state'])
#model = SemanticSegmentationImproved(model_state['specs'])
#model.load_state_dict(load(model_state['state']))
