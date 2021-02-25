from torch import load, argmax, zeros
from torch.nn import Softmax
from os.path import exists
from argparse import ArgumentParser
from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved

from pathlib import Path


def create_submission(model_type, batch_size):

    model_one_path = 'improved_semantic-model_one.pt'
    model_two_path = 'improved_semantic-model_two.pt'
    model_three_path = 'improved_semantic-model_three.pt'

    dataset = load("semantic_segmentation_dataset.pt")
    data_val = dataset["images_tr"][dataset["sets_tr"] == 2]

    model_one_state = load(model_one_path)
    model_one = SemanticSegmentationImproved(model_one_state['specs'])
    model_one.load_state_dict(model_one_state['state'])
    model_one.eval()

    model_two_state = load(model_two_path)
    model_two = SemanticSegmentationImproved(model_two_state['specs'])
    model_two.load_state_dict(model_two_state['state'])
    model_two.eval()

    model_three_state = load(model_three_path)
    model_three = SemanticSegmentationImproved(model_three_state['specs'])
    model_three.load_state_dict(model_three_state['state'])
    model_three.eval()

    pred_one_val = evaluate(data_val,model_one,batch_size)
    pred_two_val = evaluate(data_val,model_two,batch_size)
    pred_three_val = evaluate(data_val,model_three,batch_size)

    pred_val = ((pred_one_val+pred_two_val+pred_three_val)/3).short()

    assert pred_val.size() == (200, 32, 32), f"Expected the output of the validation set to be of size (200, 32, 32) but was {pred_val.size()} instead"

    with Path(f"kaggle_{model_type}_val_submission.csv").open(mode="w") as writer:
        writer.write("Id,Category\n")
        for i in range(pred_val.size(0)):
            for j in range(pred_val.size(1)):
                for k in range(pred_val.size(2)):
                    writer.write(f"{i}_{j}_{k},{pred_val[i, j, k]}\n")


def evaluate(data, model, batch_size):
    num_examples = data.size(0)
    img_size = 32

    soft_max = Softmax(dim=1)
    pred_vals = zeros(num_examples, img_size, img_size)

    for i in range(0, num_examples, batch_size):
        pred_vals[i: i + batch_size] = argmax(soft_max(model(data[i: i + batch_size])), dim=1).squeeze()

    del data
    return pred_vals.long()


if __name__ == '__main__':
    # change model_type and batch_size to suit your needs
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="improved", type=str, help="Specify model type")
    parser.add_argument("--batch_size", default=100, type=int, help="specify the batch size")
    args, _ = parser.parse_known_args()

    create_submission(**args.__dict__)