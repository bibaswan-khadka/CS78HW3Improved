from semantic_segmentation_base import SemanticSegmentationBase
from semantic_segmentation_improved import SemanticSegmentationImproved
from create_dataset import create_dataset
from train import train
from utils import distrib
from torch import save, random
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch
from utils import distrib

# seeding the random number generator. You can disable the seeding for the improvement model
random.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def semantic_segmentation(model_type="base"):
    """
    sets up and trains a semantic segmentation model

    Arguments
    ---------
    model_type:  (String) a string in {'base', 'improved'} specifying the targeted model type
    """
    
    # the dataset
    train_dl, val_dl = create_dataset("semantic_segmentation_dataset.pt")

    # an optional export directory
    exp_dir = f"{model_type}_models"

    classcount, rgbmean = distrib(train_dl)

    classcount = 1/classcount
    classcount = classcount.to(device)
    if model_type == "base":
        # specify netspec_opts
        netspec_opts = {
            "name": ["conv_1","bn_1","relu_1","conv_2","bn_2","relu_2","conv_3","bn_3","relu_3","conv_4","bn_4","relu_4", "conv_5","upsample_4x","skip_6","sum_6","upsample_2x"],
            "kernel_size": [3,0,0,3,0,0,3,0,0,3,0,0,1,4,1,0,4],
            # Fill filter size for relu and sum as well since skip layers and others use them
            "num_filters": [16,16,16,32,32,32,64,64,64,128,128,128,36,36,36,36,36],
            "stride": [1,0,0,2,0,0,2,0,0,2,0,0,1,4,1,0,2],
            "layer_type": ['conv','bn','relu','conv','bn','relu','conv','bn','relu','conv','bn','relu','conv','convt','skip','sum','convt'],
            "input": [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,5,(14,13),15],
            "pad": [1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1]
        }
        # specify train_opt
        train_opts = {
            "lr": 0.1,
            "weight_decay": 0.001,
            "batch_size": 24,
            "momentum": 0.9,
            "num_epochs": 34,
            "step_size": 30,
            "gamma": 0.1,
            "objective": CrossEntropyLoss()
        }

        model = SemanticSegmentationBase(netspec_opts)
        model.to(device)

    elif model_type == "improved":

        # specify netspec_opts
        netspec_opts = {
            "name": ["conv_1","bn_1","relu_1",'pool_1',"conv_2","bn_2","relu_2","pool_2","conv_3","bn_3","relu_3","pool_3","conv_4","bn_4","relu_4", "drop_1","conv_5","upsample_4x","skip_6", "sum_6", "skip_10", "upsample_skip_10","sum_10","upsample_2x"],
            "kernel_size": [3,0,0,2,3,0,0,2,3,0,0,2,3,0,0,0,1,4,1,0,1,4,0,4],
            # Fill filter size for relu and sum as well since skip layers and others use them
            "num_filters": [128,128,128,128,256,256,256,256,512,512,512,512,1024,1024,1024,1024,36,36,36,36,36,36,36,36],
            "stride": [1,0,0,2,1,0,0,2,1,0,0,2,1,0,0,0,1,4,1,0,1,2,0,2],
            "layer_type": ['conv','bn','relu','pool','conv','bn','relu','pool','conv','bn','relu','pool','conv','bn','relu','drop','conv','convt','skip','sum','skip','convt','sum','convt'],
            "input": [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,6,(18,17),10,20,(21,19),22],
            "pad": [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1]
        }
        # specify train_opts
        train_opts = {
            "lr": 0.1,
            "weight_decay": 0.001,
            "batch_size": 24,
            "momentum": 0.9,
            "num_epochs": 12,
            "step_size": [7,10],
            "gamma": 0.1,
            "objective": CrossEntropyLoss(classcount.float())
        }

        model = SemanticSegmentationImproved(netspec_opts)
        model.to(device)
        CNN_model_params = torch.load('improved_state_dict_CNN(128,256,512).pt')
        model_params = model.state_dict().copy()
        #print(model.state_dict().keys())
        #for p in model.named_parameters():
          #print(p)
        #i = 0
        #freezelayers = {0,1,2,3,4,5}
        #for p in model.named_parameters():
          #if i in freezelayers:
            #print(p)
          #i = i+1
        model_params['net.conv_1.weight'] = CNN_model_params['conv0.weight']
        model_params['net.conv_1.bias'] = CNN_model_params['conv0.bias']
        model_params['net.bn_1.weight'] = CNN_model_params['bn1.weight']
        model_params['net.bn_1.bias'] = CNN_model_params['bn1.bias']
        #model_params['net.bn_1.running_mean'] = CNN_model_params[ 'bn1.running_mean']
        #model_params['net.bn_1.running_var'] = CNN_model_params[ 'bn1.running_var']
        #model_params['net.bn_1.num_batches_tracked'] = CNN_model_params['bn1.num_batches_tracked']

        model_params['net.conv_2.weight'] = CNN_model_params['conv4.weight']
        model_params['net.conv_2.bias'] = CNN_model_params['conv4.bias']
        #model_params['net.bn_2.weight'] = CNN_model_params['bn5.weight']
        #model_params['net.bn_2.bias'] = CNN_model_params['bn5.bias']
        #model_params['net.bn_2.running_mean'] = CNN_model_params[ 'bn5.running_mean']
        #model_params['net.bn_2.running_var'] = CNN_model_params[ 'bn5.running_var']
        #model_params['net.bn_2.num_batches_tracked'] = CNN_model_params['bn5.num_batches_tracked']

        #model_params['net.conv_3.weight'] = CNN_model_params['conv8.weight']
        #model_params['net.conv_3.bias'] = CNN_model_params['conv8.bias']
        #model_params['net.bn_3.weight'] = CNN_model_params['bn9.weight']
        #model_params['net.bn_3.bias'] = CNN_model_params['bn9.bias']
        #model_params['net.bn_3.running_mean'] = CNN_model_params[ 'bn9.running_mean']
        #model_params['net.bn_3.running_var'] = CNN_model_params[ 'bn9.running_var']
        #model_params['net.bn_3.num_batches_tracked'] = CNN_model_params['bn9.num_batches_tracked']
        
        model.load_state_dict(model_params)
        index = 0
        freezelayers = {0,1,2,3,4,5}
        for p in model.parameters():
          if index in freezelayers:
            p.requires_grad = False
          index +=1
        #for p in model.named_parameters():
          #print(p)
    else:
        raise ValueError(f"Error: unknown model type {model_type}")

    # train the model
    train(model, train_dl, val_dl, train_opts, exp_dir=exp_dir)

    # save model's state and architecture to the base directory
    model = {"state": model.state_dict(), "specs": netspec_opts}
    save(model, f"{model_type}_semantic-model.pt")

    plt.savefig(f"{model_type}_semantic.png")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_type", default="improved", type=str, help="Specify model type")
    args, _ = parser.parse_known_args()

    semantic_segmentation(**args.__dict__)
