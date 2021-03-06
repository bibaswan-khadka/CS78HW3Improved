from torch import nn
from sum_layer import Sum


class SemanticSegmentationBase(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the base semantic segmentation model. Given that there are
        several layers, we strongly recommend that you keep your layers in an nn.ModuleDict as described in
        the assignment handout. nn.ModuleDict mirrors the operations of Python dictionaries.

        You will specify the architecture of the module in the constructor. And then define the forward
        propagation in the forward method as described in the handout.

        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network. netspec_opts has the keys
                                    1. kernel_size: (list) of size L where L is the number of layers
                                        representing the kernel sizes
                                    2. layer_type: (list) of size L indicating the type of each layer
                                    3. num_filters: (list) of size L representing the number of filters for each layer
                                    4. stride: (list) of size L indicating the striding factor of each layer
                                    5. input: (List) of size L containing the layer number of the inputs for each layer.

        """
        super(SemanticSegmentationBase, self).__init__()

        self.net = nn.ModuleDict()
        
        # add the hidden layers as specified in the handout
        name = netspec_opts['name']
        kernel_size = netspec_opts['kernel_size']
        num_filters = netspec_opts['num_filters']
        stride = netspec_opts['stride']
        layer_type = netspec_opts['layer_type']
        input = netspec_opts['input']
        pad = netspec_opts['pad']

        for index, (n,k_s,n_f,s,l_t,i,p) in enumerate(zip(name,kernel_size,num_filters,stride,layer_type,input,pad)):
            if l_t == "conv":
                if index == 0:
                    self.net[n] = nn.Conv2d(3,n_f,k_s,s,p)
                else:
                    self.net[n] = nn.Conv2d(num_filters[i], n_f, k_s, s,p)
            if l_t == "bn":
                self.net[n] = nn.BatchNorm2d(num_filters[i])
            if l_t == "relu":
                self.net[n] = nn.ReLU()
            if l_t == "convt":
                self.net[n] = nn.ConvTranspose2d(in_channels=num_filters[i],out_channels=n_f,kernel_size=k_s,stride=s,padding=p,groups=num_filters[i],bias=False)
            if l_t == "skip":
                self.net[n] = nn.Conv2d(num_filters[i],n_f,k_s,s,p)
            if l_t == "sum":
                self.net[n] = Sum()


    def forward(self, x):
        """
        Define the forward propagation of the base semantic segmentation model here. Starting with the input, pass
        the output of each layer to the succeeding layer until the final layer. Return the output of final layer
        as the predictions.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W) where C' is the number of classes.

        """

        # implement the forward propagation as defined in the handout
        out_conv_1 = self.net['conv_1'](x)
        out_bn_1 = self.net['bn_1'](out_conv_1)
        out_relu_1 = self.net['relu_1'](out_bn_1)
        out_conv_2 = self.net['conv_2'](out_relu_1)
        out_bn_2 = self.net['bn_2'](out_conv_2)
        out_relu_2 = self.net['relu_2'](out_bn_2)
        out_conv_3 = self.net['conv_3'](out_relu_2)
        out_bn_3 = self.net['bn_3'](out_conv_3)
        out_relu_3 = self.net['relu_3'](out_bn_3)
        out_conv_4 = self.net['conv_4'](out_relu_3)
        out_bn_4 = self.net['bn_4'](out_conv_4)
        out_relu_4 = self.net['relu_4'](out_bn_4)
        out_conv5 = self.net['conv_5'](out_relu_4)
        out_upsample_4x = self.net['upsample_4x'](out_conv5)
        out_skip_6 = self.net['skip_6'](out_relu_2)
        out_betterfeat = self.net['sum_6'](out_skip_6,out_upsample_4x)
        out_upsample_2x = self.net['upsample_2x'](out_betterfeat)
        out = out_upsample_2x

        # return the final activation volume
        return out
