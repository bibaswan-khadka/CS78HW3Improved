from torch import nn
from sum_layer import Sum

# import all other functions you may need


class SemanticSegmentationImproved(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the improve semantic segmentation model.


        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network.

        """
        super(SemanticSegmentationImproved, self).__init__()
        
        # implement the improvement model architecture

        self.net = nn.ModuleDict()
        
        # add the hidden layers as specified in the handout
        name = netspec_opts['name']
        kernel_size = netspec_opts['kernel_size']
        num_filters = netspec_opts['num_filters']
        stride = netspec_opts['stride']
        layer_type = netspec_opts['layer_type']
        inpu = netspec_opts['input']
        pad = netspec_opts['pad']

        for index, (n,k_s,n_f,s,l_t,i,p) in enumerate(zip(name,kernel_size,num_filters,stride,layer_type,inpu,pad)):
            if l_t == "conv":
                if index == 0:
                    self.net[n] = nn.Conv2d(3,n_f,k_s,s,p)
                else:
                    self.net[n] = nn.Conv2d(num_filters[i], n_f, k_s, s,p)
            if l_t == "bn":
                self.net[n] = nn.BatchNorm2d(num_filters[i])
            if l_t == "pool":
                self.net[n] = nn.AvgPool2d(k_s, s, 0)
            if l_t == "relu":
                self.net[n] = nn.ReLU()
            if l_t == "convt":
                self.net[n] = nn.ConvTranspose2d(in_channels=num_filters[i],out_channels=n_f,kernel_size=k_s,stride=s,padding=p,groups=num_filters[i],bias=False)
            if l_t == "skip":
                self.net[n] = nn.Conv2d(num_filters[i],n_f,k_s,s,p)
            if l_t == "sum":
                self.net[n] = Sum()
            if l_t == 'drop':
                self.net[n] = nn.Dropout(p=0.2)
            


    def forward(self, x):
        """
        Define the forward propagation of the improvement model.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W), where C' is the number of classes.

        """

        # implement the forward propagation

        out_conv_1 = self.net['conv_1'](x)
        out_bn_1 = self.net['bn_1'](out_conv_1)
        out_relu_1 = self.net['relu_1'](out_bn_1)
        out_pool_1 = self.net['pool_1'](out_relu_1)
        out_conv_2 = self.net['conv_2'](out_pool_1)
        out_bn_2 = self.net['bn_2'](out_conv_2)
        out_relu_2 = self.net['relu_2'](out_bn_2)
        out_pool_2 = self.net['pool_2'](out_relu_2)
        out_conv_3 = self.net['conv_3'](out_pool_2)
        out_bn_3 = self.net['bn_3'](out_conv_3)
        out_relu_3 = self.net['relu_3'](out_bn_3)
        out_pool_3 = self.net['pool_3'](out_relu_3)
        out_conv_4 = self.net['conv_4'](out_pool_3)
        out_bn_4 = self.net['bn_4'](out_conv_4)
        out_relu_4 = self.net['relu_4'](out_bn_4)
        out_drop_1 = self.net['drop_1'](out_relu_4)
        out_conv5 = self.net['conv_5'](out_drop_1)
        out_upsample_4x = self.net['upsample_4x'](out_conv5)
        out_skip_6 = self.net['skip_6'](out_relu_2)
        out_betterfeat = self.net['sum_6'](out_skip_6,out_upsample_4x)
        out_skip_10 = self.net['skip_10'](out_relu_3)
        out_upsample_skip_10 = self.net['upsample_skip_10'](out_skip_10)
        out_sum_10 = self.net['sum_10'](out_upsample_skip_10,out_betterfeat)
        out_upsample_2x = self.net['upsample_2x'](out_sum_10)
        out = out_upsample_2x

        # return the final activation volume
        return out
