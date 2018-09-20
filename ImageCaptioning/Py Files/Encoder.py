#####################################################################################################################################
#                                                                                                                                   #
#                                                    ENCODER MODULE                                                                 #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embedding_dim = 512):
        '''
            Using Transfer Learning, loading pretrained module
            and replacing top fc layer adding a batchnorm layer
        '''
        super(Encoder, self).__init__()
        self.module = models.resnet152(pretrained=True)

        # Using module as a fixed feature extractor
        # freezing weights of the pretrained module
        # so that no attenuation can be done to their weights
        self.freeze_weights() 

        self.linear = nn.Linear(self.module.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum = 0.01)
        self.init_weights()

    def init_weights(self):
        '''initializing weights'''
        self.linear.weight.data.normal_(0.0, 0.02)  # mean, std
        self.linear.bias.data.fill_(0)

        # for xavier initialisation 
        # nn.init.xavier_normal_(self.linear.weight)
        # nn.init.xavier_normal_(self.linear.bias)

    def freeze_weights(self):
        '''
            Setting up the requires_grad as False 
            so that no gradient can be computed for them and hence 
            no change in the weights for the layers of the module
        '''

        for param in self.module.parameters():
            param.requires_grad = False

    def forward(self, images):
        '''
            Defining forward pass for the network
        '''
        features = self.module(images)
        features = self.linear(features)
        embed = self.bn(features)
        return embed