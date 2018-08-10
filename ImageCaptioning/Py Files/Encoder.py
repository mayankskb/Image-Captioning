#####################################################################################################################################
#                                                                                                                                   #
#                                                    ENCODER MODULE                                                                 #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embedding_dim = 512):
        super(Encoder, self).__init__()
        self.module = models.resnet152(pretrained=True)
        self.linear = nn.Linear(self.module.fc.in_features, embedding_dim)
        self.module.fc = self.linear
    
    def forward(self, images):
        embed = self.module(images)
        return embed