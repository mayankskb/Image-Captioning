#####################################################################################################################################
#                                                                                                                                   #
#                                                   TESTING MODULE                                                                  #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import os
import time
import pickle
import json
import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.autograd import Variable
from Vocab_builder import Vocab_builder
from Preprocessing import load_Captions
from DataLoader import DataLoader, shuffle_data
from Encoder import Encoder
from Decoder import DecoderRNN


if __name__ == '__main__':
    
    test_dir = '../Processed Data/test/images'
    print('STARTING THE TESTING PHASE ...........')
    
    # Reading the vocab file
    with open(os.path.join('../Model Training', 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # Transforming the image file by Resizing, making tensor from it and then 
    # Normalizing the image by mean and standard deviation
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))])
    
    vocab_size = vocab.index
    hidden_dim = 512
    embedding_dim = 512

    # Path where the input saved module is present    
    encoder_saved_module = '../Model Training/iter_230_cnn.pkl'
    decoder_saved_module = '../Model Training/iter_230_lstm.pkl'

    # Initializing the Encoder and Decoder Network with arguments passed
    encoder = Encoder(embedding_dim = embedding_dim)
    decoder = DecoderRNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = vocab_size)

    # Reading the pretrained weights for Encoder and Decoder
    encoder.load_state_dict(torch.load(encoder_saved_module))
    decoder.load_state_dict(torch.load(decoder_saved_module)) 
    
    # Taking input from user for image to be captioned
    img = str(input('Enter the image id for which you want to get caption'))
    image_path = os.path.join(test_dir, img)
    print(image_path)

    image = transforms(Image.open(image_path))
    image = image.unsqueeze(0)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        image = Variable(image).cuda()
    else:
        image = Variable(image)
    
    # Passing the input from the network 
    encoder_out = encoder(image)
    decoder_out = decoder(encoder_out)

    # Printing the outputs
    print(vocab.get_sentence(decoder_out))