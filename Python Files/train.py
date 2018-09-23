#####################################################################################################################################
#                                                                                                                                   #
#                                                   TRAINING MODULE                                                                 #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import os
import time
import pickle
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.autograd import Variable
from Vocab_builder import Vocab_builder
from Preprocessing import load_Captions
from DataLoader import DataLoader, shuffle_data
from Encoder import Encoder
from Decoder import DecoderRNN


# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials



if __name__ == '__main__':
    
    # 1. Authenticate and create the PyDrive client.
    # auth.authenticate_user()
    # gauth = GoogleAuth()
    # gauth.credentials = GoogleCredentials.get_application_default()
    # drive = GoogleDrive(gauth)
    
    train_dir = '../Processed Data/dev'
    print('STARTING THE TRAINING PHASE ...........')
    # Building the vocabulary using the caption file available at the train directory
    print('-' * 100)
    print('Building Vocabulary')
    caption_dict = load_Captions(train_dir)
    vocab = Vocab_builder(caption_dict = caption_dict, threshold = 5)
    print('-' * 100)

    # Dictionary Dumping
    with open(os.path.join('../Model Training', 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
        print('-' * 100)
        print('DUMPED DICTIONARY')
        print('-' * 100)
    
    transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataloader = DataLoader(dir_path = train_dir, vocab = vocab, transform = transforms)

    data = dataloader.gen_data()
    vocab_size = vocab.index
    hidden_dim = 512

    learning_rate = 1e-3
    embedding_dim = 512

    encoder = Encoder()
    decoder = DecoderRNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = vocab_size)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    params = list(encoder.linear.parameters()) + list(decoder.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = params, lr = learning_rate)

    num_epoch = 1000
    save_every = 10

    print('-' * 100)
    print('Starting training network')
    print('-' * 100)
    
    for epoch in range(num_epoch):
        shuffled_images, shuffled_captions = shuffle_data(data = data)
        num_captions = len(shuffled_captions)

        loss_list = []

        tic = time.time()
        for i in range(num_captions):
            image_id = shuffled_images[i]
            image = dataloader.get_image(image_id)
            image = image.unsqueeze(0)

            if torch.cuda.is_available():
                image = Variable(image).cuda()
                caption = torch.cuda.LongTensor(shuffled_captions[i])
            else:
                image = Variable(image)
                caption = torch.LongTensor(shuffled_captions[i])

            caption_train = caption[:-1]

            encoder.zero_grad()
            decoder.zero_grad()

            encod_out = encoder(image)
            decoder_output = decoder(encod_out, caption_train)

            loss = criterion(decoder_output, caption)
            loss.backward()

            optimizer.step()

            loss_list.append(loss)

        toc = time.time()

        avg_loss = torch.mean(torch.Tensor(loss_list))
        print('epoch %d avg_loss %f time %.2f mins'%(epoch, avg_loss, (toc-tic)/60))
        if epoch % save_every == 0:
            torch.save(encoder.state_dict(), os.path.join('../Model Training/', 'iter_%d_encoder.pt'%(epoch)))
            torch.save(decoder.state_dict(), os.path.join('../Model Training/', 'iter_%d_decoder.pt'%(epoch)))
            
            # encoder_file = drive.CreateFile({'title' : os.path.join('../Model Training/', 'iter_%d_encoder.pkl'%(epoch))})
            # encoder_file.SetContentFile(os.path.join('../Model Training/', 'iter_%d_encoder.pkl'%(epoch)))
            # encoder_file.Upload()
            
            # decoder_file = drive.CreateFile({'title' : os.path.join('../Model Training/', 'iter_%d_decoder.pkl'%(epoch))})
            # decoder_file.SetContentFile(os.path.join('../Model Training/', 'iter_%d_decoder.pkl'%(epoch)))
            # decoder_file.Upload()
