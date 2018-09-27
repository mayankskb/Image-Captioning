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
import csv
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
from Vocab_builder import Vocab_builder
from Preprocessing import load_Captions
from DataLoader import DataLoader, shuffle_data
from Encoder import Encoder
from Decoder import DecoderRNN


if __name__ == '__main__':
    
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
    
    # Transforming the input
    #  -  Resizing the image, 
    #  -  Converting it into a tensor object,
    #  -  Normalizing it with Mean and standard deviation
    transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),
                                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataloader = DataLoader(dir_path = train_dir, vocab = vocab, transform = transforms)

    data = dataloader.gen_data()
    vocab_size = vocab.index
    hidden_dim = 512

    learning_rate = 1e-3
    embedding_dim = 512

    # Initializing Encoder and Decoder Network passing appropriate arguments
    encoder = Encoder()
    decoder = DecoderRNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = vocab_size)

    # Converting tensors into cuda based tensors if available
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Concatenating the parameters of Encoder and Decoder Network into one
    params = list(encoder.linear.parameters()) + list(decoder.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = params, lr = learning_rate)

    num_epoch = 1000
    save_every = 10

    print('-' * 100)
    print('Starting training network')
    print('-' * 100)
    
    file = open('Losses.txt', 'a')

    for epoch in range(num_epoch):
        # Suffling image per epoch to get training order different
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
        file.write(str(avg_loss.item()))
        file.write(',')
        print('epoch %d avg_loss %f time %.2f mins'%(epoch, avg_loss, (toc-tic)/60))
        if epoch % save_every == 0:
            torch.save(encoder.state_dict(), os.path.join('../Model Training/', 'iter_%d_encoder.pt'%(epoch)))
            torch.save(decoder.state_dict(), os.path.join('../Model Training/', 'iter_%d_decoder.pt'%(epoch)))

            
            img = str(input('Enter the image Path : '))

            img = transforms(Image.open(img))
            img = img.unsqueeze(0)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            else:
                img = Variable(img)

            decoder_test = DecoderRNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = vocab_size)
            decoder_test.load_state_dict(torch.load(os.path.join('../Model Training/', 'iter_%d_decoder.pt'%(epoch)))) 
    

            enc_out = encoder(img)
            dec_out = decoder_test(img)
            print(vocab.get_sentence(dec_out))