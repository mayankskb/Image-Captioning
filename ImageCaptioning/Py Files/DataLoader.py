#####################################################################################################################################
#                                                                                                                                   #
#                                                    DATA LOADER MODULE                                                             #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import os
import torch
import nltk
import json
from PIL import Image

class DataLoader():
    def __init__(self, dir_path, vocab, transform):
        self.images = None
        self.captions_dict = None
        self.vocab = vocab
        self.transform = transform
        self.load_captions(dir_path)
        self.load_images(dir_path)
    
    def load_captions(self, dir_path):
        file_path = os.path.join(dir_path, 'captions.txt')
        captions_dict = {}
        with open(file_path) as f:
            for line in f:
                line_caption = json.loads(line)
                for k, c in line_caption.items():
                    captions_dict[k] = c
        self.captions_dict = captions_dict
        
    def load_images(self, dir_path):
        file_path = os.path.join(dir_path, 'images')
        files = os.listdir(file_path)
        images = {}
        for file in files:
            extn = file.split('.')[1]
            if extn == 'jpg':
                images[file] = self.transform(Image.open(os.path.join(file_path, file)))
        self.images = images
    
    def gen_data(self):
        images = []
        captions = []
        for cur_image, cur_caption in self.captions_dict.items():
            num_captions = len(cur_caption)
            images.extend([cur_image] * num_captions)
            for caption in cur_caption:
                captions.append(self.captions2ind(caption))
                
        data = images, captions
        return data
    
    def captions2ind(self, caption):
        vocab = self.vocab
        token = nltk.tokenize.word_tokenize(caption.lower())
        vec = []
        vec.append(vocab.get_id('<start>'))
        vec.extend([vocab.get_id(word) for word in token])
        vec.append(vocab.get_id('<end>'))
        
        return vec
    
    def get_image(self, image_id):
        return self.images[image_id]      

# For shuffling the Data
def shuffle_data(data, seed = 0):
    images, captions = data
    shuffle_images = []
    shuffled_captions = []
    
    num_images = len(images)
    torch.manual_seed(seed)
    
    perm = list(torch.randperm(num_images))
    for i in range(num_images):
        shuffle_images.append(images[perm[i]])
        shuffled_captions.append(captions[perm[i]])
        
    return shuffle_images, shuffled_captions