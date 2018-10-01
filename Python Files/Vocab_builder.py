#####################################################################################################################################
#                                                                                                                                   #
#                                                  VOCABULARY - BUILDER CLASS                                                       #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import nltk
from collections import Counter


class Vocab_builder():
    '''
        Vocabulary Builder Class which will map words to index and index to words based on the threshold 
        which is the number of minimum count required for a word to be included into the dictionary
    '''
    def __init__(self, caption_dict, threshold):
        self.word2ind = {}
        self.ind2word = {}
        self.index = 0
        self.build_vocab(caption_dict, threshold)
    
    def add_words(self, word):
        if word not in self.word2ind:
            self.word2ind[word] = self.index
            self.ind2word[self.index] = word
            self.index += 1
    
    def get_id(self, word):
        if word in self.word2ind:
            return self.word2ind[word]
        else:
            print('Word not found in dictionary')
            return self.word2ind['<NULL>']
    
    def get_word(self, index):
        return self.ind2word[index]
    
    def build_vocab(self, caption_dict, threshold):
        counter = Counter()
        tokens = []
        
        for _, captions in caption_dict.items():
            for caption in captions:
                caption_token = nltk.tokenize.word_tokenize(caption.lower())
                
                tokens.extend(caption_token)
        
        counter.update(tokens)
        words = [word for word, count in counter.items() if count > threshold]
        
        self.add_words('<NULL>')
        self.add_words('<start>')
        self.add_words('<end>')
        
        for word in words:
            self.add_words(word)
    
    def get_sentence(self, ids_list):
        sent = ''
        for cur_id in ids_list:
            cur_word = self.ind2word[cur_id.item()]
            sent += ' ' + cur_word
            if cur_word == '<end>':
                break
        return sent