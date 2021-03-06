#####################################################################################################################################
#                                                                                                                                   #
#                                                    DECODER MODULE                                                                 #
#                                                                                                                                   #   
#                                                                                                                                   #
#####################################################################################################################################

# Importing Requisites
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, caption):
        seq_length = len(caption) + 1
        embeds = self.word_embeddings(caption)
        embeds = torch.cat((features, embeds), 0)
        gru_out, hidden = self.gru(embeds.unsqueeze(1))
        out = self.linear(gru_out.view(seq_length, -1))
        return out
    

    def get_caption_ids(self, encod_out, hidden = None, seq_len = 20):
        inputs = encod_out.unsqueeze(1)
        ids_list = []
        for t in range(seq_len):
            gru_out, hidden = self.gru(inputs, hidden)
            # generating single word at a time
            linear_out = self.linear(gru_out.squeeze(1))
            _, predicted = linear_out.max(dim=1)
            ids_list.append(predicted)
            inputs = self.word_embeddings(predicted)
            inputs = inputs.unsqueeze(1)
        return ids_list
        