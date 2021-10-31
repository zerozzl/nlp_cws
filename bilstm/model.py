import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torchcrf import CRF


class CWS(nn.Module):

    def __init__(self, num_tags, char_vocab_size, char_embed_size, stack_lstm, hidden_size, num_hidden_layer,
                 embed_dropout_rate, lstm_dropout_rate, use_crf,
                 use_word_embed, word_vocab_size, word_embed_size, embed_freeze, use_cpu):
        super(CWS, self).__init__()
        self.use_crf = use_crf
        self.stack_lstm = stack_lstm
        self.use_word_embed = use_word_embed
        self.use_cpu = use_cpu

        input_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)
        self.embed_dropout = nn.Dropout(embed_dropout_rate)

        if use_word_embed:
            input_size += word_embed_size * 2
            self.word_embedding = nn.Embedding(word_vocab_size, word_embed_size)

        if stack_lstm:
            self.lstm_b = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_hidden_layer,
                                  batch_first=True,
                                  dropout=lstm_dropout_rate,
                                  bidirectional=False)
            self.lstm_f = nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_hidden_layer,
                                  batch_first=True,
                                  dropout=lstm_dropout_rate,
                                  bidirectional=False)
            self.linear = nn.Linear(hidden_size, num_tags)
        else:
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_hidden_layer,
                                batch_first=True,
                                dropout=lstm_dropout_rate,
                                bidirectional=True)
            self.linear = nn.Linear(hidden_size * 2, num_tags)

        if embed_freeze:
            for param in self.char_embedding.parameters():
                param.requires_grad = False
            if use_word_embed:
                for param in self.word_embedding.parameters():
                    param.requires_grad = False

        if self.use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def init_char_embedding(self, pretrained_embeddings):
        self.char_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_word_embedding(self, pretrained_embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, x, masks, x_len, decode=True, tags=None):
        sents = []
        if self.stack_lstm:
            for sent in x:
                sents.append(np.flip(sent, axis=0))
        else:
            sents = x

        sents = [torch.LongTensor(np.array(s)) for s in sents]
        sents = rnn.pad_sequence(sents, batch_first=True)
        sents = sents.cpu() if self.use_cpu else sents.cuda()

        masks = [torch.BoolTensor(np.array(m)) for m in masks]
        masks = rnn.pad_sequence(masks, batch_first=True)
        masks = masks.cpu() if self.use_cpu else masks.cuda()

        if not decode:
            tags = [torch.LongTensor(np.array(t)) for t in tags]
            tags = rnn.pad_sequence(tags, batch_first=True)
            tags = tags.cpu() if self.use_cpu else tags.cuda()

        out = self.char_embedding(sents[:, :, 0])
        if self.use_word_embed:
            w_emb = torch.cat([self.word_embedding(sents[:, :, i]) for i in range(1, sents.size()[2])], dim=2)
            out = torch.cat((out, w_emb), dim=2)
        out = self.embed_dropout(out)

        out = rnn.pack_padded_sequence(out, x_len, batch_first=True)
        if self.stack_lstm:
            out, _ = self.lstm_b(out)
            out, _ = rnn.pad_packed_sequence(out, batch_first=True)

            out = torch.flip(out, dims=[1])
            out = [o[torch.max(o > 0, 1)[0]] for o in out]
            out = rnn.pad_sequence(out, batch_first=True)
            out = rnn.pack_padded_sequence(out, x_len, batch_first=True)

            out, _ = self.lstm_f(out)
            out, _ = rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(out)
            out, _ = rnn.pad_packed_sequence(out, batch_first=True)

        out = self.linear(out)

        if decode:
            if self.use_crf:
                pred = self.crf.decode(out, masks)
            else:
                out = F.softmax(out, dim=2)
                out = torch.argmax(out, dim=2)
                pred = out.cpu().numpy()
            return pred
        else:
            if self.use_crf:
                loss = -self.crf(out, tags, masks)
            else:
                out_shape = out.size()
                loss = self.ce_loss(out.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                    tags.reshape(out_shape[0] * out_shape[1]))
            return loss
