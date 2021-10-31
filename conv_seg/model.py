import torch
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF


class ConvGluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        super(ConvGluBlock, self).__init__()
        self.conv_d = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=1))
        self.conv_g = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=1))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv_d(x) * torch.sigmoid(self.conv_g(x))
        out = self.dropout(out)
        return out


class CWS(nn.Module):

    def __init__(self, num_tags, char_vocab_size, char_embed_size, num_hidden_layer, channel_size, kernel_size,
                 dropout_rate, use_crf, use_word_embed, word_vocab_size, word_embed_size, embed_freeze):
        super(CWS, self).__init__()
        self.use_crf = use_crf
        self.use_word_embed = use_word_embed

        input_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        if use_word_embed:
            input_size += word_embed_size * 2
            self.word_embedding = nn.Embedding(word_vocab_size, word_embed_size)

        self.convs = nn.ModuleList(
            [ConvGluBlock(input_size, channel_size, kernel_size, dropout_rate)]
            + [ConvGluBlock(channel_size, channel_size, kernel_size, dropout_rate) for _ in
               range(num_hidden_layer - 1)])
        self.linear = nn.Linear(channel_size, num_tags)

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

    def forward(self, x, masks, decode=True, tags=None):
        out = self.char_embedding(x[:, :, 0])
        if self.use_word_embed:
            w_emb = torch.cat([self.word_embedding(x[:, :, i]) for i in range(1, x.size()[2])], dim=2)
            out = torch.cat((out, w_emb), dim=2)
        out = self.dropout(out)

        out = out.permute(0, 2, 1)
        for conv in self.convs:
            out = conv(out)
        out = out.permute(0, 2, 1)

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
