import codecs
import numpy as np
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES = '<S>'
TAG_TO_ID = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}


class SIGHANDataset(Dataset):
    def __init__(self, data_path, max_len=0,
                 do_pad=False, pad_token=TOKEN_PAD,
                 do_to_id=False, char_tokenizer=None,
                 for_bert=False, do_sort=False,
                 add_word_feature=False, word_tokenizer=None,
                 debug=False):
        self.data = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                sent, tag, sent_len = process_sent(line, max_len)

                if for_bert:
                    sent = [TOKEN_CLS] + sent
                    tag = ['S'] + tag

                if do_pad:
                    sent, mask = pad_sent(sent, max_len, pad_token)
                    tag = pad_tag(tag, max_len)
                else:
                    mask = [1] * len(sent)

                if add_word_feature:
                    sent = [TOKEN_EDGES] + sent + [TOKEN_EDGES]
                    sent = [[sent[i]] + [sent[i - 1] + sent[i]] + [sent[i] + sent[i + 1]] for i in
                            range(1, len(sent) - 1)]

                if do_to_id:
                    if add_word_feature:
                        sent = np.array(sent)
                        uni_id = char_tokenizer.convert_tokens_to_ids(sent[:, 0])
                        bi_id = word_tokenizer.convert_tokens_to_ids(sent[:, 1:])
                        sent = np.concatenate((uni_id, bi_id), axis=1).tolist()
                    else:
                        sent = char_tokenizer.convert_tokens_to_ids(sent)
                    tag = convert_tags_to_ids(tag)

                self.data.append([sent, tag, mask, sent_len])

                if debug:
                    if len(self.data) >= 10:
                        break

        if do_sort:
            self.data = sorted(self.data, key=lambda x: x[3], reverse=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_max_sent_length(self):
        max_length = 0
        for sent in self.data:
            if len(sent) > max_length:
                max_length = len(sent)
        return max_length

    def get_sent_length_info(self):
        length_map = {100: 0, 200: 0, 300: 0, 400: 0, 500: 0, 600: 0, 700: 0}
        for sent in self.data:
            for leng in length_map:
                if len(sent) <= leng:
                    length_map[leng] = length_map[leng] + 1
                    break
        return length_map


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append([self.token_to_id.get(token, self.token_to_id[unk_token])])
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_len):
        tokens = [self.id_to_token[i] for i in ids]
        if max_len > 0:
            tokens = tokens[:max_len]
        return tokens


def process_sent(sent, max_len=0):
    tokens = []
    tag = []
    for word in sent.split():
        tokens.extend(list(word))
        if len(word) == 1:
            tag.append('S')
        else:
            tag.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])

    if max_len > 0:
        tokens = tokens[:max_len]
        tags = tag[:max_len]
    return tokens, tags, len(tokens)


def pad_sent(sent, max_len, pad_token=TOKEN_PAD):
    sent_pad = sent[:max_len] + [pad_token] * (max_len - len(sent))
    mask = [1] * len(sent[:max_len]) + [0] * (max_len - len(sent))
    return sent_pad, mask


def pad_tag(tag, max_len):
    return tag[:max_len] + ['S'] * (max_len - len(tag))


def load_pretrain_embedding(filepath, add_pad=False, pad_token=TOKEN_PAD, add_unk=False, unk_token=TOKEN_UNK):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        meta_info = fin.readline().strip().split()  # title
        # vocav_size = int(meta_info[0])
        embed_size = int(meta_info[1])

        token_to_id = {}
        embed = []

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

    return token_to_id, embed


def convert_tags_to_ids(tags):
    return [TAG_TO_ID.get(tag) for tag in tags]


def convert_ids_to_tags(ids, max_len):
    tags = [ID_TO_TAG[i] for i in ids]
    if max_len > 0:
        tags = tags[:max_len]
    return tags


def decode_sents(seqs, stags):
    sents = []
    for seq, stag in zip(seqs, stags):
        words = []
        word = []
        for c, tag in zip(seq, stag):
            word.append(c)
            if tag == 'S' or tag == 'E':
                words.append(''.join(word))
                word = []
        words.append(''.join(word))
        sents.append(words)
    return sents
