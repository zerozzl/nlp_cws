import os
import logging
from argparse import ArgumentParser
import codecs
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from conv_seg.model import CWS
from utils.utils import setup_seed
from utils.dataloader import SIGHANDataset, Tokenizer, TAG_TO_ID, load_pretrain_embedding, convert_ids_to_tags, \
    decode_sents
from utils import modelloader, evaluator


def data_collate_fn(data):
    data = np.array(data)

    sents = torch.LongTensor(np.array(data[:, 0].tolist()))
    tags = torch.LongTensor(np.array(data[:, 1].tolist()))
    masks = torch.BoolTensor(np.array(data[:, 2].tolist()))
    sents_len = data[:, 3].tolist()

    return sents, tags, masks, sents_len


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        sents, tags, masks, _ = data

        sents = sents.cpu() if args.use_cpu else sents.cuda()
        tags = tags.cpu() if args.use_cpu else tags.cuda()
        masks = masks.cpu() if args.use_cpu else masks.cuda()

        loss = model(sents, masks, decode=False, tags=tags)
        loss = loss.mean()

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, output_path, dataloader, tokenizer, model, epoch):
    gold_answers = []
    pred_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            sents, tags, masks, sents_len = data

            sents = sents.cpu() if args.use_cpu else sents.cuda()
            tags = tags.cpu() if args.use_cpu else tags.cuda()
            masks = masks.cpu() if args.use_cpu else masks.cuda()

            preds = model(sents, masks)

            sents = sents.cpu().numpy()
            tags = tags.cpu().numpy()

            sents = sents[:, :, 0]

            sents_ch = [tokenizer.convert_ids_to_tokens(sents[i], sents_len[i]) for i in range(len(sents_len))]
            tags_gold = [convert_ids_to_tags(tags[i], sents_len[i]) for i in range(len(sents_len))]
            tags_pred = [convert_ids_to_tags(preds[i], sents_len[i]) for i in range(len(sents_len))]

            gold_ans = decode_sents(sents_ch, tags_gold)
            pred_ans = decode_sents(sents_ch, tags_pred)

            gold_answers.extend(gold_ans)
            pred_answers.extend(pred_ans)

    pre, rec, f1 = evaluator.evaluate(output_path, epoch, gold_answers, pred_answers)
    return pre, rec, f1


def main(args):
    if args.debug:
        args.batch_size = 3

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    setup_seed(0)

    output_path = '%s/%s' % (args.output_path, args.task)
    if args.embed_freeze:
        output_path += '_embfix'
    if args.use_word_embed:
        output_path += '_wemb'
    if args.use_crf:
        output_path += '_crf'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info("loading pretrained embedding")
    char_to_id, pretrain_char_embed = load_pretrain_embedding(args.pretrained_char_emb_path,
                                                              add_pad=True, add_unk=True)
    char_tokenizer = Tokenizer(char_to_id)

    word_tokenizer = None
    word_to_id = {}
    if args.use_word_embed:
        word_to_id, pretrain_word_embed = load_pretrain_embedding(args.pretrained_word_emb_path,
                                                                  add_pad=True, add_unk=True)
        word_tokenizer = Tokenizer(word_to_id)

    logging.info("loading dataset")
    train_dataset = SIGHANDataset('%s/sighan2005-%s/train.txt' % (args.data_path, args.task), args.max_inp_size,
                                  do_pad=True, do_to_id=True, char_tokenizer=char_tokenizer,
                                  add_word_feature=args.use_word_embed, word_tokenizer=word_tokenizer,
                                  debug=args.debug)
    dev_dataset = SIGHANDataset('%s/sighan2005-%s/dev.txt' % (args.data_path, args.task), args.max_inp_size,
                                do_pad=True, do_to_id=True, char_tokenizer=char_tokenizer,
                                add_word_feature=args.use_word_embed, word_tokenizer=word_tokenizer,
                                debug=args.debug)
    test_dataset = SIGHANDataset('%s/sighan2005-%s/test.txt' % (args.data_path, args.task), args.max_inp_size,
                                 do_pad=True, do_to_id=True, char_tokenizer=char_tokenizer,
                                 add_word_feature=args.use_word_embed, word_tokenizer=word_tokenizer,
                                 debug=args.debug)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset))
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                    sampler=DistributedSampler(dev_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                    shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = modelloader.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        model = CWS(len(TAG_TO_ID), len(char_to_id), args.char_embed_size,
                    args.num_hidden_layer, args.channel_size, args.kernel_size, args.dropout_rate,
                    args.use_crf, args.use_word_embed, len(word_to_id), args.word_embed_size,
                    args.embed_freeze)
        model = model.cpu() if args.use_cpu else model.cuda()

        model.init_char_embedding(np.array(pretrain_char_embed))
        if args.use_word_embed:
            model.init_word_embedding(np.array(pretrain_word_embed))

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)

        dev_pre, dev_rec, dev_f1 = evaluate(
            args, output_path, dev_dataloader, char_tokenizer, model, epoch)

        logging.info('epoch[%s/%s], train_loss: %s' % (
            epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], val_precision: %s, val_recall: %s, val_f1: %s' % (
            epoch, args.epoch_size, dev_pre, dev_rec, dev_f1))

        modelloader.save(output_path, 'last.pth', model, optimizer, epoch, dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1

            test_pre, test_rec, test_f1 = evaluate(
                args, output_path, test_dataloader, char_tokenizer, model, epoch)
            logging.info('epoch[%s/%s], test_precision: %s, test_recall: %s, test_f1: %s' % (
                epoch, args.epoch_size, test_pre, test_rec, test_f1))

            modelloader.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

            with codecs.open('%s/best_score.txt' % output_path, 'w', 'utf-8') as fout:
                fout.write('Dev precision: %s, recall: %s, f1: %s\n' % (dev_pre, dev_rec, dev_f1))
                fout.write('Test precision: %s, recall: %s, f1: %s\n' % (test_pre, test_rec, test_f1))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task',
                        default='msr')
    parser.add_argument('--data_path', dest='data_path',
                        default='../data/datasets/')
    parser.add_argument('--pretrained_char_emb_path', dest='pretrained_char_emb_path',
                        default='../data/embeddings/news_tensite.w2v200')
    parser.add_argument('--pretrained_word_emb_path', dest='pretrained_word_emb_path',
                        default='../data/embeddings/news_tensite.msr.words.w2v50')
    parser.add_argument('--pretrained_model_path', dest='pretrained_model_path',
                        default=None)
    parser.add_argument('--output_path', dest='output_path',
                        default='../runtime/conv_seg')
    parser.add_argument('--use_crf', dest='use_crf', type=bool,
                        default=False)
    parser.add_argument('--use_word_embed', dest='use_word_embed', type=bool,
                        default=False)
    parser.add_argument('--max_inp_size', dest='max_inp_size', type=int,
                        default=300)
    parser.add_argument('--char_embed_size', dest='char_embed_size', type=int,
                        default=200)
    parser.add_argument('--word_embed_size', dest='word_embed_size', type=int,
                        default=50)
    parser.add_argument('--embed_freeze', dest='embed_freeze', type=bool,
                        default=False)
    parser.add_argument('--num_hidden_layer', dest='num_hidden_layer', type=int,
                        default=5)
    parser.add_argument('--channel_size', dest='channel_size', type=int,
                        default=200)
    parser.add_argument('--kernel_size', dest='kernel_size', type=int,
                        default=3)
    parser.add_argument('--dropout_rate', dest='dropout_rate', type=float,
                        default=0.2)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=512)
    parser.add_argument('--epoch_size', dest='epoch_size', type=int,
                        default=100)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=0.001)
    parser.add_argument('--lr_warmup_proportion', dest='lr_warmup_proportion', type=float,
                        default=0.2)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', dest='use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', dest='multi_gpu', type=bool, help='run with: -m torch.distributed.launch',
                        default=True)
    parser.add_argument('--local_rank', dest='local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', dest='debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)
