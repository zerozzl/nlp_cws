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
from transformers import BertTokenizer

from bert.model import CWS
from utils.utils import setup_seed
from utils.dataloader import SIGHANDataset, TAG_TO_ID, convert_ids_to_tags, decode_sents
from utils import modelloader, evaluator


def data_collate_fn(data):
    data = np.array(data)

    sents = torch.LongTensor(np.array(data[:, 0].tolist()))
    tags = torch.LongTensor(np.array(data[:, 1].tolist()))
    masks = torch.BoolTensor(np.array(data[:, 2].tolist()))
    segments = torch.zeros(sents.size(), dtype=torch.long)
    sents_len = data[:, 3].tolist()

    return sents, tags, masks, segments, sents_len


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()
        sents, tags, masks, segments, _ = data

        sents = sents.cpu() if args.use_cpu else sents.cuda()
        tags = tags.cpu() if args.use_cpu else tags.cuda()
        masks = masks.cpu() if args.use_cpu else masks.cuda()
        segments = segments.cpu() if args.use_cpu else segments.cuda()

        loss = model(sents, segments, masks, decode=False, tags=tags)
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
            sents, tags, masks, segments, sents_len = data

            sents = sents.cpu() if args.use_cpu else sents.cuda()
            tags = tags.cpu() if args.use_cpu else tags.cuda()
            masks = masks.cpu() if args.use_cpu else masks.cuda()
            segments = segments.cpu() if args.use_cpu else segments.cuda()

            preds = model(sents, segments, masks)

            sents = sents.cpu().numpy()
            tags = tags.cpu().numpy()

            sents_ch = [tokenizer.convert_ids_to_tokens(sents[i])[1:sents_len[i] + 1] for i in range(len(sents_len))]
            tags_gold = [convert_ids_to_tags(tags[i][1:], sents_len[i]) for i in range(len(sents_len))]
            tags_pred = [convert_ids_to_tags(preds[i][1:], sents_len[i]) for i in range(len(sents_len))]

            gold_ans = decode_sents(sents_ch, tags_gold)
            pred_ans = decode_sents(sents_ch, tags_pred)

            gold_answers.extend(gold_ans)
            pred_answers.extend(pred_ans)

    pre, rec, f1 = evaluator.evaluate(output_path, epoch, gold_answers, pred_answers)
    return pre, rec, f1


def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    setup_seed(0)

    output_path = '%s/%s' % (args.output_path, args.task)
    if args.bert_freeze:
        output_path += '_freeze'
    if args.use_crf:
        output_path += '_crf'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tokenizer = BertTokenizer.from_pretrained('%s/vocab.txt' % args.pretrained_bert_path)

    logging.info("loading dataset")
    train_dataset = SIGHANDataset('%s/sighan2005-%s/train.txt' % (args.data_path, args.task), args.max_inp_size,
                                  do_pad=True, do_to_id=True, char_tokenizer=tokenizer, for_bert=True, debug=args.debug)
    dev_dataset = SIGHANDataset('%s/sighan2005-%s/dev.txt' % (args.data_path, args.task), args.max_inp_size,
                                do_pad=True, do_to_id=True, char_tokenizer=tokenizer, for_bert=True, debug=args.debug)
    test_dataset = SIGHANDataset('%s/sighan2005-%s/test.txt' % (args.data_path, args.task), args.max_inp_size,
                                 do_pad=True, do_to_id=True, char_tokenizer=tokenizer, for_bert=True, debug=args.debug)

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
        model = CWS('%s/config.json' % args.pretrained_bert_path, '%s/pytorch_model.bin' % args.pretrained_bert_path,
                    len(TAG_TO_ID), args.bert_freeze, args.use_crf)
        model = model.cpu() if args.use_cpu else model.cuda()

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
            args, output_path, dev_dataloader, tokenizer, model, epoch)

        logging.info('epoch[%s/%s], train_loss: %s' % (
            epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], val_precision: %s, val_recall: %s, val_f1: %s' % (
            epoch, args.epoch_size, dev_pre, dev_rec, dev_f1))

        modelloader.save(output_path, 'last.pth', model, optimizer, epoch, dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1

            test_pre, test_rec, test_f1 = evaluate(
                args, output_path, test_dataloader, tokenizer, model, epoch)
            logging.info('epoch[%s/%s], test_precision: %s, test_recall: %s, test_f1: %s' % (
                epoch, args.epoch_size, test_pre, test_rec, test_f1))

            modelloader.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

            with codecs.open('%s/best_score.txt' % output_path, 'w', 'utf-8') as fout:
                fout.write('Dev precision: %s, recall: %s, f1: %s\n' % (dev_pre, dev_rec, dev_f1))
                fout.write('Test precision: %s, recall: %s, f1: %s\n' % (test_pre, test_rec, test_f1))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task',
                        default='pku')
    parser.add_argument('--data_path', dest='data_path',
                        default='../data/datasets/')
    parser.add_argument('--pretrained_bert_path', dest='pretrained_bert_path',
                        default='../data/bert/bert-base-chinese/')
    parser.add_argument('--pretrained_model_path', dest='pretrained_model_path',
                        default=None)
    parser.add_argument('--output_path', dest='output_path',
                        default='../runtime/bert/')
    parser.add_argument('--bert_freeze', dest='bert_freeze', type=bool,
                        default=False)
    parser.add_argument('--use_crf', dest='use_crf', type=bool,
                        default=False)
    parser.add_argument('--max_inp_size', dest='max_inp_size', type=int,
                        default=300)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32)
    parser.add_argument('--epoch_size', dest='epoch_size', type=int,
                        default=10)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        default=5e-5)
    parser.add_argument('--lr_warmup_proportion', dest='lr_warmup_proportion', type=float,
                        default=0.1)
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
