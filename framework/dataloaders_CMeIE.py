import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from random import choice
from transformers import BertTokenizer

BERT_MAX_LEN = 512


def seq_padding(batch, padding=0):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    return np.array([
        np.concatenate([x, [padding] * (max_length - len(x))]) if len(x) < max_length else x for x in batch
    ])


class REDataset(Dataset):
    def __init__(self, path, rel_dict_path, pretrain_path):
        super().__init__()
        self.path = path
        self.data = json.load(open(path, encoding='utf-8'))
        id2rel, rel2id = json.load(open(rel_dict_path, encoding='utf-8'))
        id2rel = {int(i): j for i, j in id2rel.items()}
        self.num_rels = len(id2rel)
        self.id2rel = id2rel
        self.rel2id = rel2id
        self.maxlen = 512
        self.berttokenizer = BertTokenizer.from_pretrained(pretrain_path)

        for sent in self.data:
            triple_list = []
            for triple in sent['triple_list']:
                triple_list.append(tuple(triple))
            sent['triple_list'] = triple_list

        self.data_length = len(self.data)
        print("new loder")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        ret = self._tokenizer(item)
        return ret

    def _tokenizer(self, line):
        text = ' '.join(line['text'].split()[:self.maxlen])
        tokens = self._tokenize(text)
        if len(tokens) > self.maxlen:
            tokens = tokens[:self.maxlen]
        text_len = len(tokens)
        s2ro_map = {}
        for triple in line['triple_list']:
            triple = (self._tokenize(triple[0])[1:-1], triple[1], self._tokenize(triple[2])[1:-1])
            sub_head_idx = self.find_head_idx(tokens, triple[0])
            obj_head_idx = self.find_head_idx(tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                if sub not in s2ro_map:
                    s2ro_map[sub] = []
                s2ro_map[sub].append((obj_head_idx,
                                      obj_head_idx + len(triple[2]) - 1,
                                      self.rel2id[triple[1]]))
        if s2ro_map:
            token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            for s in s2ro_map:
                sub_heads[s[0]] = 1
                sub_tails[s[1]] = 1
            sub_head, sub_tail = choice(list(s2ro_map.keys()))
            obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
            for ro in s2ro_map.get((sub_head, sub_tail), []):
                obj_heads[ro[0]][ro[2]] = 1
                obj_tails[ro[1]][ro[2]] = 1
            att_mask = torch.ones(len(token_ids)).long()
            return [token_ids, att_mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails]
        else:
            return []

    def _tokenize(self, text):
        tokens = self.berttokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens

    @staticmethod
    def find_head_idx(source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    def metric(self, model, h_bar=0.5, t_bar=0.5, exact_match=False, output_path=None):
        save_data = []
        orders = ['subject', 'relation', 'object']
        correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
        for line in tqdm(self.data):
            Pred_triples = set(self.extract_items(model, line['text'], h_bar=h_bar, t_bar=t_bar))
            Gold_triples = set(tuple(triple) for triple in line['triple_list'])  # 确保Gold_triples是元组形式

            Pred_triples_eval, Gold_triples_eval = self.partial_match(Pred_triples,
                                                                      Gold_triples) if not exact_match else (
                Pred_triples, Gold_triples)

            correct_num += len(Pred_triples_eval & Gold_triples_eval)
            predict_num += len(Pred_triples_eval)
            gold_num += len(Gold_triples_eval)

            if output_path:
                temp = {
                    'text': line['text'],
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in Gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in Pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                    ]
                }
                save_data.append(temp)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False)

        precision = correct_num / predict_num
        recall = correct_num / gold_num
        f1_score = 2 * precision * recall / (precision + recall)

        print(f'correct_num: {correct_num}\npredict_num: {predict_num}\ngold_num: {gold_num}')
        print('f1: %.4f, precision: %.4f, recall: %.4f' % (f1_score, precision, recall))
        return precision, recall, f1_score

    def partial_match(self, pred_set, gold_set):
        pred = {(i[0], i[1], i[2]) for i in pred_set}  # 对于中文，不进行特殊处理
        gold = {(i[0], i[1], i[2]) for i in gold_set}  # 对于中文，不进行特殊处理
        return pred, gold

    def extract_items(self, model, text_in, h_bar=0.5, t_bar=0.5):
        tokens = self._tokenize(text_in)
        token_ids = self.berttokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > self.maxlen:
            token_ids = token_ids[:self.maxlen]
        token_ids_np = np.array([token_ids])
        token_ids = torch.tensor(token_ids).unsqueeze(0).long().cuda()
        sub_heads_logits, sub_tails_logits = model.predict_sub(token_ids)
        sub_heads, sub_tails = np.where(sub_heads_logits[0].cpu() > h_bar)[0], \
        np.where(sub_tails_logits[0].cpu() > h_bar)[0]
        subjects = []
        for sub_head in sub_heads:
            sub_tail = sub_tails[sub_tails >= sub_head]
            if len(sub_tail) > 0:
                sub_tail = sub_tail[0]
                subject = tokens[sub_head: sub_tail + 1]
                subjects.append((subject, sub_head, sub_tail))
        if subjects:
            triple_list = []
            token_ids = torch.from_numpy(np.repeat(token_ids_np, len(subjects), 0)).long().cuda()
            sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
            sub_heads, sub_tails = torch.from_numpy(sub_heads).cuda(), torch.from_numpy(sub_tails).cuda()
            obj_heads_logits, obj_tails_logits = model.predict_obj(token_ids, sub_heads, sub_tails)

            for i, subject in enumerate(subjects):
                sub = subject[0]
                sub = self.cat_wordpiece(sub)
                obj_heads, obj_tails = np.where(obj_heads_logits[i].cpu() > t_bar), np.where(
                    obj_tails_logits[i].cpu() > t_bar)
                for obj_head, rel_head in zip(*obj_heads):
                    for obj_tail, rel_tail in zip(*obj_tails):
                        if obj_head <= obj_tail and rel_head == rel_tail:
                            rel = self.id2rel[rel_head]
                            obj = tokens[obj_head: obj_tail + 1]
                            obj = self.cat_wordpiece(obj)
                            triple_list.append((sub, rel, obj))
                            break
            triple_set = set()
            for s, r, o in triple_list:
                triple_set.add((s, r, o))
            return list(triple_set)
        else:
            return []

    def cat_wordpiece(self, tokens):
        new_tokens = []
        for token in tokens:
            if token.startswith('##'):
                if new_tokens:
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_tokens.append(token[2:])
            else:
                new_tokens.append(token)
        return ''.join(new_tokens)

    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))
        token_ids, att_mask, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails = data
        num_rels = obj_heads[0].shape[1]
        sub_heads_batch = torch.from_numpy(seq_padding(sub_heads))
        sub_tails_batch = torch.from_numpy(seq_padding(sub_tails))
        obj_heads_batch = torch.from_numpy(seq_padding(obj_heads, np.zeros(num_rels)))
        obj_tails_batch = torch.from_numpy(seq_padding(obj_tails, np.zeros(num_rels)))
        sub_head_batch, sub_tail_batch = torch.from_numpy(np.array(sub_head)).long(), torch.from_numpy(
            np.array(sub_tail)).long()

        return  sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch


def RELoader(path, rel2id, pretrain_path, batch_size, shuffle, num_workers=8, collate_fn=REDataset.collate_fn):
    dataset = REDataset(path=path, rel_dict_path=rel2id, pretrain_path=pretrain_path)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader


class SentenceREDataset(Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, path, rel2id_path, pretrain_path):
        super().__init__()
        self.path = path
        self.berttokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.rel2id = json.load(open(rel2id_path))
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        f = open(path, encoding='utf-8')
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                self.data.append(eval(line))
        f.close()
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item))
        return [self.rel2id[item['relation']]] + seq

    def tokenizer(self, item):
        sentence = item['token']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        ent0 = ' '.join(sentence[pos_head[0]:pos_head[1]])
        ent1 = ' '.join(sentence[pos_tail[0]:pos_tail[1]])
        sent = ' '.join(sentence)
        re_tokens = self._tokenize(sent)

        if len(re_tokens) > BERT_MAX_LEN:
            re_tokens = re_tokens[:BERT_MAX_LEN]

        ent0 = self._tokenize(ent0)[1:-1]
        ent1 = self._tokenize(ent1)[1:-1]

        heads_s = self.find_head_idx(re_tokens, ent0)
        heads_e = heads_s + len(ent0) - 1

        tails_s = self.find_head_idx(re_tokens, ent1)
        tails_e = tails_s + len(ent1) - 1

        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        heads = torch.zeros(avai_len).float()
        tails = torch.zeros(avai_len).float()

        for i in range(avai_len):
            if i >= heads_s and i <= heads_e:
                heads[i] = 1.0
            if i >= tails_s and i <= tails_e:
                tails[i] = 1.0

        indexed_tokens = torch.tensor(indexed_tokens).long()
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[:avai_len] = 1

        return indexed_tokens, heads, tails, att_mask

    def _tokenize(self, tokens):
        re_tokens = ['[CLS]']
        re_tokens += self.berttokenizer.tokenize(tokens)
        re_tokens.append('[SEP]')
        return re_tokens

    def find_head_idx(self, source, target):
        target_len = len(target)
        for i in range(len(source)):
            if source[i: i + target_len] == target:
                return i
        return -1

    @staticmethod
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        batch_labels = torch.tensor(labels).long()
        seqs = data[1:]
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(pad_sequence(seq, batch_first=True, padding_value=0))
        return [batch_labels] + batch_seqs


def SentenceRELoader(path, rel2id, pretrain_path, batch_size, shuffle, num_workers=8,
                     collate_fn=SentenceREDataset.collate_fn):
    dataset = SentenceREDataset(path=path, rel2id_path=rel2id, pretrain_path=pretrain_path)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader
