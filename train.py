from encoder.bert_encoder import BERTEncoder
from models.mspre_sentence import MspRE_SEN
from models.mspre_triple import MSPRE_TR
from framework.sentence_re import Sentence_RE
from framework.triple_re import Triple_RE
from framework.triple_re_Chinese import Triple_RE_CHINESE
from configs import Config
from utils import count_params
import numpy as np
import torch
import random, argparse
torch.cuda.set_device(0)

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--dataset', default='SciERC', type=str,
                        help='specify the dataset from ["BB","CMeIE","Chemport","SciERC"]')
    args = parser.parse_args()
    dataset = args.dataset
    is_train = args.train
    config = Config()
    if config.seed is not None:
        print(config.seed)
        seed_torch(config.seed)

    if dataset == 'CMeIE':
        print('train--' + dataset + config.CMeIE_ckpt)
        config.class_nums = config.CMeIE_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base_chinese)
        model = MSPRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE_CHINESE(model,
                              train=config.CMeIE_train,
                              val=config.CMeIE_val,
                              test=config.CMeIE_test,
                              rel2id=config.CMeIE_rel2id,
                              pretrain_path=config.bert_base_chinese,
                              ckpt=config.CMeIE_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)

        framework.train_model(dataset)
        framework.load_state_dict(config.CMeIE_ckpt)
        print('test:' + config.CMeIE_ckpt)
        framework.test_set.metric(framework.model)
    elif dataset == 'BB':
        print('train--' + dataset + config.BB_ckpt)
        config.class_nums = config.BB_class
        sentence_encoder = BERTEncoder(pretrain_path=config.scibert_scivocab_cased)

        model = MSPRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE(model,
                              train=config.BB_train,
                              val=config.BB_val,
                              test=config.BB_test,
                              rel2id=config.BB_rel2id,
                              pretrain_path=config.scibert_scivocab_cased,
                              ckpt=config.BB_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)

        framework.train_model(dataset)
        framework.load_state_dict(config.BB_ckpt)
        print('test:' + config.BB_ckpt)
        framework.test_set.metric(framework.model)
    elif dataset == 'Chemport':
        print('train--' + dataset + config.Chemport_ckpt)
        config.class_nums = config.Chemport_class
        sentence_encoder = BERTEncoder(pretrain_path=config.bert_base_cased)

        model = MSPRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE(model,
                              train=config.Chemport_train,
                              val=config.Chemport_val,
                              test=config.Chemport_test,
                              rel2id=config.Chemport_rel2id,
                              pretrain_path=config.bert_base_cased,
                              ckpt=config.Chemport_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)

        framework.train_model(dataset)
        framework.load_state_dict(config.Chemport_ckpt)
        print('test:' + config.Chemport_ckpt)
        framework.test_set.metric(framework.model)
    elif dataset == 'SciERC':
        print('train--' + dataset + config.SciERC_ckpt)
        config.class_nums = config.SciERC_class
        sentence_encoder = BERTEncoder(pretrain_path=config.scibert_scivocab_cased)

        model = MSPRE_TR(sentence_encoder, config)
        count_params(model)
        framework = Triple_RE(model,
                              train=config.SciERC_train,
                              val=config.SciERC_val,
                              test=config.SciERC_test,
                              rel2id=config.SciERC_rel2id,
                              pretrain_path=config.scibert_scivocab_cased,
                              ckpt=config.SciERC_ckpt,
                              batch_size=config.batch_size,
                              max_epoch=config.epoch,
                              lr=config.lr,
                              num_workers=4)

        framework.train_model(dataset)
        framework.load_state_dict(config.SciERC_ckpt)
        print('test:' + config.SciERC_ckpt)
        framework.test_set.metric(framework.model)
    else:
        print('unkonw dataset')