import os
import json
import random


class Config(object):
    def __init__(self):
        root_path = 'datasets'
        self._get_path(root_path)
        self.batch_size = 6
        self.max_length = 100
        self.epoch = 150
        self.lr = 1e-1
        
        self.patience = 9 #early stopping patience level
        self.training_criteria = 'micro_f1' #or 'macro_f1'

        self.gat_layers = 2
        self.hidden_size = 768

        self.BB_class = 2
        self.Chemport_class = 5
        self.SciERC_class = 7
        self.CMeIE_class=44
        self.class_nums = None

        self.seed = 2024

        self.pool_type = 'avg'
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        if not os.path.exists('saveModel'):
            os.mkdir('saveModel')

        self.semeval_ckpt = 'checkpoint/semeval.pth.tar'
        self.BB_ckpt = 'checkpoint/BB.pth.tar'
        self.Chemport_ckpt = 'checkpoint/Chemport.pth.tar'
        self.SciERC_ckpt = 'checkpoint/SciERC.pth.tar'
        self.CMeIE_ckpt = 'checkpoint/CMeIE.pth.tar'

        self.semeval_pth = 'semeval.pth'
        self.BB_pth = 'BB.pth'
        self.Chemport_pth = 'Chemport.pth'
        self.SciERC_pth = 'SciERC.pth'
        self.CMeIE_pth = 'CMeIE.pth'

    def _get_path(self, root_path):
        self.root_path = root_path
        # bert base uncase bert\bert-base-uncased

        self.bert_base_cased = os.path.join(root_path, 'bert/bert_base_cased')
        self.scibert_scivocab_cased = os.path.join(root_path, 'bert/scibert_scivocab_cased')
        self.bert_base_chinese = os.path.join(root_path, 'bert/bert_base_chinese')

        # BB-triple
        self.BB_rel2id = os.path.join(root_path, 'data/BB/rel2id.json')
        self.BB_train = os.path.join(root_path, 'data/BB/BB_train.json')
        self.BB_val = os.path.join(root_path, 'data/BB/BB_dev.json')
        self.BB_test = os.path.join(root_path, 'data/BB/new_test_epo.json')

        # Chemport-triple
        self.Chemport_rel2id = os.path.join(root_path, 'data/ChemPort/rel2id.json')
        self.Chemport_train = os.path.join(root_path, 'data/ChemPort/ChemPort_train.json')
        self.Chemport_val = os.path.join(root_path, 'data/ChemPort/ChemPort_dev.json')
        self.Chemport_test = os.path.join(root_path, 'data/ChemPort/new_train_epo.json')

        # SciERC-triple
        self.SciERC_rel2id = os.path.join(root_path, 'data/SciERC/rel2id.json')
        self.SciERC_train = os.path.join(root_path, 'data/SciERC/SciERC_train.json')
        self.SciERC_val = os.path.join(root_path, 'data/SciERC/SciERC_dev.json')
        self.SciERC_test = os.path.join(root_path, 'data/SciERC/new_test_seo.json')

        # CMeIE-triple
        self.CMeIE_rel2id = os.path.join(root_path, 'data/CMeIE/rel2id.json')
        self.CMeIE_train = os.path.join(root_path, 'data/CMeIE/CMeIE_train.json')
        self.CMeIE_val = os.path.join(root_path, 'data/CMeIE/CMeIE_dev.json')
        self.CMeIE_test = os.path.join(root_path, 'data/CMeIE/CMeIE_test.json')
