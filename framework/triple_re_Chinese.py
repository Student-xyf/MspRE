import os
import torch
from torch import nn, optim
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .dataloaders_CMeIE import RELoader, REDataset
from configs import Config

class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class Triple_RE_CHINESE(nn.Module):

    def __init__(self,
                 model,
                 train,
                 val,
                 test,
                 rel2id,
                 pretrain_path,
                 ckpt,
                 batch_size=16,
                 num_workers=6,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        self.train_loder = RELoader(
            path=train,
            rel2id=rel2id,
            pretrain_path=pretrain_path,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.val_set = REDataset(path=val, rel_dict_path=rel2id, pretrain_path=pretrain_path)
        self.test_set = REDataset(path=test, rel_dict_path=rel2id, pretrain_path=pretrain_path)
        # Model
        # self.model = nn.DataParallel(model)
        self.model = model
        # Criterion
        self.loss_func = nn.BCELoss()
        # Params and optimizer
        params = self.model.parameters()
        self.lr = lr
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, params), lr, weight_decay=weight_decay)
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt
    def train_model(self, dataset_name, warmup=True):
        config = Config()
        best_f1 = 0
        best_epoch = 0  # 新增变量来记录最高 f1 值的 epoch
        global_step = 0
        wait =0

        # 获取当前时间的字符串
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        writer = SummaryWriter(f'logs/{dataset_name}_{current_time}')  # 初始化TensorBoard

        for epoch in range(self.max_epoch):
            # Train
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_sent_loss = AverageMeter()
            t = tqdm(self.train_loder)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                # sentence
                loss = self.model(*data)
                # Log
                avg_sent_loss.update(loss.item(), 1)
                t.set_postfix(sent_loss=avg_sent_loss.avg)

                # Optimize
                if warmup == True:
                    warmup_step = 300
                    if global_step < warmup_step:
                        warmup_rate = float(global_step) / warmup_step
                    else:
                        warmup_rate = 1.0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * warmup_rate

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                ###

            # 记录每个 epoch 的平均损失值
            writer.add_scalar('Loss/train_epoch', avg_sent_loss.avg, epoch)

                # Val
            print("=== Epoch %d val ===" % epoch)
            self.eval()
            precision, recall, f1 = self.val_set.metric(self.model)
            if f1 > best_f1 or f1 < 1e-4:
                if f1 > 1e-4:
                    best_f1 = f1
                    best_epoch = epoch  # 更新记录最高 f1 值的 epoch
                    print("Best ckpt and saved at Epoch %d." % best_epoch)
                    torch.save({'state_dict': self.model.state_dict()}, self.ckpt)

                    model_path = os.path.join('./saveModel', config.Chemport_pth)
                    torch.save(self.model, model_path)
            else:
                wait +=1
                if wait>150:
                    print('Epoch %05d: early stopping' % (epoch + 1))
                    break
            print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f, best_epoch: %d\n' % (f1, precision, recall, best_f1, best_epoch))
        #torch.save({'state_dict': self.model.state_dict()}, 'checkpoint/final.pth.tar')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f, best_epoch: %d\n' % (f1, precision, recall, best_f1, best_epoch))
        writer.close()  # 记得关闭writer

    def load_state_dict(self, ckpt):
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])