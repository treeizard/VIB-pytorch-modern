import torch
import math
from pathlib import Path

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter

from utils import cuda, Weight_EMA_Update
from data_process.datasets import return_data
from model import ToyNet


class Solver(object):

    def __init__(self, args):
        self.args = args

        # Device setup
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        # Training Parameters
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.eps = 1e-9
        self.K = args.K
        self.beta = args.beta
        self.num_avg = args.num_avg
        self.global_iter = 0
        self.global_epoch = 0

        # Network & Optimizer
        self.toynet = ToyNet(self.K).to(self.device)
        self.toynet_ema = Weight_EMA_Update(
            ToyNet(self.K).to(self.device),
            self.toynet.state_dict(),
            decay=0.999
        )

        # Checkpoint directory
        self.ckpt_dir = Path(args.ckpt_dir) / args.env_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Optimizers:
        self.optim = optim.Adam(self.toynet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)

        # Load checkpoint if specified
        if args.load_ckpt:
            self.load_checkpoint(args.load_ckpt)

        # Logging and history
        self.history = {
            'avg_acc': 0.0,
            'info_loss': 0.0,
            'class_loss': 0.0,
            'total_loss': 0.0,
            'epoch': 0,
            'iter': 0
        }

        # TensorBoard setup
        self.tensorboard = args.tensorboard
        if self.tensorboard:
            self.summary_dir = Path(args.summary_dir) / args.env_name
            self.summary_dir.mkdir(parents=True, exist_ok=True)
            self.tf = SummaryWriter(log_dir=self.summary_dir)
            self.tf.add_text(tag='argument', text_string=str(args), global_step=self.global_epoch)

        # Data loader
        self.data_loader = return_data(args)

    def set_mode(self, mode='train'):
        mode = mode.lower()
        if mode == 'train':
            self.toynet.train()
            self.toynet_ema.model.train()
        elif mode == 'eval':
            self.toynet.eval()
            self.toynet_ema.model.eval()
        else:
            raise ValueError("Mode error. It should be either 'train' or 'eval'")
        
    def info_loss_fun(self, mu, std):
        info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))
        return info_loss
    
    def train(self):
        self.set_mode('train')
        for e in range(self.epoch):
            self.global_epoch += 1

            for idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = images.to(self.device)
                y = labels.to(self.device)

                (mu, std), logit = self.toynet(x)

                class_loss = F.cross_entropy(logit, y) / math.log(2)
                info_loss = self.info_loss_fun(mu, std)
                total_loss = class_loss + self.beta * info_loss

                izy_bound = math.log(10, 2) - class_loss.item()
                izx_bound = info_loss.item()

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                self.toynet_ema.update(self.toynet.state_dict())

                prediction = logit.argmax(dim=1)
                accuracy = (prediction == y).float().mean()

                if self.num_avg != 0:
                    _, avg_soft_logit = self.toynet(x, self.num_avg)
                    avg_prediction = avg_soft_logit.argmax(dim=1)
                    avg_accuracy = (avg_prediction == y).float().mean()
                else:
                    avg_accuracy = torch.zeros_like(accuracy, device=self.device)

                # Console logging
                if self.global_iter % 100 == 0:
                    print(f"i:{idx + 1} IZY:{izy_bound:.2f} IZX:{izx_bound:.2f} "
                        f"acc:{accuracy.item():.4f} avg_acc:{avg_accuracy.item():.4f} "
                        f"err:{1 - accuracy.item():.4f} avg_err:{1 - avg_accuracy.item():.4f}")

                # TensorBoard logging
                if self.tensorboard and self.global_iter % 10 == 0:
                    self.tf.add_scalars('performance/accuracy', {
                        'train_one-shot': accuracy.item(),
                        'train_multi-shot': avg_accuracy.item()
                    }, self.global_iter)

                    self.tf.add_scalars('performance/error', {
                        'train_one-shot': 1 - accuracy.item(),
                        'train_multi-shot': 1 - avg_accuracy.item()
                    }, self.global_iter)

                    self.tf.add_scalars('performance/cost', {
                        'train_one-shot_class': class_loss.item(),
                        'train_one-shot_info': info_loss.item(),
                        'train_one-shot_total': total_loss.item()
                    }, self.global_iter)

                    self.tf.add_scalars('mutual_information/train', {
                        'I(Z;Y)': izy_bound,
                        'I(Z;X)': izx_bound
                    }, self.global_iter)

            if self.global_epoch % 2 == 0:
                self.scheduler.step()

            self.test()

        print(" [*] Training Finished!")

    def test(self, save_ckpt=True):
        self.set_mode('eval')

        class_loss = 0.0
        info_loss = 0.0
        total_loss = 0.0
        izy_bound = 0.0
        izx_bound = 0.0
        correct = 0.0
        avg_correct = 0.0
        total_num = 0

        with torch.no_grad():
            for _, (images, labels) in enumerate(self.data_loader['test']):
                x = images.to(self.device)
                y = labels.to(self.device)

                (mu, std), logit = self.toynet_ema.model(x)

                # Cross-entropy loss (sum over batch)
                batch_class_loss = F.cross_entropy(logit, y, reduction='sum') / math.log(2)
                batch_info_loss = (-0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2))).sum() / math.log(2)
                batch_total_loss = batch_class_loss + self.beta * batch_info_loss

                class_loss += batch_class_loss.item()
                info_loss += batch_info_loss.item()
                total_loss += batch_total_loss.item()
                total_num += y.size(0)

                izy_bound += (math.log(10, 2) - batch_class_loss.item()) * y.size(0)
                izx_bound += batch_info_loss.item()

                prediction = logit.argmax(dim=1)
                correct += (prediction == y).float().sum().item()

                if self.num_avg != 0:
                    _, avg_soft_logit = self.toynet_ema.model(x, self.num_avg)
                    avg_prediction = avg_soft_logit.argmax(dim=1)
                    avg_correct += (avg_prediction == y).float().sum().item()
                else:
                    avg_correct = 0.0

        # Normalize by total number of samples
        accuracy = correct / total_num
        avg_accuracy = avg_correct / total_num if self.num_avg != 0 else 0.0
        class_loss /= total_num
        info_loss /= total_num
        total_loss /= total_num
        izy_bound /= total_num
        izx_bound /= total_num

        print('[TEST RESULT]')
        print(f'e:{self.global_epoch} IZY:{izy_bound:.2f} IZX:{izx_bound:.2f} '
            f'acc:{accuracy:.4f} avg_acc:{avg_accuracy:.4f} '
            f'err:{1 - accuracy:.4f} avg_err:{1 - avg_accuracy:.4f}\n')

        # Save checkpoint if best accuracy improved
        if self.history['avg_acc'] < avg_accuracy:
            self.history.update({
                'avg_acc': avg_accuracy,
                'class_loss': class_loss,
                'info_loss': info_loss,
                'total_loss': total_loss,
                'epoch': self.global_epoch,
                'iter': self.global_iter
            })
            if save_ckpt:
                self.save_checkpoint('best_acc.tar')

        # TensorBoard logging
        if self.tensorboard:
            self.tf.add_scalars('performance/accuracy', {
                'test_one-shot': accuracy,
                'test_multi-shot': avg_accuracy
            }, self.global_iter)

            self.tf.add_scalars('performance/error', {
                'test_one-shot': 1 - accuracy,
                'test_multi-shot': 1 - avg_accuracy
            }, self.global_iter)

            self.tf.add_scalars('performance/cost', {
                'test_one-shot_class': class_loss,
                'test_one-shot_info': info_loss,
                'test_one-shot_total': total_loss
            }, self.global_iter)

            self.tf.add_scalars('mutual_information/test', {
                'I(Z;Y)': izy_bound,
                'I(Z;X)': izx_bound
            }, self.global_iter)

        self.set_mode('train')


    def save_checkpoint(self, filename='best_acc.tar'):
        model_states = {
                'net':self.toynet.state_dict(),
                'net_ema':self.toynet_ema.model.state_dict(),
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.global_epoch,
                'history':self.history,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.toynet.load_state_dict(checkpoint['model_states']['net'])
            self.toynet_ema.model.load_state_dict(checkpoint['model_states']['net_ema'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))
