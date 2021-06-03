import torch
import numpy as np
import pandas as pd
import cv2
import utility
from decimal import Decimal
from tqdm import tqdm
from option import args
import os
from torchvision import transforms 
from torchvision.utils import save_image 
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import math
import imageio

class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.float_scale = opt.float_scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        int_scale = max(self.scale)
        float_scale = self.float_scale
        scale = int_scale + float_scale
        res_scale = scale / int_scale 
        self.ckp.set_epoch(epoch)


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        for name, param in self.model.named_parameters():
            splitname = name.split('.')
            if splitname[1] != 'SRCNN':
                param.requires_grad = False


        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])

            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[(i-1) - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            ##여기서 sr사이즈 hr사이즈
            loss_primary = self.loss(sr[-1], hr)
            loss_primary += self.loss(sr[0], lr[0])
            
            # compute dual loss
            # loss_dual = self.loss(sr2lr[0], lr[0])
            # for i in range(1, len(self.scale)):
            #     loss_dual += self.loss(sr2lr[i], lr[i])

            # compute average loss
            # average_feat =(sr[-1]+fflip_sr[-1])/2
            # loss_average = self.loss(average_feat, hr)

            

            #copute flip loss
            loss_flip =0
            # for i in range(0, len(sr)):
            #     loss_flip+= self.loss(sr[i], fflip_sr[i])

            # compute total loss
            loss =  loss_primary
            # loss =  loss_primary+ self.opt.dual_weight 
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
    
    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))

        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            int_scale = max(self.scale)
            float_scale = self.float_scale
            scale = int_scale + float_scale
            res_scale = scale / int_scale
            for si, s in enumerate([int_scale]):
                eval_psnr = 0
                eval_simm =0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)


                    N,C,H,W = lr[0].size()
                    outH,outW = int(H*scale),int(W*scale)
                    timer_test.tic()

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]

                    sr= sr.contiguous().view(N, C,outH,outW)
                    sr = utility.quantize(sr, self.opt.rgb_range)

                    timer_test.hold()
               

                    if not no_eval:
                        psnr = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                   
                        # hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                        # sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                        # simm = utility.SSIM(hr_numpy, sr_numpy)
                        # eval_simm += simm

                        eval_psnr +=psnr

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                # eval_simm = eval_simm / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                # print('SIMM:',eval_simm)



        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')
        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
        