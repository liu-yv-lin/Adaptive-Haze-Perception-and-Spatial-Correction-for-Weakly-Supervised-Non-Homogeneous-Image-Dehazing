
import torch
import random
import utility
import data
import model
import loss
from option import args

import torch
import os
import utility
import numpy as np
import datetime
from tqdm import tqdm
import os
import math
from utility import *
from random import uniform
import torch.nn as nn

# from trainer_semi import *

import os



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
global args
count = 100

args.testpath ='dataset/O-HAZE/scale_16'
test_log_path = 'experiment/all_result_1006/HSTS_result'
test_result = 'experiment/result_test_hsts' # 模型文件路径
# test_log_name = 'HSTS.txt'


args.save = 'test'
args.model = 'semi_dehaze'
args.dataset = 'new_data_val'
torch.manual_seed(args.seed)
args.test_only = True
args.pre_train = '' # 预训练模型保存路径


filename = "experiment/result_hsts"

checkpoint = utility.checkpoint(args)
args.manualSeed = random.randint(1, 10000)

class Test_semi():
    def __init__(self, opt, loader, my_model, ema_model, my_loss, ckp):
        self.opt = opt
        self.epoch = 0
        self.loader_train = loader
        self.loader_test = loader
        self.model = my_model
        self.ema_model = ema_model
        self.loss = torch.nn.L1Loss().cuda()
        self.optimizer = utility.DataLoader_Rain1400(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_flag = opt.dual_flag
        self.consistency = opt.consistency
        self.consistency_rampup = opt.consistency_rampup
        self.labeled_bs = opt.labeled_bs
        self.error_last = 1e8
        self.curiter = 0
        self.max_iters = self.opt.epochs * len(self.loader_train) // self.opt.nAveGrad

        self.perceptual_loss = nn.L1Loss()

    def test(self, count):
        self.model.eval()
        avgPSNR1 = 0.0
        avgSSIM1 = 0.0

        avgPSNR2 = 0.0
        avgSSIM2 = 0.0

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            i = 0
            for idx_img, gt in enumerate(tqdm_test):

                input, gt, trans = gt
                no_eval = (gt.nelement() == 1)
                if not no_eval:
                    haze, gt, trans = self.prepare([input, gt, trans])
                else:
                    haze = self.prepare(input)

                predict1, predict2 = self.model(haze)

                dehaze1 = predict1
                dehaze1 = quantize(dehaze1)
                dehaze2 = predict2
                dehaze2 = quantize(dehaze2)
                gt = 255 * gt
                dehaze1 = 255 * dehaze1
                dehaze2 = 255 * dehaze2
                input = 255 * input

                avgSSIM1 += SSIM(dehaze1, gt)
                avgPSNR1 += PSNR(dehaze1, gt)

                avgSSIM2 += SSIM(dehaze2, gt)
                avgPSNR2 += PSNR(dehaze2, gt)

                i += 1

                dehaze_image_name = str(i) + '_dehaze' + '.jpg'
                save_mine(dehaze1, filename, dehaze_image_name)

                haze_image_name = str(i) + '_haze' + '.jpg'
                save_mine(input, filename, haze_image_name)

            log = '[PSNR1: %.4f]\t[SSIM1: %.4f]\n' \
                  % ( avgPSNR1 / len(self.loader_test), avgSSIM1 / len(self.loader_test),)
            print(log)




    def augmentation(self, input, trans):
        input= input.cpu()
        trans = trans.cpu()
        for i in range(0,input.shape[0]):
            haze = input[i]
            depth = trans[i]
            beta = uniform(0, 0.4)
            maxhazy = depth.max()
            depth = depth / (maxhazy)
            transmission = np.exp(-beta * depth)

            a = 0.2 * uniform(0, 1)
            m = depth.shape[1]
            n = depth.shape[2]
            rep_atmosphere = np.tile(np.full([3, 1, 1], a), [1, m, n])

            haze_image = haze * transmission + torch.Tensor(rep_atmosphere) * (1 - transmission)
            input[i] = haze_image
        return input

    def step(self):
        self.scheduler.step()

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        def _prepare(tensor):
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.epoch
            return epoch >= self.opt.epochs




def save_mine(img_tensor,img_filename,img_name):
    img = img_tensor.data.cpu().numpy()
    img = np.squeeze(img, 0)
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    cv2.imwrite(os.path.join(img_filename, img_name), img)


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def getLoader(datasetName, dataroot, transform,  batchSize=1, workers=1,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):

  if datasetName == 'new_data_train':
    from data.new_data_train import new_data as commonDataset
  if datasetName == 'new_data_val':
    from data.new_data_test import new_data as commonDataset

  dataset = commonDataset(args, dataroot)
  return dataset

if checkpoint.ok:
    # loader = data.Data(args)
    dataset_test = getLoader(args.data_test,
                           args.testpath,
                           args.batchSize,
                           args.workers,
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='val',
                           shuffle=False,
                           seed=args.manualSeed)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=int(args.workers))
    print(dataloader_test)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None

    if  args.model == 'semi_dehaze':
        net = model.Model_Semi(args, checkpoint)
        ema_net = model.Model_Semi(args, checkpoint)
        ema_net = create_emamodel(ema_net)
        T = Test_semi(args, dataloader_test, net, ema_net, loss, checkpoint)

    T.test(count)
    checkpoint.done()


