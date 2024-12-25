
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
from save_image_own import *
from torchvision import utils as vutils
# from model.vgg import VGGNetFeats
import torchvision
from loss import color_loss
import torch.nn as nn
from loss import Edge_loss
def get_dark_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])

    return -dc

def lwf_sky(img, J, J_o, w=15):
    dc = get_dark_channel(img, w)
    dc_shaped = dc.repeat(1, 3, 1, 1)
    J_1 = J[dc_shaped > 0.6]
    J_2 = J_o[dc_shaped > 0.6]

    if len(J_1) == 0:
        return 0

    loss = F.smooth_l1_loss(J_1, J_2)

    return loss

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class UncertainLoss(nn.Module):
    def __init__(self,v_num):
        super(UncertainLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma=nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self,*input):
        loss = 0
        for i in range(self.v_num):
            loss+=input[i]/(2*self.sigma[i] **2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss


class Trainer_semi():
    def __init__(self, opt, loader, my_model, ema_model, my_loss, ckp):
        self.opt = opt
        self.epoch = 0
        self.loader_train = loader
        self.loader_test = loader
        self.model = my_model
        self.ema_model = ema_model
        self.loss = torch.nn.L1Loss().cuda()
        self.color_loss = color_loss.L_color().cuda()
        self.edge_loss = Edge_loss.EdgeLoss().cuda()

        self.weighted_loss_func = UncertainLoss(2).cuda()

        self.optimizer = utility.DataLoader_Rain1400(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_flag = opt.dual_flag
        self.consistency = opt.consistency
        self.consistency_rampup = opt.consistency_rampup
        self.labeled_bs = opt.labeled_bs
        self.error_last = 1e8
        self.curiter = 0
        self.max_iters = self.opt.epochs * len(self.loader_train) // self.opt.nAveGrad
        self.testpath = self.opt.testpath
        self.dir = './experiment/' + self.opt.save

        self.log_train = os.path.join(self.dir+'/'+self.opt.model_sava_dirName,'train.txt')
        self.log_test = os.path.join(self.dir+'/'+self.opt.model_sava_dirName, 'test.txt')

        self.perceptual_loss = nn.L1Loss()

    def train(self, epoch):
        aveGrad = 0
        self.epoch = epoch
        self.model.train()
        self.ema_model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        sup_loss_dehaze_record = AvgMeter()
        sup_loss_per_record, con_loss_dehaze_record, loss_ema_perceptual_record = AvgMeter(), AvgMeter(), AvgMeter()
        sup_loss_t_record, con_loss_t_record = AvgMeter(),AvgMeter()
        loss_color_record, loss_ema_color_record = AvgMeter(),AvgMeter()
        CLAHE_loss_record,CLAHE_ema_loss_record = AvgMeter(),AvgMeter()
        loss_nongdu_record = AvgMeter()
        loss_edge_record = AvgMeter()
        uncertainLoss_record = AvgMeter()


        for batch, gt in enumerate(self.loader_train):
            input, gt, trans, ato = gt  # zhelide gt列表 [haze,gt,trans,ato]
            input, gt, trans, ato = self.prepare([input, gt, trans, ato])#####

            timer_data.hold()
            timer_model.tic()

            self.optimizer.param_groups[0]['lr'] = self.opt.lr * (
                         1 - (self.epoch / self.opt.epochs)) ** self.opt.power
            self.optimizer.zero_grad()
            if not self.dual_flag:
                noise = torch.clamp(torch.randn_like(input[self.labeled_bs:]) * 0.1, -0.1, 0.1)
                ema_input = input[self.labeled_bs:]
                ema_input = ema_input.to('cuda')
                predict1, predict2 = self.model(input)
                with torch.no_grad():

                    ema_predict1,ema_predict2 = self.ema_model(ema_input)
                loss_dehaze = self.loss(predict1[0:self.labeled_bs], gt[0:self.labeled_bs])\
                              + self.loss(predict2[0:self.labeled_bs], gt[0:self.labeled_bs])

                loss_color = torch.mean(self.color_loss(predict1)) + torch.mean(self.color_loss(predict2))

                loss_edge = self.edge_loss(predict1[0:self.labeled_bs],gt[0:self.labeled_bs]) +self.edge_loss(predict2[0:self.labeled_bs],gt[0:self.labeled_bs])



                sup_loss = loss_dehaze  + 0.3*loss_color + 0.2*loss_edge  # 有监督损失

                con_loss_dehaze = self.loss(predict1[self.labeled_bs:], ema_predict1) +\
                                  self.loss(predict2[self.labeled_bs:], ema_predict2)



                loss_ema_color =  torch.mean(self.color_loss(ema_predict1)+self.color_loss(ema_predict2))


                con_loss = con_loss_dehaze
                consistency_weight = self.get_current_consistency_weight(epoch)
                loss = 0.8 * loss_dehaze + 0.3 * loss_color + 0.65 * loss_edge \
                       + consistency_weight * con_loss + 0.3 * loss_ema_color


            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()
                aveGrad += 1
                sup_loss_dehaze_record.update(loss_dehaze.data, self.opt.batchSize)
                con_loss_dehaze_record.update(con_loss_dehaze.data, self.opt.batchSize)
                loss_color_record.update(loss_color.data,self.opt.batchSize)
                loss_ema_color_record.update(loss_ema_color.data,self.opt.batchSize)
                loss_edge_record.update(loss_edge.data,self.opt.batchSize)

            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            if aveGrad % (self.opt.nAveGrad / self.opt.batchSize) == 0:
                self.optimizer.step()
                self.update_ema_variables(self.model, self.ema_model, self.opt.ema_decay, self.curiter)
                self.curiter = self.curiter + 1
                aveGrad = 0


            timer_model.hold()
            if (batch + 1) % self.opt.print_every == 0:
                log = 'Trainer_semi \n [epoch %d],[uncertainLoss %.5f],[sup_loss_dehaze %.5f],[sup_loss_per %.5f]' \
                      '[loss_dehaze_con %.5f], [loss_ema_perceptual %.5f],[lr %.13f]' \
                      '[loss_color %.5f],[loss_edge  %.5f] '% \
                      (self.epoch,uncertainLoss_record.avg, sup_loss_dehaze_record.avg, sup_loss_per_record.avg,
                       con_loss_dehaze_record.avg,loss_ema_perceptual_record.avg,self.optimizer.param_groups[0]['lr'],
                       loss_color_record.avg,loss_edge_record.avg)
                print(log)

            timer_data.tic()


        if epoch > 100 :
            snapshot_name = 'epoch%d' % (epoch)
            torch.save(self.model.state_dict(), os.path.join(self.dir, self.opt.model_sava_dirName, snapshot_name + '.pt'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.dir, self.opt.model_sava_dirName, snapshot_name + '_optim.pt'))

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

                input,gt,trans = gt
                no_eval = (gt.nelement() == 1)
                if not no_eval:
                    haze, gt, trans = self.prepare([input, gt, trans])
                else:
                    haze = self.prepare(input)

                predict1, predict2= self.model(haze)

                dehaze1 = predict1
                dehaze1 = quantize(dehaze1)
                dehaze2 = predict2
                dehaze2 = quantize(dehaze2)
                gt = 255 * gt
                dehaze1 = 255 * dehaze1
                dehaze2 = 255 * dehaze2
                input = 255*input

                avgSSIM1 += SSIM(dehaze1, gt)
                avgPSNR1 += PSNR(dehaze1, gt)

                avgSSIM2 += SSIM(dehaze2, gt)
                avgPSNR2 += PSNR(dehaze2, gt)

                i += 1

                filename = "experiment/semi_transformer/results_indoor"
                gt_image_name = str(i) + '_gt' + '.jpg'
                save_mine(gt,filename,gt_image_name)

                dehaze_image_name = str(i) + '_dehaze' + '.jpg'
                save_mine(dehaze1,filename,dehaze_image_name)

                haze_image_name = str(i) + '_haze' + '.jpg'
                save_mine(input, filename, haze_image_name)

            log = '[epoch:%d]\t[PSNR1: %.4f]\t[SSIM1: %.4f]\n' \
                  % (count, avgPSNR1 / len(self.loader_test), avgSSIM1 / len(self.loader_test),)
            print(log)
            open(self.log_test, 'a').write(self.testpath + '\n')
            open(self.log_test, 'a').write(log + '\n')

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

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


def save_mine(img_tensor,img_filename,img_name):
    img = img_tensor.data.cpu().numpy()
    img = np.squeeze(img, 0)
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 0, 2)
    cv2.imwrite(os.path.join(img_filename, img_name), img)