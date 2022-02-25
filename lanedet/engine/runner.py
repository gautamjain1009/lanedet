import time
import torch
from torch.autograd import grad 
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2

from lanedet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from lanedet.datasets import build_dataloader
from lanedet.utils.recorder import build_recorder
from lanedet.utils.net_utils import save_model, load_network
from mmcv.parallel import MMDataParallel 

from lanedet.models.nets.detector import *
from lanedet.models.heads.lane_seg import *

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        
        # for name, module in self.net.named_modules():
        #     print(name)
        # print(self.net.Detector)
        self.net.to(torch.device('cuda'))
        # self.net = torch.nn.parallel.DataParallel(
        #         self.net, device_ids = range(self.cfg.gpus)).cuda()
        
        # self.net = MMDataParallel(
        #         self.net, device_ids = range(self.cfg.gpus)).cuda()

        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.warmup_scheduler = None
        # TODO(zhengtu): remove this hard code
        if self.cfg.optimizer.type == 'SGD':
            self.warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=5000)
        self.metric = 0.
        self.val_loader = None

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(k, torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1

            # data = self.to_cuda(data)
            
            for k in data:
                data[k] = data[k].cuda()
            # # output = self.net(data)
            
            self.optimizer.zero_grad()
            
            ###################################### 
            all_f = self.net.featurizer(data['img']) ### list of Resnet layers out
            # print(all_f[0].shape)
            
            output = self.net.classifier(all_f, data) # (batch_size,classes,h,W)
            all_p = output['outs']
            # print(all_p['seg'].shape)

            # # print(output.keys())
            # # print(data['img'].shape) # --- [8,3,368,640]
                    
            #TODO: Implement the RSC algorithm and check it enhances the accuracy. 
            """
            RSC algorithm: https://github.com/facebookresearch/DomainBed/blob/25f173caa689f20828629b2e42f90193f203fdfa/domainbed/algorithms.py#L866
            """
            
            all_p = torch.max(all_p['seg'], 1)[0]
            print("===> predictions shape", all_p.shape)
            
            print(data['mask'].shape)
            def muted_gradients(all_p, data, all_f):
                # Equation (1): compute gradients with respect to representation            
                all_g = grad((all_p * data['mask']).sum(), all_f,retain_graph=True) ## this loss can be manually calcualted also using outs *
                all_g_0 = all_g[0]
                print("===>Gradient_shape", all_g_0.shape)

                # Equation (2): compute top-gradient-percentile mask
                #TODO: set the hyperparam for muting the gradients drop_f
                
                percentiles = np.percentile(all_g_0.cpu(), 33, axis =0) ## TODO:confirm does axis =0 means percentile taken along batches 
                percentiles = torch.Tensor(percentiles)
                print(percentiles.shape)
                percentiles = percentiles.unsqueeze(0).repeat(8,1,1,1)

                maskf = all_g_0.lt(percentiles.cuda()).float()
                print("** printing masks shapes")
                print(maskf.shape)
                
                # Equation (3): mute top-gradient-percentile activations
                # all_f_muted = [maskf * bbone_f for bbone_f in bbone_features]
                all_f_muted = maskf * all_f
                
                return all_f_muted

            all_f_muted = [muted_gradients(all_p, data, layerwise_features) for layerwise_features in all_f]
          
            # Equation (4): compute muted predictions
            muted_output = self.net.classifier(all_f_muted, data)
            
            loss = muted_output['loss']
            ##############################################
            loss.backward()
            
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            if self.warmup_scheduler:
                self.warmup_scheduler.dampen()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(muted_output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        self.recorder.logger.info('Start training...')
        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.net.eval()
        predictions = []
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            for k in data:
                data[k] = data[k].cuda()
            
            with torch.no_grad():
                # output = self.net(data)
                # output = self.net.module.get_lanes(output)
                all_f = self.net.featurizer(data['img']) ### list of Resnet layers out
                
                output = self.net.classifier(all_f, data) # (batch_size,classes,h,W)               
                

                output = self.net.get_lanes(output)
                predictions.extend(output)
            
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        out = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        self.recorder.logger.info(out)
        metric = out
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(is_best=True)
        self.recorder.logger.info('Best metric: ' + str(self.metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler,
                self.recorder, is_best)
