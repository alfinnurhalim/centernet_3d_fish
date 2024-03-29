from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import math

from models.losses import FocalLoss, L1Loss, BinRotLoss,MSELoss,ReIdLoss
from models.utils import _sigmoid
from utils.debugger import Debugger
from .base_trainer import BaseTrainer
from models.utils import _transpose_and_gather_feat

class FishLoss(torch.nn.Module):
  def __init__(self, opt):
    super(FishLoss, self).__init__()
    self.crit = torch.nn.MSELoss()
    self.crit_reg = L1Loss()
    self.crit_reg_mse = MSELoss()
    self.crit_class= ReIdLoss()

    self.classifier = torch.nn.Linear(opt.reid, opt.max_id)

    self.opt = opt
  
  def forward(self, outputs, batch):
    opt = self.opt

    loss = 0

    output = outputs[0]
    output['hm'] = _sigmoid(output['hm'])
    output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.

    hm_loss = self.crit(output['hm'], batch['hm'])
    off_loss = self.crit_reg(output['reg'], batch['reg_mask'],
                            batch['ind'], batch['reg'])

    dep_loss = self.crit_reg(output['dep'], batch['reg_mask'],
                              batch['ind'], batch['dep'])
    dim_loss = self.crit_reg(output['dim'], batch['reg_mask'],
                              batch['ind'], batch['dim'])
    rot_loss = self.crit_reg(output['rot'], batch['reg_mask'],
                              batch['ind'], batch['rot'])

    # wh_loss = self.crit_reg(output['wh'], batch['reg_mask'],
    #                           batch['ind'], batch['wh']) 

    # reid_loss = self.crit_class(output['reid'], batch['reg_mask'],
    #                           batch['ind'], batch['reid'],self.classifier) 
    
    rot_loss = rot_loss * 0.01
    # wh_loss = wh_loss * 0.1

    loss = loss + hm_loss
    loss = loss + off_loss

    loss = loss + dep_loss
    loss = loss + dim_loss
    loss = loss + rot_loss

    # loss = loss + wh_loss 

    # loss = loss + reid_loss

    loss_stats = {'loss': loss,
                  'hm_loss': hm_loss,
                  'off_loss': off_loss,
                  'dep_loss': dep_loss, 
                  'dim_loss': dim_loss,
                  'rot_loss': rot_loss,
                  # 'wh_loss' : wh_loss,
                  # 'reid_loss' : reid_loss
                 }

    return loss, loss_stats

class FishTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(FishTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 
                  'hm_loss', 
                  'off_loss',
                  'dep_loss',
                  'dim_loss',
                  'rot_loss',
                  # 'wh_loss',
                  # 'reid_loss'
                  ]

    loss = FishLoss(opt)
    return loss_states, loss

  