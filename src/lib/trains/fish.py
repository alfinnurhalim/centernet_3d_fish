from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, L1Loss, BinRotLoss,MSELoss
from models.utils import _sigmoid
from utils.debugger import Debugger
from .base_trainer import BaseTrainer

class FishLoss(torch.nn.Module):
  def __init__(self, opt):
    super(FishLoss, self).__init__()
    self.crit = torch.nn.MSELoss()
    self.crit_reg = L1Loss()
    self.crit_reg_mse = MSELoss()
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

    loss = loss + hm_loss
    loss = loss + off_loss

    loss = loss + dep_loss
    loss = loss + dim_loss
    # loss = loss + rot_loss

    loss_stats = {'loss': loss,
                  'hm_loss': hm_loss,
                  'off_loss': off_loss,
                  'dep_loss': dep_loss, 
                  'dim_loss': dim_loss,
                  'rot_loss': rot_loss, 
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
                  'rot_loss']

    loss = FishLoss(opt)
    return loss_states, loss

  