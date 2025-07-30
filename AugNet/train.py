#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
import torch
import torch.nn as nn
from tools.funcs import *
from PIL import Image
import os
import numpy as np
# from steganalyzer import SRNet,LWENet,CovNet,YedNet
import cv2

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1


def train(device, train_loader, epoch, AugNet, sampler, optimizer_Aug, Amp, num):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  end = time.time()
  ADV1 = ADVLoss('SRNet', 'steganalyzer/SRNet-BB-suni04/model_best.pth')
  ADV2 = ADVLoss('CovNet', 'steganalyzer/CovNet-BB-suni04/model_best.pth')
  ADV3 = ADVLoss('LWENet', 'steganalyzer/LWENet-BB-suni04/model_best.pth')
  for i, sample in enumerate(train_loader):
    data_time.update(time.time() - end)
    data, label = sample['data'], sample['label']
    shape = list(data.size())
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    label = label.reshape(-1)
    data, label = data.to(device), label.to(device)

    # update the augmentation network
    optimizer_Aug.zero_grad()
    P = AugNet(data)
    noise = sampler(P * 0.5, P * 0.5)  # sampling
    Augdata = data + Amp * noise
    lossnum = (torch.sum(abs(noise), dim=[1, 2, 3]).mean() - num) ** 2
    # print("data.shape:",data.shape)
    # print("Augdata.shape:", Augdata.shape)
    ADVL1 = ADV1(Augdata, label)
    ADVL2 = ADV2(Augdata, label)
    ADVL3 = ADV3(Augdata, label)

    #stego扣掉对抗噪声仍然等于stego
    label_1 = torch.ones(label.shape).long().to(device)
    Augdata1 = data - Amp * noise
    ADVL11 = ADV1(Augdata1, label_1)
    ADVL22 = ADV2(Augdata1, label_1)
    ADVL33 = ADV3(Augdata1, label_1)

    loss = 1 * lossnum + 2 * (ADVL1 + ADVL2 + ADVL3) + 1 * (ADVL11 + ADVL22 + ADVL33)
    losses.update(loss.item(), data.size(0))
    loss.backward()
    optimizer_Aug.step()

    batch_time.update(time.time() - end)
    end = time.time()

    if (i+1) % TRAIN_PRINT_FREQUENCY == 0:
      logging.info('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        data_time=data_time, loss=losses))


def evaluate(device, eval_loader, epoch, AugNet, sampler, optimizer_Aug, Amp, num, best_loss, PARAMS_PATH):
  AugNet.eval()
  losses = AverageMeter()
  ADV1 = ADVLoss('SRNet', 'steganalyzer/SRNet-BB-suni04/model_best.pth')
  ADV2 = ADVLoss('CovNet', 'steganalyzer/CovNet-BB-suni04/model_best.pth')
  ADV3 = ADVLoss('LWENet', 'steganalyzer/LWENet-BB-suni04/model_best.pth')
  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']
      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)
      P = AugNet(data)
      noise = sampler(P * 0.5, P * 0.5)  # sampling
      Augdata = data + Amp * noise
      lossnum = (torch.sum(abs(noise), dim=[1, 2, 3]).mean() - num) ** 2
      ADVL1 = ADV1(Augdata, label)
      ADVL2 = ADV2(Augdata, label)
      ADVL3 = ADV3(Augdata, label)

      # stego扣掉对抗噪声仍然等于stego
      label_1 = torch.ones(label.shape).long().to(device)
      Augdata1 = data - Amp * noise
      ADVL11 = ADV1(Augdata1, label_1)
      ADVL22 = ADV2(Augdata1, label_1)
      ADVL33 = ADV3(Augdata1, label_1)

      loss = 1 * lossnum + 2 * (ADVL1 + ADVL2 + ADVL3) + 1 * (ADVL11 + ADVL22 + ADVL33)
      losses.update(loss.item(), data.size(0))

  if losses.avg < best_loss and epoch > 1:
    best_loss = losses.avg
    all_state = {
      # 'original_state': model.state_dict(),
      'AugNet_state': AugNet.state_dict(),
      'optimizer_state': optimizer_Aug.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)

  logging.info('-' * 8)
  logging.info('val loss:{:.4f}'.format(losses.avg))
  logging.info('Best loss:{:.4f}'.format(best_loss))
  logging.info('-' * 8)
  return best_loss

def Test(device, eval_loader, AugNet, sampler, Amp, num, output_dir):
  AugNet.eval()
  with torch.no_grad():
    for sample in eval_loader:
      data, label, name = sample['data'], sample['label'],sample['name']
      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)
      # print(label.shape)
      data, label = data.to(device), label.to(device)
      P = AugNet(data)
      noise = sampler(P * 0.5, P * 0.5)  # sampling
      # noise = sampler(P)
      noise = Amp * noise
      Augdata = torch.clamp(data + noise, min=0, max=255)
      output_image = Augdata.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # 形状 (1, height, width)
      print('name:',name)
      img_path = os.path.join(output_dir, name[0])
      Image.fromarray(output_image).save(img_path)

  # logging.info('-' * 8)
  # # logging.info('Loss:{:.4f}'.format(losses.avg))
  # logging.info('-' * 8)
  return 0

def Test_sub(device, eval_loader, AugNet, sampler, Amp, num, output_dir):
  AugNet.eval()
  with torch.no_grad():
    for sample in eval_loader:
      data, label, name = sample['data'], sample['label'],sample['name']
      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)
      # print(label.shape)
      data, label = data.to(device), label.to(device)
      P = AugNet(data)
      noise = sampler(P * 0.5, P * 0.5)  # sampling
      # noise = sampler(P)
      noise = Amp * noise
      Augdata = torch.clamp(data - noise, min=0, max=255)
      output_image = Augdata.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # 形状 (1, height, width)
      print('name:',name)
      img_path = os.path.join(output_dir, name[0])
      Image.fromarray(output_image).save(img_path)

