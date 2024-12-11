import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import random
from functools import reduce
import operator

from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

def get_dataset(image_set, transform, args, eval_mode=False):
    from data.dataset_ import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=eval_mode
                      )
    num_classes = 2

    return ds, num_classes

 
# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight)


def evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions,_,_ = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            for j in range(sentences.size(-1)):
                output = model(image, sentences[:,:,j], attentions[:,:,j], training=False)

                iou, I, U = IoU(output, target)
                acc_ious += iou
                mean_IoU.append(iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
                seg_total += 1
        iou = acc_ious / total_its

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * mIoU, 100 * cum_I / cum_U
    
def minmax_scale(input):
  min_val = np.min(input)
  max_val=np.max(input)
  out = (input-min_val)/(max_val-min_val)
  return out

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, best_oIoU, cont_loss):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        
        total_its += 1

        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)
        
        H, W = image.shape[-2], image.shape[-1]
        l = sentences.shape[-1]
        image = image.reshape(-1, 3, H, W)
        target = target.reshape(-1, H, W)         
        sentences = sentences.reshape(-1, 1, l)
        attentions = attentions.reshape(-1, 1, l)
        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        
        out, s = model(image, sentences, attentions)
    
        loss1 = 2*criterion(out, target.detach())
        loss2 = 0.4*cont_loss(s, target.detach())
        loss = loss1 + loss2 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        
        metric_logger.update(loss=loss.item(),   lr=optimizer.param_groups[0]["lr"]) #
        torch.cuda.synchronize()
    
    return best_oIoU

    
class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')        
        
    def forward(self, m, target):

        loss = self.loss(m, target.float())
        
        return loss

def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args, eval_mode=True)
    
    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, 
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    print(data_loader.__len__())
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)

    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    best_oIoU = -1
    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        single_model.load_state_dict(checkpoint['model'])
        best_oIoU = checkpoint['best_oIoU']
        del checkpoint


    # parameters to optimize
    backbone_no_decay = list()
    backbone_decay = list()
    net_decay = list()
    for name, m in single_model.backbone.named_parameters():
            if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
                backbone_no_decay.append(m)
            else:
                backbone_decay.append(m)

    for name, m in single_model.net.named_parameters():
        if 'QLformer.embeddings.' in name:
            m.requires_grad = False
        
    for name, m in single_model.net.named_parameters():
        if m.requires_grad:
            net_decay.append(m)
            
    
    for name, m in single_model.text_encoder.named_parameters():
        if '.embeddings.' in name:
            m.requires_grad = False
  
    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0, },
        {'params': backbone_decay},
        {'params': net_decay},
        {"params": [p for p in single_model.text_encoder.parameters() if p.requires_grad]}, 
    ]
    
    # optimizer
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # housekeeping
    start_time = time.time()
    iterations = 0
    
    resume_epoch = -999
    cont_criterion = AlignLoss().cuda()
    torch.cuda.empty_cache()
    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)

        best_oIoU = train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations, best_oIoU, cont_criterion)

        iou, overallIoU = evaluate(model, data_loader_test)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            best_oIoU = overallIoU
            dict_to_save = {'model': single_model.state_dict(), 'best_oIoU': best_oIoU,
                                 'epoch': epoch, 'args': args,
                                }

            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'model_best_refcoco.pth'))
          

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    print("local rank = ",args.local_rank)
    utils.init_distributed_mode(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
