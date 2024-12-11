import datetime
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_dataset(image_set, transform, args):
    from data.dataset_ import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes

def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)

def evaluate(model, data_loader, device, demo = False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    cnt = 1
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, sent_list, img_ndarray= data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            

            original_h, original_w = img_ndarray.size(1), img_ndarray.size(2)
            
            
            GT = F.interpolate(target.unsqueeze(0).float(), (original_h, original_w))
            GT = GT.squeeze()
            GT = GT.cpu().data.numpy()
            GT = GT.astype(np.int8)
            target = target.cpu().data.numpy()
            img_ndarray = img_ndarray.squeeze()
            img_ndarray = img_ndarray.cpu().data.numpy()

            if demo:
                # original image visualization
                visualization0 = Image.fromarray(img_ndarray)
                # GT image visualization
                visualization1 = overlay_davis(img_ndarray, GT)
                visualization1 = Image.fromarray(visualization1)
            
            for j in range(sentences.size(-1)):
                output = model(image, sentences[:,:,j], attentions[:,:,j], training=False)
               
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1
                sen = sent_list[j][0].replace('/','')
                
                
                if demo:
                    # Visualization of the segmented results
                    result = output.argmax(1, keepdim=True)
                    result = F.interpolate(result.float(), (original_h, original_w))
                    result = result.squeeze() 
                    result = result.cpu().data.numpy()
                    result = result.astype(np.int8)
                
                    visualization = overlay_davis(img_ndarray, result)
                    visualization = Image.fromarray(visualization)
                    visualization.save(os.path.join('/database/refcoco/', f'{cnt}_pred_{sen}.jpg'))
                    visualization0.save(os.path.join('/database/refcoco/', f'{cnt}_ori_{sen}.jpg'))
                    visualization1.save(os.path.join('/database/refcoco/', f'{cnt}_GT_{sen}.jpg'))

                cnt+=1


    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    checkpoint = torch.load(args.resume, map_location='cuda')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    evaluate(model, data_loader_test, device=device, demo=False)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
