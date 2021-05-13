import torch
import imageio
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import importlib
import os

import voc12.dataloader
from misc import imutils


def work(model, databin, args):
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    with torch.no_grad():
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model(img[0]) for img in pack['img']]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})

            if iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def mywork(model, args):
    img_dir = "/home/pfc/code/object_detect/irn/voc12/data/VOC12/infer/JPEGImages"
    img_normal = voc12.dataloader.TorchvisionNormalize()
    with torch.no_grad():
        for img_name in os.listdir(img_dir):
            curimg_path = os.path.join(img_dir, img_name)
            img = imageio.imread(curimg_path)
            size = (img.shape[0], img.shape[1])

            ms_img_list = []
            for s in args.cam_scales:
                if s == 1:
                    s_img = img
                else:
                    s_img = imutils.pil_rescale(img, s, order=3)
                s_img = img_normal(s_img)
                s_img = imutils.HWC_to_CHW(s_img)
                ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
            if len(args.cam_scales) == 1:
                ms_img_list = ms_img_list[0]

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            # img_variable = Variable(img.unsqueeze(0))
            outputs = [model(torch.Tensor(img)) for img in ms_img_list]

            strided_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)
            highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                         mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

            valid_cat = torch.zeros(1).type(torch.uint8)

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_cam.cpu(), "high_res": highres_cam.cpu().numpy()})


def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    model.load_state_dict(torch.load(args.cam_weights_name + '.pth', map_location='cpu'), strict=True)
    model.eval()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.train_list, voc12_root=args.voc12_root, scales=args.cam_scales)
    work(model, dataset, args)
    # mywork(model, args)