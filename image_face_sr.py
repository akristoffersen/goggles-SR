import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt
import torch
from PIL import Image
from dlib_alignment import dlib_detect_face, face_recover
import torchvision.transforms as transforms
from models.SRGAN_model import SRGANModel
import argparse
import utils

def get_FaceSR_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=64)

    # data dir
    parser.add_argument('--pretrain_model_G', type=str, default='90000_G.pth')
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    args = parser.parse_args("")

    return args

border = 0.25

def single_image_face_sr(img_path, image=None):
    
    # getting faces:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if img_path != None:
        image = cv2.imread(img_path)
        print(img_path)
    dims = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print("Faces:", len(faces))
    face_crops = []
    
    for x, y, w, h in faces:
        offsets = (int(w * border), int(h * border)) # w, h
        
        point = (max(0, x - offsets[0]), max(0, y - offsets[1]))
        
        face_crop_bgr = image[point[1]: point[1] + h + 2 * offsets[1], \
                              point[0]: point[0] + w + 2 * offsets[0]]
        face_crop = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        print(face_crop.shape)
        face_crops.append((point, face_crop))
    # doing super res
    _transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])
    
    sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
    sr_model.load()
    
    def sr_forward(img, padding=0.5, moving=0.1):
        # img_aligned, M = dlib_detect_face(img, padding=padding, image_size=img.shape[:-1], moving=moving)
        input_img = torch.unsqueeze(_transform(Image.fromarray(img)), 0)
        sr_model.var_L = input_img.to(sr_model.device)
        sr_model.test()
        output_img = sr_model.fake_H.squeeze(0).cpu().numpy()
        output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        # rec_img = face_recover(output_img, M * 4, img)
        return output_img
    
    # setting up the resized image
    
    new_image_bgr = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
    new_image = cv2.cvtColor(new_image_bgr, cv2.COLOR_BGR2RGB)
    
    for face in face_crops:
        face_image = face[1]
        position = face[0] + (face_image.shape)[:-1]

        output_img = sr_forward(face_image)
        new_dims = output_img.shape[:-1]
        new_image[4 * position[1]: 4 * position[1] + new_dims[1], 4 * position[0] : 4 * position[0] + new_dims[0]] = output_img
    
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    if img_path != None:
        output_path = img_path[:-4] + "_face_sr" + ".png"
        cv2.imwrite(output_path, new_image)
        print("done!")
    return new_image

if __name__ == '__main__':
    single_image_face_sr(sys.argv[1])

