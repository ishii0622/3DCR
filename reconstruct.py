import torch
import os
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import utils_image as util
from utils import utils_estimate as est
from total_variation_3d import TotalVariation3D
import time
import math
import sys
torch.set_printoptions(threshold=np.inf)

class MultiImgs():
    def __init__(self, imgs):
        self.color  = imgs.size()[0]
        self.layer  = imgs.size()[1]
        self.height = imgs.size()[2]
        self.width  = imgs.size()[3]
        self.stat   = imgs

    def nv(self):   
        total = self.layer*self.height*self.width
        return total

    def adapt(self):
        img = self.stat
        sort = torch.zeros(self.layer, self.height, self.width)
        img = torch.rot90(img, 2, [1, 2])
        for i in range(self.layer):
            sort[i] = img[self.layer-i-1]
        return img


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    target_name = 'phantom1_input'
    apa_size = 5
    resolution = 0.92
    NA = 0.75

    iter_num = 1000

    # hyper parameter
    lr = 6.0e-3
    mu = 1.0e-2

    n_channels = 1

    targets = 'inputs/input'
    results = 'results'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # in_path, out_path
    # ----------------------------------------
    in_path  = os.path.join(targets, target_name)  # path for Low training images

    # ----------------------------------------
    # read multi-focus microscopic images
    # ----------------------------------------
    in_paths = util.get_image_paths(in_path)
    img_list = util.imreads_uint(in_paths, n_channels)
    imgs     = est.imglist2tensor(img_list)          # tensor(C, N, H, W)

    input_imgs = MultiImgs(imgs)
    if input_imgs.color == 1:
        input_imgs.stat = torch.squeeze(input_imgs.stat)
    width = input_imgs.width
    height = input_imgs.height
    layer = input_imgs.layer
    nv = input_imgs.nv()
    
    mu = mu/layer

    # ----------------------------------------
    # Discretize incident light
    # ----------------------------------------
    diameter = est.set_diameter(NA, layer)
    ray_num, ray_check = est.set_incidental_light(diameter, apa_size, resolution)
    intensity = 1/ray_num

    # ----------------------------------------
    # out_path
    # ----------------------------------------
    result_name = target_name + '_ray' + str(ray_num)
    out_path = os.path.join(results, result_name)  # path for Estimated images
    util.mkdir(out_path)

    # ----------------------------------------
    # generate the ray matrix
    # ----------------------------------------
    ray_mat = est.range_matrix_generation(ray_num, ray_check, layer, diameter, apa_size, resolution)
    ray_mat = torch.from_numpy(ray_mat).clone()    # (ray_num, 2*layer-1, , )
    print('light_path',ray_mat.shape)

    ray_mat = torch.unsqueeze(ray_mat, 1)
    ray_mat = ray_mat.to(torch.float32)
    print('ray matrix', ray_mat.shape)  # (ray_num, 1, 2*layer-1, , )
    print('ray mem', sys.getsizeof(ray_mat.storage()), 'byte')
    
    ray_mat = ray_mat.to(device)
    kernel_list = []
    for s in range(layer):
        kernel_list.append(ray_mat[:, :, s:s+layer, :, :])
    print(kernel_list[0].shape)
    # ----------------------------------------
    # setting trans object
    # ----------------------------------------
    alpha = est.default_transmittance(0, input_imgs.stat)
    print('alpha', alpha.shape)

    alpha = torch.log(alpha)
    alpha = torch.unsqueeze(alpha, dim=0)
    alpha = torch.unsqueeze(alpha, dim=0)   # (1, 1, layer, height, width)
    
    omega = torch.tensor(alpha.clone().detach(), requires_grad=True, device=device) #TODO: modify omega's initialization method

    alpha_I = est.get_transmittance(target_name, nv)
    alpha_I = torch.reshape(alpha_I, (layer, height, width))
    print(alpha_I.shape)

    real_imgs = input_imgs.stat
    real_imgs = real_imgs.to(device)
    print('real_imgs', real_imgs.shape)

    error = nn.MSELoss(reduction='sum')
    tv_loss = TotalVariation3D(is_mean_reduction=False)
    optimizer = torch.optim.SGD([omega], lr=lr)
    # optimizer = torch.optim.Adam([omega], lr=lr)
    rmse = nn.MSELoss(reduction='mean')

    est_imgs = []

    start = time.perf_counter()
    for i in range(iter_num):
        print('iteration', i)
        optimizer.zero_grad()
        for s in range(layer):
            out = F.conv3d(omega, kernel_list[s], padding=(0, 13, 13)).squeeze()
            out = intensity * torch.sum(torch.exp(out), dim=0).squeeze()
            # loss = error(out,real_imgs[s, :, :])
            loss = error(out,real_imgs[s, :, :]) + mu * tv_loss(omega) # TODO: TV norm to tmp 
            loss.backward()
            if i==iter_num-1:
                est_imgs.append(out.cpu())
        optimizer.step()

    end = time.perf_counter()

    # ----------------------------------------
    # clipping weight
    # ----------------------------------------
    omega = torch.clip(omega, min=math.log(0.01), max=0)
    
    # ----------------------------------------
    # calculate rmse
    # ----------------------------------------
    trans = torch.exp(omega.detach())
    trans = trans.to('cpu')
    alpha_loss = rmse(trans.squeeze(), alpha_I)
    print('alpha_loss', torch.sqrt(alpha_loss))
    print('amount time =', end-start)
    
    # ----------------------------------------
    # generate image from brightness value
    # ----------------------------------------
    image_path = os.path.join(out_path, 'image')
    util.mkdir(image_path)
    for i, img in enumerate(est_imgs):
        img = img.data.squeeze().float().clamp_(0, 1).numpy()
        img = np.uint8((img*255.0).round())
        if i < 10:
            util.imsave(img, os.path.join(image_path, '0'+str(i)+'_'+target_name+'.bmp'))
        else:
            util.imsave(img, os.path.join(image_path, str(i)+'_'+target_name+'.bmp'))

    # ----------------------------------------
    # generate slice from transmittance
    # ----------------------------------------
    slice_path = os.path.join(out_path, 'slice')
    util.mkdir(slice_path)
    imgs_trans = est.tensor2imglist(trans)
    for i in range(layer):
        if i < 10:
            util.imsave(imgs_trans[i], os.path.join(slice_path, 'est_trans_0' + str(i) + '.bmp'))
        else:
            util.imsave(imgs_trans[i], os.path.join(slice_path, 'est_trans_' + str(i) + '.bmp'))

    # ----------------------------------------
    # output transmittance as .txt 
    # ----------------------------------------
    trans = torch.reshape(trans, (-1,))
    list_trans = torch.Tensor.tolist(trans)
    list_str_trans = [f'{n:.06f}' for n in list_trans]
    voxeldata_path = os.path.join(out_path, 'voxeldata.txt')
    with open(voxeldata_path, mode='w') as f:
        f.write(' '.join(list_str_trans))

if __name__ == '__main__':
    main()