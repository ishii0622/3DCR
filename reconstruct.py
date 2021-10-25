import torch
import os
import numpy as np
import torch.nn as nn
from utils import utils_image as util
from utils import utils_estimate as est
import time
import math
import sys
import matplotlib.pyplot as plt
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
        return sort


class MyNet(nn.Module):
    def __init__(self, layer, height, width):
        super(MyNet, self).__init__()
        self.conv3d = nn.Conv3d(in_channels  = 1, 
                                out_channels = 1, 
                                kernel_size  = (layer, height, width), 
                                bias         = False)

    def forward(self, x, intensity):
        x = self.conv3d(x)
        x = torch.squeeze(x)
        x = torch.exp(x)
        x = x * intensity
        x = torch.sum(x, dim=0)
        x = torch.squeeze(x)

        return x


def main():
    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    target_name = 'simucell1_input'
    apa_size = 5
    light_area_division = pow(apa_size, 2)
    resolution = 0.92
    NA = 0.75
    model_name = 'ircnn_gray'

    iter_num = 100

    # hyper parameter
    lr = 6.0e-3

    n_channels = 3 if 'color' in model_name else 1

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
    nv = input_imgs.nv()

    # ----------------------------------------
    # Discretize incident light
    # ----------------------------------------
    diameter = est.set_diameter(NA, input_imgs.layer)
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
    light_path = est.range_matrix_generation(ray_num, ray_check, input_imgs.layer, diameter, apa_size, resolution)
    light_path = torch.from_numpy(light_path).clone()    # (ray_num, 2*layer-1, , )
    print('light_path',light_path.shape)

    ray_mat = est.adjust_ray(light_path, input_imgs.height, input_imgs.width)
    ray_mat = torch.unsqueeze(ray_mat, 1)
    print('ray matrix', ray_mat.shape)  # (ray_num, 1, 2*layer-1, , )
    print('ray mem', sys.getsizeof(ray_mat.storage()), 'byte')
    
    ray_mat = ray_mat.to(device)

    # ----------------------------------------
    # setting trans object
    # ----------------------------------------
    alpha = est.default_transmittance(0, input_imgs.stat)
    print('alpha', alpha.shape)

    omega = torch.log(alpha)
    omega = torch.unsqueeze(omega, dim=0)
    omega = torch.unsqueeze(omega, dim=0)   # (1, 1, layer, height, width)

    alpha_I = est.get_transmittance(target_name, nv)
    alpha_I = torch.reshape(alpha_I, (input_imgs.layer, input_imgs.height, input_imgs.width))
    print(alpha_I.shape)

    real_imgs = input_imgs.adapt()
    real_imgs = real_imgs.to(device)
    print('real_imgs', real_imgs.shape)

    # ----------------------------------------
    # setting network
    # ----------------------------------------
    mynet = MyNet(input_imgs.layer, input_imgs.height, input_imgs.width)
    mynet.conv3d.weight.data = nn.Parameter(omega)
    mynet = mynet.to(device)

    error = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(mynet.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(mynet.parameters(), lr=lr)
    rmse = nn.MSELoss(reduction='mean')

    history = {
        'iter': [],
        'loss': [],
    }

    start = time.perf_counter()
    for i in range(iter_num):
        print('iteration', i)
        optimizer.zero_grad()
        out = mynet(ray_mat, intensity)
        loss = error(out, real_imgs)
        loss.backward()
        optimizer.step()

    end = time.perf_counter()

    print('amount time =', end-start)
    
    # ----------------------------------------
    # clipping weight
    # ----------------------------------------
    for p in mynet.parameters():
        p.data.clamp_(math.log(0.01), 0)

    # ----------------------------------------
    # calculate rmse
    # ----------------------------------------
    trans = torch.exp(mynet.conv3d.weight.data.detach())
    trans = trans.to('cpu')
    alpha_loss = rmse(trans.squeeze(), alpha_I)
    print('alpha_loss', torch.sqrt(alpha_loss))

    # ----------------------------------------
    # generate image from brightness value
    # ----------------------------------------
    imgs_out = MultiImgs(out.float().clamp_(0, 1).cpu().unsqueeze(0))
    if imgs_out.color == 1:
        imgs_out.stat = torch.squeeze(imgs_out.stat)
    imgs_E = est.tensor2imglist(imgs_out.adapt())
    for i in range(input_imgs.layer):
        if i < 10:
            util.imsave(imgs_E[i], os.path.join(out_path, '0'+str(i)+'_'+model_name+'.bmp'))
        else:
            util.imsave(imgs_E[i], os.path.join(out_path, str(i) + '_' + model_name + '.bmp'))

    # ----------------------------------------
    # generate image from transmittance
    # ----------------------------------------
    imgs_trans = est.tensor2imglist(trans)
    for i in range(input_imgs.layer):
        if i < 10:
            util.imsave(imgs_trans[i], os.path.join(out_path, 'est_trans_0' + str(i) + '.bmp'))
        else:
            util.imsave(imgs_trans[i], os.path.join(out_path, 'est_trans_' + str(i) + '.bmp'))


if __name__ == '__main__':
    main()