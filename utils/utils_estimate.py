import cv2
import torch
import numpy as np
from pathlib import Path
import math


def imglist2tensor(img_list):
    for idx, img in enumerate(img_list):
        if idx == 0:
            imgs = np.copy(np.expand_dims(img, axis=0))
        else:
            imgs = np.concatenate([imgs, np.expand_dims(img, axis=0)])
    imgs = imgs / 255.0
    imgs = torch.from_numpy(imgs)
    imgs = imgs.to(torch.float32)
    imgs = imgs.permute(3, 0, 1, 2)  # NHWC -> CNHW

    return imgs


def tensor2imglist(imgs):
    imgs = imgs.data.squeeze().float().clamp_(0, 1).numpy()
    return np.uint8((imgs*255.0).round())


def set_diameter(NA, layer_num):
    ratio = NA/math.sqrt(1-math.pow(NA, 2))
    diameter = ratio * (2*layer_num-1)
    return diameter


def set_criscross(diameter):
    if math.ceil(diameter)%2==0:
        crisscross = math.ceil(diameter)+1
    else:
        crisscross = math.ceil(diameter)
    return crisscross


def set_incidental_light(diameter, num_div, scale):
    diameter = diameter / scale
    area_size = set_criscross(diameter)

    num_div = int(num_div)
    radius = diameter/2
    center = area_size/2
    offset = area_size/num_div

    a = np.zeros((num_div, num_div))
    ray_check = []
    Nr = 0

    for i in range(num_div):
        for j in range(num_div):
            x = pow(offset * (0.5 + i) - center, 2)
            y = pow(offset * (0.5 + j) - center, 2)
            if(x+y) <= pow(radius, 2):
                a[i][j] = Nr
                ray_check.append(Nr)
                Nr += 1
            else:
                a[i][j] = -1
                ray_check.append(-1)

    return Nr, ray_check


def boundary_decision(pre, now):
    """
    boundary_decision
    Boundary determination in the xy-plane
    return: Bool value
    """
    judge = (math.floor(pre) != math.floor(now)) and (now != math.floor(now)) and (pre != math.floor(pre))
    return judge


def range_matrix_generation(ray_number, ray_check, layer_num, diameter, apa_size, resolution=0.92):
    """
    range_matrix_generation
    """
    depth = 2*layer_num-1
    CRISSCROSS = set_criscross(diameter/resolution)
    offset = CRISSCROSS / apa_size
    center = CRISSCROSS / 2
    p = [0]
    z_step = 1/resolution  # 1.0869565173913175

    sign_x = sign_y = 0
    check_line_x = check_line_y = 0

    light_path = np.zeros((ray_number, depth, CRISSCROSS, CRISSCROSS))

    for n in range(0, math.ceil(apa_size/2), 1):
        # print('n =', n)
        for m in range(0, math.ceil(apa_size/2), 1):
            # print('m =', m)
            count = 0
            vox_iter2 = -1

            if ray_check[n * apa_size + m] != -1:
                # print('ray =', ray_check[n * apa_size + m])
                ray_iter = ray_check[n * apa_size + m]
                ray_sym1 = ray_check[n * apa_size + (apa_size - 1 - m)]
                ray_sym2 = ray_check[(apa_size - 1 - n) * apa_size + m]
                ray_sym3 = ray_check[(apa_size - 1 - n) * apa_size + (apa_size - 1 - m)]
                # print(ray_iter, ray_sym1, ray_sym2, ray_sym3)
                for z_iter in range(0, layer_num+1, 1):
                    z1 = (depth - z_iter) * z_step
                    k1 = 2 * z1 / (depth * z_step) - 1
                    y1 = center + k1 * ((n + 0.5) * offset - center)
                    x1 = center + k1 * ((m + 0.5) * offset - center)
                    # print("(x1,y1,z1) =", x1, y1, z1)

                    check = 0
                    x2 = y2 = x3 = y3 = 0
                    z2 = -100 * z_step
                    z3 = -200 * z_step

                    if count == 0:
                        p[0] = x1
                        p.append(y1)
                        p.append(z1)
                        count += 1

                    elif count > 0:
                        pre_x = p[(count - 1) * 3]
                        pre_y = p[(count - 1) * 3 + 1]
                        pre_z = p[(count - 1) * 3 + 2]
                        i = j = 0
                        nx = ny = 0
                        if boundary_decision(pre_x, x1):
                            nx = math.floor(x1) - math.floor(pre_x)
                            sign_x = 1
                            check_line_x = 1
                            check = 1
                        if boundary_decision(pre_y, y1):
                            ny = math.floor(y1) - math.floor(pre_y)
                            sign_y = 1
                            check_line_y = 1
                            if check != 1:
                                check = 2
                            else:
                                check = 3

                        while (i < nx) or (j < ny):
                            if ((check == 1) or (check == 3)) and (nx > 0):
                                x2 = math.floor(pre_x) + sign_x * i + check_line_x
                                if (m + 0.5) * offset - center != 0:
                                    k2 = (x2 - center) / ((m + 0.5) * offset - center)
                                else:
                                    k2 = 0
                                y2 = center + k2 * ((n + 0.5) * offset - center)
                                z2 = (k2 + 1) * depth * z_step / 2

                            if ((check == 2) or (check == 3)) and (ny > 0):
                                y3 = math.floor(pre_y) + sign_y * j + check_line_y
                                if (n + 0.5) * offset - center != 0:
                                    k3 = (y3 - center) / ((n + 0.5) * offset - center)
                                else:
                                    k3 = 0
                                x3 = center + k3 * ((m + 0.5) * offset - center)
                                z3 = (k3 + 1) * depth * z_step / 2

                            if z2 > z3:
                                if (z2 <= 21 * z_step) and (z2 >= 0) and (z2 != pre_z):
                                    p.append(x2)
                                    p.append(y2)
                                    p.append(z2)
                                    count += 1
                                i += 1
                                check = 1

                            elif z2 < z3:
                                if (z3 <= 21 * z_step) and (z3 >= 0) and (z3 != pre_z):
                                    p.append(x3)
                                    p.append(y3)
                                    p.append(z3)
                                    count += 1
                                j += 1
                                check = 2

                            else:
                                if (z2 <= 21 * z_step) and (z2 >= 0) and (z2 != pre_z) and (z3 <= 21 * z_step) and (z3 >= 0) and (z3 != pre_z):
                                    p.append(x2)
                                    p.append(y2)
                                    p.append(z2)
                                    count += 1
                                i += 1
                                j += 1
                                check = 3

                        p.append(x1)
                        p.append(y1)
                        p.append(z1)
                        count += 1
                        for k in range(0, count - 1, 1):
                            X1 = p[k * 3]
                            Y1 = p[k * 3 + 1]
                            Z1 = p[k * 3 + 2]
                            X2 = p[(k+1) * 3]
                            Y2 = p[(k+1) * 3 + 1]
                            Z2 = p[(k+1) * 3 + 2]

                            d = math.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2 + (Z1 - Z2) ** 2) * resolution
                            # print('z_iter', z_iter, 'distance', d)

                            z_tmp = Z1 * resolution

                            z_point = z_iter - 1
                            z_point_sym = depth - z_point - 1
                            y_point = math.floor(Y1)
                            y_point_sym = CRISSCROSS - 1 - math.floor(Y1)
                            x_point = math.floor(X1)
                            x_point_sym = CRISSCROSS - 1 - math.floor(X1)

                            vox_iter = z_point * (CRISSCROSS**2) + y_point * CRISSCROSS + x_point

                            if 0 <= math.floor(X1) < CRISSCROSS and 0 <= math.floor(Y1) < CRISSCROSS:
                                if z_tmp > 0 and d != 0:
                                    if vox_iter != vox_iter2:
                                        light_path[ray_iter, z_point, y_point, x_point] = d
                                        light_path[ray_sym1, z_point, y_point, x_point_sym] = d
                                        light_path[ray_sym2, z_point, y_point_sym, x_point] = d
                                        light_path[ray_sym3, z_point, y_point_sym, x_point_sym] = d
                                        if z_iter < layer_num+1:
                                            light_path[ray_sym3, z_point_sym, y_point, x_point] = d
                                            light_path[ray_sym2, z_point_sym, y_point, x_point_sym] = d
                                            light_path[ray_sym1, z_point_sym, y_point_sym, x_point] = d
                                            light_path[ray_iter, z_point_sym, y_point_sym, x_point_sym] = d
                                    vox_iter2 = vox_iter

                        p[0] = p[(count-1) * 3]
                        p[1] = p[(count-1) * 3 + 1]
                        p[2] = p[(count-1) * 3 + 2]
                        del p[2:-1]
                        count = 1
                del p[0:-1]
    return light_path


def adjust_ray(light_path, height, width):
    height = 2*height-1
    width = 2*width-1
    ex_light_path = torch.zeros(light_path.size()[0],
                                light_path.size()[1],
                                height,
                                width)

    y_dir = int((height - light_path.size()[2])/2)
    x_dir = int((width - light_path.size()[3])/2)
    y_end = light_path.size()[2]+y_dir
    x_end = light_path.size()[3]+x_dir

    ex_light_path[:, :, y_dir:y_end, x_dir:x_end] += light_path

    return ex_light_path


def get_transmittance(target_name, nv):
    """
    get_transmittance
    """
    if "simucell1" in target_name:
        target = "simucell1"
    elif "simucell2" in target_name:
        target = "simucell2"
    elif "simucell3" in target_name:
        target = "simucell3"
    elif "phantom1" in target_name:
        target = "phantom1"
    path = Path(__file__).parent
    root = "../inputs/voxeldata/" + target + "_voxdata.txt"
    path /= root
    path = Path(path)
    print(path)
    with open(path) as f:
        contents = f.read().split()

    a = torch.zeros(1, nv)
    for voxel_num in range(0, nv, 1):
        a[0][voxel_num] = float(contents[voxel_num])
    return a


def default_transmittance(flag, img):
    """
    default_transmittance
    flag:0 Set based on the luminance value of the teacher data
        :1 Initial value of transmittance = 0.75
    img:size[nv]
    :return: transmittance
    """
    a = torch.zeros_like(img)
    if flag == 0:
        a = torch.pow(img, 1/img.size()[0])
        a = a.to(torch.float32)

    elif flag == 1:
        a = a * 0.75
        a = a.to(torch.float32)

    return a


def main():
    layer = 11
    resolution = 0.92
    NA = 0.75
    for i in range(50):
        apa_size = 2*i + 1
        diameter = set_diameter(NA, layer)
        ray_num, ray_check = set_incidental_light(diameter, apa_size, resolution)
        light_path = range_matrix_generation(ray_num, ray_check, layer, diameter, apa_size, resolution)
        light_path = torch.from_numpy(light_path).clone()    # (ray_num, 2*layer-1, , )
        print('i=', apa_size, 'light_path',light_path.shape)
    # z = light_path.size()[1]
    # y = light_path.size()[2]
    # x = light_path.size()[3]
    # count = 0
    # for m in range(z):
    #     for j in range(y):
    #         for k in range(x):
    #             if light_path[4, m, j, k]!=0:
    #                 print(4, m, j, k)
    #                 print(light_path[4,m, j, k])
    #                 count+=light_path[4, m, j, k]
    # print('count', count)


if __name__ == '__main__':
    main()