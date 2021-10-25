import numpy as np
import math
import matplotlib.pyplot as plt


def psf_function(diameter, ratio, divide):
    """
    def psf_function(float diameter, int ratio, int divide)
    return psf[sqrt(ray_num)][sqrt(ray_num)]
    """
    radius = diameter / 2.0  # 半径
    split_iter = int(math.sqrt(divide))

    area = ratio * ratio * radius * radius
    length = int(math.ceil(diameter))
    if (length % 2) == 0:
        length += 1
    center = length / 2
    offset = length / split_iter

    a = np.zeros((split_iter, split_iter))

    # x_list = []
    # y_list = []

    for i in range(math.ceil(split_iter / 2)):
        for j in range(math.ceil(split_iter / 2)):
            for k in range(ratio):
                for l in range(ratio):
                    x = ((i * ratio + k) * offset - (ratio * center)) * ((i * ratio + k) * offset - (ratio * center))
                    y = ((j * ratio + l) * offset - (ratio * center)) * ((j * ratio + l) * offset - (ratio * center))
                    if (x + y) <= area:
                        a[i][j] += 1
                        # x_list.append((i + float(k / ratio)) * offset)
                        # y_list.append((j + float(l / ratio)) * offset)
            a[i][split_iter - 1 - j] = a[i][j]
            a[split_iter - 1 - i][j] = a[i][j]
            a[split_iter - 1 - i][split_iter - 1 - j] = a[i][j]

    a = a / (ratio * ratio)
    print('a=', a)
    a = (a * offset * offset) / (radius * radius * math.pi)
    print('a=', a)

    """
    plt.scatter(x_list, y_list)
    plt.title("x / y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
    """

    np.savetxt('sample.txt', a, delimiter=',')

    return a


if __name__ == '__main__':
    print('diameter=', end='')
    diameter_1 = float(input())  # 直径
    print('ratio=', end='')
    ratio_1 = int(input())  # 倍率
    print('How to divide=', end='')
    divide_1 = int(input())  # 領域の分割数
    psf = psf_function(diameter_1, ratio_1, divide_1)
