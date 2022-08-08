import matplotlib.pyplot as plt
import numpy as np
from selfover_polys import subdivide_selfoverlapping
from skimage.draw import polygon
import test_polys


if __name__ == '__main__':
    poly = test_polys.test_10()
    poly = poly[0, ::-1, :]

    img = np.zeros((512, 512, 3), dtype=np.double)
    rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)
    img[rr, cc, 1] = 1

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')

    sub_polys = subdivide_selfoverlapping(poly)
    img_sub_polys = np.zeros((512, 512, 3), dtype=np.double)
    for col, sub_poly in enumerate(sub_polys):
        rr, cc = polygon(sub_poly[:, 0], sub_poly[:, 1], img_sub_polys.shape)
        img_sub_polys[rr, cc, 1] = col + 1

    plt.subplot(1, 2, 2)
    plt.imshow(img_sub_polys)
    plt.axis('off')

    plt.show()