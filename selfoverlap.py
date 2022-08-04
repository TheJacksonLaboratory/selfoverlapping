import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np

import cv2
import test_polys

from selfover_polys import subdivide_selfoverlapping



if __name__ == '__main__':
    for poly_test in [test_polys.test_1, test_polys.test_2, test_polys.test_3, test_polys.test_4, test_polys.test_5, test_polys.test_6, test_polys.test_7, test_polys.test_8, test_polys.test_9, test_polys.test_10]:
        vertices = poly_test()

        # try:
        polys = subdivide_selfoverlapping(vertices[0])

        fig, ax = plt.subplots()
        
        patches = []
        for id, poly in enumerate(polys):
            patches.append(Polygon(poly, True))

        ax.plot(vertices[0][:, 0], vertices[0][:, 1], 'b-')
        ax.plot([vertices[0][-1, 0], vertices[0][0, 0]], [vertices[0][-1, 1], vertices[0][0, 1]], 'b-')

        colors = 100 * np.random.rand(len(polys))
        p = PatchCollection(patches, alpha=0.5)
        p.set_array(colors)
        ax.add_collection(p)
        
        plt.show()
        # except ValueError:
        #     print('Polygon is not self-overlapping')
        # except IndexError:
        #     print('Polygon is not self-overlapping')
        
        polys = vertices
        
        fig, ax = plt.subplots()
        ax.plot(vertices[0][:, 0], vertices[0][:, 1], 'b-')
        ax.plot([vertices[0][-1, 0], vertices[0][0, 0]], [vertices[0][-1, 1], vertices[0][0, 1]], 'b-')

        patches = []
        for poly in polys:
            patches.append(Polygon(poly, True))
            
        colors = 100 * np.random.rand(len(polys))
        p = PatchCollection(patches, alpha=0.5)
        p.set_array(colors)
        ax.add_collection(p)
        
        plt.show()

        im = np.zeros([700, 700, 3], dtype=np.uint8)
        
        cv2.drawContours(im, vertices, 0, (127, 127, 127), -1)
        
        for x, y in vertices[0]:
            cv2.circle(im, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        for p, poly in enumerate(polys):
            color = int((p+1) / len(polys) * 255.0)
            color = (color, color, color)
            cv2.drawContours(im, [poly.astype(np.int32)], 0, color, -1)
            
        cv2.drawContours(im, vertices, 0, (255, 0, 0), 1)
        
        cv2.imshow('Filled poly', im)
        cv2.waitKey()
        cv2.destroyAllWindows()
