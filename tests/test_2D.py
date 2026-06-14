import gdstk
import CorrDis
import numpy as np
from CorrDis import noise_generation as noise

lib = gdstk.Library(unit=1e-9)

Sd = 0.45
Lc = 3.5
nb_res = 20

average_distance = 8000
l_cube = 1000

filename = f"gaussian_Sd{np.round(Sd,2)}_Lc{np.round(Lc, 2)}_N{nb_res}"
cell = lib.new_cell(filename)
coords = noise.corrective_2D(Lc, Lc, Sd, Sd, nb_res, nb_res)
k = 0
for i in range(nb_res):
    for j in range(nb_res):
        x = (coords[k][0] + i) * average_distance
        y = (coords[k][1] + j) * average_distance
        square = gdstk.regular_polygon((x, y), l_cube, 4)
        cell.add(square)
        k += 1

cell.write_svg("/home/denis/Documents/git/Code_RCWA/C2N_design/test_corrective.svg")
