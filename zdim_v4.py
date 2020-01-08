#==============================================================================#
#                                                                              |
# Induced dipole moment interaction computations for a molecule-NP system.     |
#                                                                              |
# Zacharias Liasi                                                              |
# KU mail: flx527@alumni.ku.dk                                                 |
#                                                                              |
# September 2019                                                               |
#                                                                              |
# Version 4                                                                    |
#                                                                              |
# GitHub repository: https://github.com/zliasi/zdim                            |
#                                                                              |
#==============================================================================#

import os
import numpy as np
import time
import psutil
from zdimpy import (
    fileread as f,
    calc,
    shprint as sh,
    txtprint as txt,
    plot,
)
import sys
import postnord

if sys.argv[1] == "sonic":

    xyz_path = "/home/liasi/Documents/zdim/xyz/"

    output_path = "/home/liasi/Documents/zdim/output/"

    out_filename = "DHA_iso.out"

    out_path = "/home/liasi/Documents/zdim/out/"

    E_EXTERNAL = np.array([5, 0, 0])

    START_TIME = time.time()

    path = "/home/liasi/Documents/zdim/xyz/"

elif sys.argv[1] == "postnord":
    postnord.overpriced_service()


for xyz_filename in os.listdir(path):

    txt_output = txt.initialise_file(
        xyz_filename, out_filename, output_path)

    COORDINATES, X_COORDINATES, Y_COORDINATES, Z_COORDINATES = f.xyz(
        xyz_path, xyz_filename)

    W, aMXX, aMYY, aMZZ = f.out(out_path, out_filename)

    R, R0 = calc.spatial_dist(COORDINATES)

    DIFFERENCE_X, DIFFERENCE_Y, DIFFERENCE_Z = calc.coord_difference(
        COORDINATES)

    TXX, TYY, TZZ = calc.T(DIFFERENCE_X, DIFFERENCE_Y, DIFFERENCE_Z, R)

    A = calc.A(TXX, TYY, TZZ)

    Wp, We, Fp, Te, STAT_POL = calc.alpha_parameters(xyz_filename)

    dipM_X = []
    dipM_Y = []
    dipM_Z = []
    absM_X = []
    absM_Y = []
    absM_Z = []

    for g, freq in enumerate(W):
        sh.iteration(W, g)
        txt.iteration(txt_output, W, g)

        A = calc.alpha(A, W, g, Wp, We, Fp, Te, STAT_POL, aMXX, aMYY, aMZZ)

        dipole_x = np.full((len(COORDINATES), 1), 1 + 0.j)
        dipole_y = np.full((len(COORDINATES), 1), 1 + 0.j)
        dipole_z = np.full((len(COORDINATES), 1), 1 + 0.j)

        counter = 0

        while True:

            counter += 1

            E = calc.E(R0, E_EXTERNAL, dipole_x, dipole_y, dipole_z,
                       X_COORDINATES, Y_COORDINATES, Z_COORDINATES)

            dipole = np.dot(A, E)

            check_x = dipole[0:len(dipole):3]
            check_y = dipole[1:len(dipole):3]
            check_z = dipole[2:len(dipole):3]

            if (np.array_equal(dipole_x, check_x) == True
                and np.array_equal(dipole_y, check_y) == True
                and np.array_equal(dipole_z, check_z) == True
                    or counter > 999):

                sh.dipole(counter, dipole)
                txt.dipole(txt_output, counter, dipole)

                plot.dipole_append(dipM_X, dipM_Y, dipM_Z,
                                   absM_X, absM_Y, absM_Z, dipole)

                break

            dipole_x = dipole[0:len(dipole):3]
            dipole_y = dipole[1:len(dipole):3]
            dipole_z = dipole[2:len(dipole):3]

    sh.min(W, absM_X, absM_Y, absM_Z)
    txt.min(txt_output, W, absM_X, absM_Y, absM_Z)

    sh.misc(START_TIME)
    txt.misc(txt_output, START_TIME)

    plot.dipole(W, dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z,
                xyz_filename, COORDINATES, output_path, out_filename)
