import numpy as np


def xyz(xyz_path, xyz_filename):
    """
    Reads the given .xyz file(s) and creates np.array()'s with the coordinates.
    It also creates a single-point molecule 3.25pts away from the NP along the
    x axis.

    Parameters
    ----------
    xyz_path : The absolute file path of the directory containing the .xyz
               file(s) needed for the computations.

    xyz_filename : Name of the .xyz file to be used.

    Returns
    -------
    COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    X_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Y_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Z_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.
    """
    NP_COORDINATES = []
    X_COORDINATES = []
    Y_COORDINATES = []
    Z_COORDINATES = []

    with open(xyz_path + xyz_filename) as XYZ:
        n_atoms = XYZ.readline()
        title = XYZ.readline()
        for line in XYZ:
            atom, x, y, z = line.split()
            NP_COORDINATES.append([float(x), float(y), float(z)])
            X_COORDINATES.append([float(x)])
            Y_COORDINATES.append([float(y)])
            Z_COORDINATES.append([float(z)])

    NP_COORDINATES = np.asarray(NP_COORDINATES)
    X_COORDINATES = np.asarray(X_COORDINATES)
    Y_COORDINATES = np.asarray(Y_COORDINATES)
    Z_COORDINATES = np.asarray(Z_COORDINATES)

    x = float(max(X_COORDINATES)) + 3.25

    M_COORDINATES = np.array([x, 0, 0])

    COORDINATES = np.vstack((M_COORDINATES, NP_COORDINATES))

    X_COORDINATES = np.vstack((M_COORDINATES[0], X_COORDINATES))
    Y_COORDINATES = np.vstack((M_COORDINATES[1], Y_COORDINATES))
    Z_COORDINATES = np.vstack((M_COORDINATES[2], Z_COORDINATES))

    return COORDINATES, X_COORDINATES, Y_COORDINATES, Z_COORDINATES


def out(out_path, out_filename):
    """
    Reads the given .out file(s) and creates np.array()'s with the frequencies
    and complex polarizabilites for the molecule.

    Parameters
    ----------
    out_path : The absolute file path of the directory containing the .out
               file(s) needed for the computations.

    out_filename : String variable containing the filename of the .out file used
                   for the current run of the code.

    Returns
    -------
    W : An np.array() containing the frequencies from the .out file.

    aMXX : An np.array() containing the x components of the complex atomistic
           polarizabilites of the molecule.

    aMYY : An np.array() containing the y components of the complex atomistic
           polarizabilites of the molecule.

    aMZZ : An np.array() containing the z components of the complex atomistic
           polarizabilites of the molecule.
    """
    W = []
    aMXX = []
    aMYY = []
    aMZZ = []

    with open(out_path + out_filename) as OUT:
        for line in OUT:
            if "XDIPLEN   XDIPLEN" in line:
                XX = line.strip().split()
                aMXX.append(complex(float(XX[4]), float(XX[5])))
                W.append(float(XX[3]))

            if "YDIPLEN   YDIPLEN" in line:
                YY = line.strip().split()
                aMYY.append(complex(float(YY[4]), float(YY[5])))

            if "ZDIPLEN   ZDIPLEN" in line:
                ZZ = line.strip().split()
                aMZZ.append(complex(float(ZZ[4]), float(ZZ[5])))

    W = np.asarray(W)
    aMXX = np.asarray(aMXX)
    aMYY = np.asarray(aMYY)
    aMZZ = np.asarray(aMZZ)

    return W, aMXX, aMYY, aMZZ


def out_extra(out_path, out_filename):
    """
    To be added...
    """
    W = []

    aMXX = []
    aMYY = []
    aMZZ = []
    aMXY = []
    aMXZ = []
    aMYZ = []

    with open(out_path + out_filename) as OUT:
        for line in OUT:
            if "XDIPLEN   XDIPLEN" in line:
                XX = line.strip().split()
                aMXX.append(complex(float(XX[4]), float(XX[5])))
                W.append(float(XX[3]))

            if "YDIPLEN   YDIPLEN" in line:
                YY = line.strip().split()
                aMYY.append(complex(float(YY[4]), float(YY[5])))

            if "ZDIPLEN   ZDIPLEN" in line:
                ZZ = line.strip().split()
                aMZZ.append(complex(float(ZZ[4]), float(ZZ[5])))

            if "XDIPLEN   YDIPLEN" in line:
                XY = line.strip().split()
                aMXY.append(complex(float(XY[4]), float(XY[5])))

            if "XDIPLEN   ZDIPLEN" in line:
                XZ = line.strip().split()
                aMXZ.append(complex(float(XZ[4]), float(XZ[5])))

            if "YDIPLEN   ZDIPLEN" in line:
                YZ = line.strip().split()
                aMYZ.append(complex(float(YZ[4]), float(YZ[5])))

    W = np.asarray(W)
    aMXX = np.asarray(aMXX)
    aMYY = np.asarray(aMYY)
    aMZZ = np.asarray(aMZZ)
    aMXY = np.asarray(aMXY)
    aMXZ = np.asarray(aMXZ)
    aMYZ = np.asarray(aMYZ)

    return W, aMXX, aMYY, aMZZ, aMXY, aMXZ, aMYZ
