
import numpy as np


def out_f(out_path, out_filename):
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


def coord_difference_f(COORDINATES, X_COORDINATES, Y_COORDINATES, Z_COORDINATES):
    """
    To be added...

    Returns
    -------
    XX_DIFFERENCE : Difference between x and x.

    YY_DIFFERENCE : Difference between y and y.

    ZZ_DIFFERENCE : Difference between z and z.

    XY_DIFFERENCE : Difference between x and y.

    XZ_DIFFERENCE : Difference between x and z.

    YZ_DIFFERENCE : Difference between y and z.

    YX_DIFFERENCE : Difference between y and x.

    ZX_DIFFERENCE : Difference between z and x.

    ZY_DIFFERENCE : Difference between z and y.
    """
    return (
        (COORDINATES - COORDINATES[:, None])[..., 0],
        (COORDINATES - COORDINATES[:, None])[..., 1],
        (COORDINATES - COORDINATES[:, None])[..., 2],
        (Y_COORDINATES - X_COORDINATES[:, None])[..., 0],
        (Z_COORDINATES - X_COORDINATES[:, None])[..., 0],
        (Z_COORDINATES - Y_COORDINATES[:, None])[..., 0],
        (X_COORDINATES - Y_COORDINATES[:, None])[..., 0],
        (X_COORDINATES - Z_COORDINATES[:, None])[..., 0],
        (Y_COORDINATES - Z_COORDINATES[:, None])[..., 0]
    )


def T_f(R, XX_DIFFERENCE, YY_DIFFERENCE, ZZ_DIFFERENCE, XY_DIFFERENCE,
        XZ_DIFFERENCE, YZ_DIFFERENCE, YX_DIFFERENCE, ZX_DIFFERENCE, ZY_DIFFERENCE):
    """
    To be added...
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        TXX = ((3 * XX_DIFFERENCE**2) / R**5) - (1 / R**3)
        TYY = ((3 * YY_DIFFERENCE**2) / R**5) - (1 / R**3)
        TZZ = ((3 * ZZ_DIFFERENCE**2) / R**5) - (1 / R**3)
        TXY = ((3 * XY_DIFFERENCE**2) / R**5) - (1 / R**3)
        TXZ = ((3 * XZ_DIFFERENCE**2) / R**5) - (1 / R**3)
        TYZ = ((3 * YZ_DIFFERENCE**2) / R**5) - (1 / R**3)
        TYX = ((3 * YX_DIFFERENCE**2) / R**5) - (1 / R**3)
        TZX = ((3 * ZX_DIFFERENCE**2) / R**5) - (1 / R**3)
        TZY = ((3 * ZY_DIFFERENCE**2) / R**5) - (1 / R**3)

    TXX[np.isnan(TXX)] = 0
    TYY[np.isnan(TYY)] = 0
    TZZ[np.isnan(TZZ)] = 0
    TXY[np.isnan(TXX)] = 0
    TXZ[np.isnan(TYY)] = 0
    TYZ[np.isnan(TZZ)] = 0
    TYX[np.isnan(TXX)] = 0
    TZX[np.isnan(TYY)] = 0
    TZY[np.isnan(TZZ)] = 0

    return TXX, TYY, TZZ, TXY, TXZ, TYZ, TYX, TZX, TZY


def combine_arrays(arrays):
    """
    To be added...
    """
    arrays = np.asarray(arrays)
    n, p, q = arrays.shape
    s = int(round(np.sqrt(n)))
    arrays = arrays.reshape(s, -1, p, q)

    return arrays.transpose(2, 0, 3, 1).reshape(s * p, -1)


def alpha_f(A, W, g, Wp, We, Fp, Te, STAT_POL, aMXX, aMYY, aMZZ,
            aMXY, aMXZ, aMYZ):
    """
    To be added...
    """
    aNP = STAT_POL * ((Fp * Wp**2) / ((We**2 - W[g]**2) - ((W[g] * Te) * 1j)))
    aNP = np.sum(aNP)
    np.fill_diagonal(A, aNP)

    A[0, 0] = aMXX[g]
    A[0, 1] = aMXY[g]
    A[0, 2] = aMXZ[g]

    A[1, 0] = aMXY[g]
    A[1, 1] = aMYY[g]
    A[1, 2] = aMYZ[g]

    A[2, 0] = aMXZ[g]
    A[2, 1] = aMXY[g]
    A[2, 2] = aMZZ[g]

    return np.linalg.inv(A)
