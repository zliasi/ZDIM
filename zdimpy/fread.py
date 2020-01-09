import numpy as np


def xyz(path):
    """
    Read the given .xyz file and stores the values in an array.

    Parameters
    ----------
    path : The absolute file path of the .xyz file.

    Returns
    -------
    coordinates : array containing the coordinates from the .xyz file.
    """
    coordinates = []
    x_coordinates = []
    y_coordinates = []
    z_coordinates = []

    with open(path) as fp:
        n_atoms = fp.readline()
        title = fp.readline()
        for line in fp:
            atom, x, y, z = line.split()
            coordinates.append([float(x), float(y), float(z)])
            x_coordinates.append([float(x)])
            y_coordinates.append([float(y)])
            z_coordinates.append([float(z)])

    return (
        np.asarray(coordinates),
        np.asarray(x_coordinates),
        np.asarray(y_coordinates),
        np.asarray(z_coordinates)
    )


def out_NP(path):
    """
    Reads the given .out file and stores the frequencies and polarizabilities in
    separate array's.

    Parameters
    ----------
    path : The absolute file path of the .out file.

    Returns
    -------
    freq : array containing the frequencies from the .out file.

    aNP_xx : array containing the xx components of the complex atomistic
             polarizabilites from the .out file.

    aNP_yy : array containing the yy components of the complex atomistic
             polarizabilites from the .out file.

    aNP_zz : array containing the zz components of the complex atomistic
             polarizabilites from the .out file.
    """
    freq = []
    aNP_xx = []
    aNP_yy = []
    aNP_zz = []

    with open(path) as fp:
        for line in fp:
            if "XDIPLEN   XDIPLEN" in line:
                XX = line.strip().split()
                aNP_xx.append(complex(float(XX[4]), float(XX[5])))
                freq.append(float(XX[3]))

            if "YDIPLEN   YDIPLEN" in line:
                YY = line.strip().split()
                aNP_yy.append(complex(float(YY[4]), float(YY[5])))

            if "ZDIPLEN   ZDIPLEN" in line:
                ZZ = line.strip().split()
                aNP_zz.append(complex(float(ZZ[4]), float(ZZ[5])))

    return (
        np.asarray(freq),
        np.asarray(aNP_xx),
        np.asarray(aNP_yy),
        np.asarray(aNP_zz)
    )


def out_MOL(path):
    """
    Reads the given .out file and stores the frequencies and polarizabilities in
    separate array's.

    Parameters
    ----------
    path : The absolute file path of the .out file.

    Returns
    -------
    freq : array containing the frequencies from the .out file.

    aMOL_xx : array containing the xx components of the complex atomistic
              polarizabilites from the .out file.

    aMOL_yy : array containing the yy components of the complex atomistic
              polarizabilites from the .out file.

    aMOL_zz : array containing the zz components of the complex atomistic
              polarizabilites from the .out file.
    """
    freq = []
    aMOL_xx = []
    aMOL_yy = []
    aMOL_zz = []

    with open(path) as fp:
        for line in fp:
            if "XDIPLEN   XDIPLEN" in line:
                XX = line.strip().split()
                aMOL_xx.append(complex(float(XX[4]), float(XX[5])))
                freq.append(float(XX[3]))

            if "YDIPLEN   YDIPLEN" in line:
                YY = line.strip().split()
                aMOL_yy.append(complex(float(YY[4]), float(YY[5])))

            if "ZDIPLEN   ZDIPLEN" in line:
                ZZ = line.strip().split()
                aMOL_zz.append(complex(float(ZZ[4]), float(ZZ[5])))

    return (
        np.asarray(freq),
        np.asarray(aMOL_xx),
        np.asarray(aMOL_yy),
        np.asarray(aMOL_zz)
    )


def freq(path):
    """
    Reads the given .out file and stores the frequencies in an array.

    Parameters
    ----------
    path : The absolute file path of the .out file.

    Returns
    -------
    freq : array containing the frequencies from the .out file.
    """
    freq = []

    with open(path) as fp:
        for line in fp:
            if "XDIPLEN   XDIPLEN" in line:
                XX = line.strip().split()
                freq.append(float(XX[3]))

    return np.asarray(freq)
