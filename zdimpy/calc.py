import numpy as np


def spatial_dist(COORDINATES):
    """
    Calculates the relative spatial distance between the atoms and the spatial
    distance from the origin to each atom.

    Parameters
    ----------
    COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Returns
    -------
    R : Relative spacial distance from each atom to all other atoms.
    R0 : Spatial distance from the origin (0, 0, 0) to each atom.
    """
    return (
        np.linalg.norm(COORDINATES - COORDINATES[:, None], axis=-1),
        np.linalg.norm(np.array([0, 0, 0]) - COORDINATES[:, None], axis=-1)
    )


def coord_difference(COORDINATES):
    """
    Calculates the difference between all the x coordinate, y coordinates and
    z coordinates, respectively.

    Parameters
    ----------
    COORDINATES : np.array() containing the coordinates for the molecule and the
    NP (see xyz_reader()).

    Returns
    -------
    DIFFERENCE_X : Difference between all the x coordinates.

    DIFFERENCE_Y : Difference between all the y coordinates.

    DIFFERENCE_Z : Difference between all the z coordinates.
    """
    return (
        (COORDINATES - COORDINATES[:, None])[..., 0],
        (COORDINATES - COORDINATES[:, None])[..., 1],
        (COORDINATES - COORDINATES[:, None])[..., 2]
    )


def T(DIFFERENCE_X, DIFFERENCE_Y, DIFFERENCE_Z, R):
    """
    Calculates all the isotropic T components.

    Parameters
    ----------
    DIFFERENCE_X : Difference between all the x coordinates.

    DIFFERENCE_Y : Difference between all the y coordinates.

    DIFFERENCE_Z : Difference between all the z coordinates.

    Returns
    -------
    TXX : np.array() containing all the x components of the T tensors.

    TYY : np.array() containing all the y components of the T tensors.

    TZZ : np.array() containing all the z components of the T tensors.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        TXX = ((3 * DIFFERENCE_X**2) / R**5) - (1 / R**3)
        TYY = ((3 * DIFFERENCE_Y**2) / R**5) - (1 / R**3)
        TZZ = ((3 * DIFFERENCE_Z**2) / R**5) - (1 / R**3)
    pass

    TXX[np.isnan(TXX)] = 0
    TYY[np.isnan(TYY)] = 0
    TZZ[np.isnan(TZZ)] = 0

    return TXX, TYY, TZZ


def A(TXX, TYY, TZZ):
    """
    Constructs a np.array() containing the T tensor matrices as the basis for
    the A matrix.

    Parameters
    ----------
    TXX: np.array() containing all the x components of the T tensors.

    TYY: np.array() containing all the y components of the T tensors.

    TZZ: np.array() containing all the z components of the T tensors.

    Returns
    -------
    A: np.array() shaped based on the size and shape of the T tensor.
    """
    M = np.array((TXX, TYY, TZZ))
    i, j, k = M.shape
    A = np.zeros((i * j, i * k), M.dtype)
    np.einsum("jiki -> ijk", A.reshape(j, i, k, i))[...] = M

    A = A.astype(complex)

    A = np.negative(A)

    return A


def alpha_parameters(xyz_filename):
    """
    Initialises the parameters needed for the apha() function based on
    which nanoparticle the computations are done for.

    Parameters
    ----------
    xyz_filename : Name of the .xyz file to be used.

    Returns
    -------
    Wp : Plasmon frequency from the out_read() function.

    We : Average excitation energy from the out_read() function.

    Fp : Plasmon strength factor from the out_read() function.

    Te : Lifetime of the excitation from the out_read() function.

    STAT_POL : The static polarizability for the given NP from the out_read()
               function.
    """
    if "cu" in xyz_filename:
        Wp = 10.83
        We = np.array([[0.291], [2.957], [5.300], [11.18]])
        Fp = np.array([[0.061], [0.104], [0.723], [0.638]])
        Te = np.array([[0.378], [1.056], [3.213], [4.305]])
        STAT_POL = 33.7420

    elif "ag" in xyz_filename:
        Wp = 9.01
        We = np.array([[0.816], [4.481], [8.185], [9.083], [20.29]])
        Fp = np.array([[0.065], [0.124], [0.011], [0.840], [5.646]])
        Te = np.array([[3.886], [0.452], [0.065], [0.916], [2.419]])
        STAT_POL = 49.9843

    else:
        Wp = 9.03
        We = np.array([[0.415], [0.830], [2.969], [4.304], [13.32]])
        Fp = np.array([[0.024], [0.010], [0.071], [0.601], [4.384]])
        Te = np.array([[0.241], [0.345], [0.870], [2.494], [2.214]])
        STAT_POL = 31.0400

    return Wp, We, Fp, Te, STAT_POL


def alpha(A, W, g, Wp, We, Fp, Te, STAT_POL, aMXX, aMYY, aMZZ):
    """
    Calculates the polarizabilites for the NP and inserts the newly calculated
    NP polarizabilites and the molecule polarizabilites from out_reader() in
    the A matrix.

    Parameters
    ----------
    W : np.array() containing the frequencies from the .out file.

    g : Counter variable that is used to run through the np.array() containing
        the frequencies (W).

    Wp : Plasmon frequency from the out_read() function.

    We : Average excitation energy from the out_read() function.

    Fp : Plasmon strength factor from the out_read() function.

    Te : Lifetime of the excitation from the out_read() function.

    STAT_POL : The static polarizability for the given NP from the out_read()
               function.

    aMXX : xx components of the molecules polarizability from the out_reader()
           function.

    aMYY : yy components of the molecules polarizability from the out_reader()
           function.

    aMZZ : zz components of the molecules polarizability from the out_reader()
           function.

    Returns
    -------
    A : A modified version of the A matrix from the A() function containing the
    isotropic polarizabilites for both the molecule and the NP.
    """
    aNP = STAT_POL * ((Fp * Wp**2) / ((We**2 - W[g]**2) - ((W[g] * Te) * 1j)))
    aNP = np.sum(aNP)
    np.fill_diagonal(A, aNP)

    A[0, 0] = aMXX[g]
    A[1, 1] = aMYY[g]
    A[2, 2] = aMZZ[g]

    return np.linalg.inv(A)


def E(R0, E_EXTERNAL, dipole_x, dipole_y, dipole_z, X_COORDINATES,
      Y_COORDINATES, Z_COORDINATES):
    """
    Computes the x, y, and z components of the electrical field, and combines
    them into one np.array() named E.

    Parameters
    ----------
    E_EXTERNAL : np.array() containing the Cartesian components of the external
                 electrical field.

    dipole_x : np.array() containing the x components of the induced dipole
               moments for both the molecule and NP.

    dipole_y : np.array() containing the y components of the induced dipole
               moments for both the molecule and NP.

    dipole_z : np.array() containing the z components of the induced dipole
               moments for both the molecule and NP.

    R0 : Spatial distance from the origin (0, 0, 0) to each atom.

    X_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Y_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Z_COORDINATES : An np.array() containing the coordinates from the given .xyz
                  file incl. an automatically created single-point molecule.

    Returns
    -------
    E: np.array() containing the electrical field vectors.
    """
    Ex = E_EXTERNAL[0] + (
        dipole_x * (1 / (R0**3) - 3 * ((X_COORDINATES**2) / (R0**5)))
        + dipole_y * ((- 3 * X_COORDINATES * Y_COORDINATES) / R0**5)
        + dipole_z * (-(3 * X_COORDINATES * Z_COORDINATES) / R0**5)
    )

    Ey = E_EXTERNAL[1] + (
        dipole_y * (1 / (R0**3) - 3 * ((Y_COORDINATES**2) / (R0**5)))
        + dipole_x * ((- 3 * Y_COORDINATES * X_COORDINATES) / R0**5)
        + dipole_z * (- (3 * Y_COORDINATES * Z_COORDINATES) / R0**5)
    )

    Ez = E_EXTERNAL[2] + (
        dipole_z * (1 / (R0**3) - 3 * ((Z_COORDINATES**2) / (R0**5)))
        + dipole_y * ((- 3 * Z_COORDINATES * Y_COORDINATES) / R0**5)
        + dipole_x * (-(3 * Y_COORDINATES * Z_COORDINATES) / R0**5)
    )

    E = []
    for element in zip(Ex, Ey, Ez):
        E.extend(element)

    E = np.asarray(E)
    E = E.astype(complex)

    return E
