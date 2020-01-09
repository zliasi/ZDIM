import numpy as np


def pol_par(xyz_file_name):
    """
    Initialises the variables needed for the nano_pol() function as well as
    their values, depending on which element the computations are done for.

    Parameters
    ----------
    xyz_file_name : Name of the .xyz file.

    Returns
    -------
    pl_freq : Plasmon frequency.

    ex_freq : Average excitation energy.

    pl_str : Plasmon strength factor.

    ex_lftm : Excitation lifetime.

    stat_pol : The static polarizability of the given nanoparticle.
    """
    cu_set = {"cu", "copper", "kobber"}
    ag_set = {"ag", "silver", "sølv"}
    au_set = {"au", "gold", "guld"}

    if any(name in xyz_file_name.lower() for name in cu_set):
        pl_freq = 0.39799501649970476
        ex_freq = np.array(
            [
                [0.01069405],
                [0.10866771],
                [0.19477134],
                [0.41085727]
            ]
        )
        pl_str = np.array(
            [
                [0.00224171],
                [0.00382193],
                [0.02656975],
                [0.02344606]
            ]
        )
        ex_lftm = np.array(
            [
                [0.01389124],
                [0.03880727],
                [0.11807553],
                [0.15820578]
            ]
        )
        stat_pol = 33.7420

    elif any(name in xyz_file_name.lower() for name in ag_set):
        pl_freq = 0.3311160928
        ex_freq = np.array(
            [
                [0.02998787255],
                [0.1646760501],
                [0.3007974716],
                [0.3337988314],
                [0.7456543310]
            ]
        )
        pl_str = np.array(
            [
                [0.002388739848],
                [0.004556980633],
                [0.0004042482819],
                [0.03086986880],
                [0.2074896182]
            ]
        )
        ex_lftm = np.array(
            [
                [0.1428098931],
                [0.01661092940],
                [0.002388739848],
                [0.03366285693],
                [0.08889787218]
            ]
        )
        stat_pol = 49.9843

    elif any(name in xyz_file_name.lower() for name in au_set):
        pl_freq = 0.33184626029476766
        ex_freq = np.array(
            [
                [0.01525096],
                [0.03050193],
                [0.1091087],
                [0.15816903],
                [0.4895008]
            ]
        )
        pl_str = np.array(
            [
                [0.00088198],
                [0.00036749],
                [0.0026092],
                [0.02208633],
                [0.16110897]
            ]
        )
        ex_lftm = np.array(
            [
                [0.00885658],
                [0.01267851],
                [0.0319719],
                [0.09165278],
                [0.08136297]
            ]
        )
        stat_pol = 31.0400

    else:
        raise NameError(
            "{0} is not a valid file name".format(xyz_file_name)
        )

    return (
        pl_freq,
        ex_freq,
        pl_str,
        ex_lftm,
        stat_pol
    )


def spatial_dist(coordinates):
    """
    Computes the spatial distance between each atom as well as the spatial
    distance from the origin, (0, 0, 0), to each atom.

    Parameters
    ----------
    coordinates : Array containing the coordinates of the atoms.

    Returns
    -------
    p_dist : Spatial distance between each atom.

    o_dist : Spatial distance from the origin, (0, 0, 0), to each atom.
    """
    return (
        np.linalg.norm(coordinates - coordinates[:, None], axis=-1),
        np.linalg.norm(np.array([0, 0, 0]) - coordinates[:, None], axis=-1)
    )


def point_diff(coordinates):
    """
    Computes the difference between each x-coordinate, each y-coordinate, and
    each z-coordinate, respectively.

    Parameters
    ----------
    coordinates : array containing the coordinates for the molecule and the
                  nanoparticle {FROM: fread.xyz()}.

    Returns
    -------
    x_diff : Difference between each of the x-coordinates.

    y_diff : Difference between each of the y-coordinates.

    z_diff : Difference between each of the z-coordinates.
    """
    return (
        (coordinates - coordinates[:, None])[..., 0],
        (coordinates - coordinates[:, None])[..., 1],
        (coordinates - coordinates[:, None])[..., 2]
    )


def T(x_diff, y_diff, z_diff, p_dist):
    """
    Computes all the elements of the the real dipole-dipole interaction tensors.

    Parameters
    ----------
    x_diff : Difference between all the x-coordinates.

    y_diff : Difference between all the y-coordinates.

    z_diff : Difference between all the z-coordinates.

    p_dist : Spatial distance between each atom.

    Returns
    -------
    T_xx : Array containing all the xx components.

    T_yy : Array containing all the yy components.

    T_zz : Array containing all the zz components.

    T_xy : Array containing all the xy components.

    T_xz : Array containing all the xz components.

    T_yz : Array containing all the yz components.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        T_xx = ((3 * x_diff**2) / p_dist**5) - (1 / p_dist**3)
        T_yy = ((3 * y_diff**2) / p_dist**5) - (1 / p_dist**3)
        T_zz = ((3 * z_diff**2) / p_dist**5) - (1 / p_dist**3)
        T_xy = ((3 * x_diff * y_diff) / p_dist**5)
        T_xz = ((3 * x_diff * z_diff) / p_dist**5)
        T_yz = ((3 * z_diff * y_diff) / p_dist**5)

    T_xx[np.isnan(T_xx)] = 0
    T_yy[np.isnan(T_yy)] = 0
    T_zz[np.isnan(T_zz)] = 0
    T_xy[np.isnan(T_xy)] = 0
    T_xz[np.isnan(T_xz)] = 0
    T_yz[np.isnan(T_yz)] = 0

    return (
        T_xx,
        T_yy,
        T_zz,
        T_xy,
        T_xz,
        T_yz
    )


def tensor_stack(arrays):
    """
    Reshapes a touple of arrays in accordance with the shape of the A matrix.

    Parameters
    ----------
    arrays : Touple of arrays.

    Returns
    -------
    array : Total array with the proper shape.
    """
    arrays = np.asarray(arrays)
    n, p, q = arrays.shape
    s = int(round(np.sqrt(n)))
    arrays = arrays.reshape(s, -1, p, q)

    return arrays.transpose(2, 0, 3, 1).reshape(s * p, -1)


def tensor_gentrificator(tensor):
    """
    Converts the given array to a negative and complex array.

    Parameters
    ----------
    tensor : NumPy type array.

    Returns
    -------
    tensor : Negative and complex version of the input array.
    """
    tensor = np.negative(tensor)
    return tensor.astype(complex)


def nano_diag(aNP_xx, aNP_yy, aNP_zz, temp_A, i):
    """
    Inserts the proper polarizability values in the diagonal of the A matrix
    depending on the given frequency.

    Parameters
    ----------
    temp_A : Temporary version of the A matrix.

    Returns
    -------
    temp_A : Temporary version of the inverted A matrix with polarizabilites in
             the diagonal.
    """
    temp_aNP = np.vstack((aNP_xx, aNP_yy, aNP_zz))
    diag_len = int(len(temp_A[1, :]) / 3)
    arr = temp_aNP[:, i]
    diag_list = np.hstack([arr for element in range(diag_len)])
    diag_array = np.asarray(diag_list)
    np.fill_diagonal(temp_A, diag_array)
    return np.linalg.inv(temp_A)


def nano_pol(temp_A, pl_freq, ex_freq, pl_str, ex_lftm, stat_pol, freq, i):
    """
    Computes the complex polarizabilites of the nanoparticle, fills the diagonal
    of the A matrix, and inverts the A matrix, respectively.

    Parameters
    ----------
    temp_A : Temporary version of the A matrix.

    freq : Array containing the frequencies from the .out file.

    i : Counter variable used to run through the array containing the
        frequencies (see freq) index for index.

    pl_freq : Plasmon frequency.

    ex_freq : Average excitation energy.

    pl_str : Plasmon strength factor.

    ex_lftm : Excitation lifetime.

    stat_pol : The static polarizability of the given nanoparticle.

    Returns
    -------
    temp_A : Temporary version of the inverted A matrix with polarizabilites in
             the diagonal.

    aNP : Computed polarizability values.
    """
    aNP = (np.sum(
        stat_pol * ((pl_str * pl_freq**2) /
                    ((ex_freq**2 - freq[i]**2) - ((freq[i] * ex_lftm) * 1j)))
    ) / 5)
    np.fill_diagonal(temp_A, aNP)

    return (
        np.linalg.inv(temp_A),
        aNP
    )


def E(o_dist, E_external, dipole_x, dipole_y, dipole_z, coordinates, x_coordinates, y_coordinates, z_coordinates):
    """
    Computes the Cartesian components of the permanent electrical field, and
    combines them into an array stored under the variable name E.

    Parameters
    ----------
    E_external : array containing the Cartesian components of the external
                 electrical field.

    dipole_x : array containing the x components of the induced dipole
               moments for both the molecule and nanoparticle.

    dipole_y : array containing the y components of the induced dipole
               moments for both the molecule and nanoparticle.

    dipole_z : array containing the z components of the induced dipole
               moments for both the molecule and nanoparticle.

    o_dist : Spatial distance from the origin (0, 0, 0) to each atom.

    x_coordinates : array containing the coordinates from the given .xyz
                    file incl. an automatically created single-point molecule.

    y_coordinates : array containing the coordinates from the given .xyz
                    file incl. an automatically created single-point molecule.

    z_coordinates : array containing the coordinates from the given .xyz
                    file incl. an automatically created single-point molecule.

    Returns
    -------
    E : array containing the electromagnetic field vectors.
    """
    E_x = E_external[0] + (
        dipole_x * (1 / (o_dist**3) - 3 * ((x_coordinates**2) / (o_dist**5)))
        + dipole_y * ((- 3 * x_coordinates * y_coordinates) / o_dist**5)
        + dipole_z * (-(3 * x_coordinates * z_coordinates) / o_dist**5)
    )

    E_y = E_external[1] + (
        dipole_y * (1 / (o_dist**3) - 3 * ((y_coordinates**2) / (o_dist**5)))
        + dipole_x * ((- 3 * y_coordinates * x_coordinates) / o_dist**5)
        + dipole_z * (- (3 * y_coordinates * z_coordinates) / o_dist**5)
    )

    E_z = E_external[2] + (
        dipole_z * (1 / (o_dist**3) - 3 * ((z_coordinates**2) / (o_dist**5)))
        + dipole_y * ((- 3 * z_coordinates * y_coordinates) / o_dist**5)
        + dipole_x * (-(3 * y_coordinates * z_coordinates) / o_dist**5)
    )

    E = []
    for element in zip(E_x, E_y, E_z):
        E.extend(element)

    E = np.asarray(E)
    E = E.astype(complex)

    return E

# ==============================================================================
#   LEGACY FUNCTIONS
# ==============================================================================


def A(Txx, Tyy, Tzz):
    """
    ****************************************************************************
    OUTDATED -- ONLY FOR ISOTROPIC DIPOLE-DIPOLE INTERACTION TENSOR.
    NEW: tensor_stack()
    ****************************************************************************

    Constructs an array containing the T tensor matrices as the basis for
    the A matrix.

    Parameters
    ----------
    Txx : array containing all the x components of the T tensors.

    Tyy : array containing all the y components of the T tensors.

    Tzz : array containing all the z components of the T tensors.

    Returns
    -------
    A : array shaped based on the size and shape of the T tensor.
    """
    M = np.array((Txx, Tyy, Tzz))
    i, j, k = M.shape
    A = np.zeros((i * j, i * k), M.dtype)
    np.einsum("jiki -> ijk", A.reshape(j, i, k, i))[...] = M

    A = A.astype(complex)

    A = np.negative(A)

    return A
