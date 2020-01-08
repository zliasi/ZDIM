import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


def dipole_append(dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z, dipole):
    """
    Appends the results in the empty lists initiated before the while loop, to
    be used in the dipole_plot() function.

    Parameters
    ----------
    dipole : The np.dot() product of A (see A()) and E (see E()).

    dipM_X : np.array() containing the x component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    dipM_Y : np.array() containing the y component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    dipM_Z : np.array() containing the z component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    absM_X : np.array() containing the x component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Y : np.array() containing the y component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Z : np.array() containing the z component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    Returns
    -------
    A modified version of the lists dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y,
    absM_Z.
    """
    dipM_X.append(np.real(dipole[0, 0]))
    dipM_Y.append(np.real(dipole[1, 0]))
    dipM_Z.append(np.real(dipole[2, 0]))
    absM_X.append(np.imag(dipole[0, 0]))
    absM_Z.append(np.imag(dipole[2, 0]))
    absM_Y.append(np.imag(dipole[1, 0]))


def dipole(W, dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z, xyz_filename,
           COORDINATES, output_path, out_filename):
    """
    Plots the complex dipoles yielded by the np.dot() product of A and E.

    Parameters
    ----------
    W : np.array() containing the frequencies from the .out file.

    dipM_X : np.array() containing the x component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    dipM_Y : np.array() containing the y component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    dipM_Z : np.array() containing the z component of the dipole values for
             the molecule, i.e. the real values from the np.array() dipole.

    absM_X : np.array() containing the x component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Y : np.array() containing the y component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Z : np.array() containing the z component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    output_path : The absolute path of the directory in which the file will be
                  saved.

    Returns
    -------
    A .png file in the given path (see output_path variable).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(W, dipM_X, 'b', label='Real')
    ax1.plot(W, absM_X, 'r', label='Imaginary')
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax2.plot(W, dipM_Y, 'b')
    ax2.plot(W, absM_Y, 'r')
    ax2.set_title("Y component", fontsize=10)
    ax2.set_ylabel(r"Induced dipole moment [a.u.]", fontsize=15)

    ax3.plot(W, dipM_Z, 'b')
    ax3.plot(W, absM_Z, 'r')
    ax3.set_title("Z component", fontsize=10)
    ax3.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    if "cu" in xyz_filename:
        plt.suptitle(
            "{0} \n Cu nanoparticle ({1} atoms)".format(
                out_filename.split('.', 1)[0], len(COORDINATES)-1), fontsize=12
        )

    elif "ag" in xyz_filename:
        plt.suptitle(
            "{0} \n Ag nanoparticle ({1} atoms)".format(
                out_filename.split('.', 1)[0], len(COORDINATES)-1), fontsize=12
        )

    else:
        plt.suptitle(
            "{0} \n Au nanoparticle ({1} atoms)".format(
                out_filename.split('.', 1)[0], len(COORDINATES)-1), fontsize=12
        )

    i = 0
    while os.path.exists(
        "{0}{1}_zplot_{2}.png".format(
            output_path, xyz_filename.split('.', 1)[0], i)
    ):
        i += 1

    plot_filename = "{0}{1}_zplot_{2}.png".format(
        output_path, xyz_filename.split('.', 1)[0], i)

    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close("all")
