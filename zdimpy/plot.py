import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


def MOL_dipole_append(dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z, dipole):
    """
    Appends the results in the empty lists initiated before the while loop, to
    be used in the dipole_plot() function.

    Parameters
    ----------
    dipole : The np.dot() product of A (see A()) and E (see E()).

    dipM_X : array containing the x component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    dipM_Y : array containing the y component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    dipM_Z : array containing the z component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    absM_X : array containing the x component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

    absM_Y : array containing the y component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

    absM_Z : array containing the z component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

    Returns
    -------
    A modified version of the lists dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y,
    absM_Z.
    """
    return (
        dipM_X.append(np.real(dipole[0, 0])),
        dipM_Y.append(np.real(dipole[1, 0])),
        dipM_Z.append(np.real(dipole[2, 0])),
        absM_X.append(np.imag(dipole[0, 0])),
        absM_Z.append(np.imag(dipole[2, 0])),
        absM_Y.append(np.imag(dipole[1, 0]))
    )


def NP_dipole_append(NP_dip_x, NP_dip_y, NP_dip_z, NP_abs_x, NP_abs_y, NP_abs_z, dipole):
    """
    To be added...
    """
    return (
        NP_dip_x.append(np.real(np.sum(dipole[0:len(dipole):3]))),
        NP_dip_y.append(np.real(np.sum(dipole[1:len(dipole):3]))),
        NP_dip_z.append(np.real(np.sum(dipole[2:len(dipole):3]))),
        NP_abs_x.append(np.imag(np.sum(dipole[0:len(dipole):3]))),
        NP_abs_y.append(np.imag(np.sum(dipole[1:len(dipole):3]))),
        NP_abs_z.append(np.imag(np.sum(dipole[2:len(dipole):3])))
    )


def dipole(freq, dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z, xyz_filename,
           coordinates, output_path, out_filename):
    """
    Plots the complex dipoles yielded by the np.dot() product of A and E.

    Parameters
    ----------
    W : array containing the frequencies from the .out file.

    dipM_X : array containing the x component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    dipM_Y : array containing the y component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    dipM_Z : array containing the z component of the dipole values for
             the molecule, i.e. the real values from the array dipole.

    absM_X : array containing the x component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

    absM_Y : array containing the y component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

    absM_Z : array containing the z component of the absorbtion values for
             the molecule, i.e. the imaginary values from the array dipole.

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

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_filename.split('.', 1)[0],
                                        xyz_filename.split('.', 1)[0],
                                        len(coordinates)-1), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_{2}.png".format(
            output_path, xyz_filename.split('.', 1)[0], i)
    ):
        i += 1

    plot_filename = "{0}{1}_{2}.png".format(
        output_path, xyz_filename.split('.', 1)[0], i)

    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close("all")


def NP_dipole(freq, NP_dip_x, NP_dip_y, NP_dip_z, NP_abs_x, NP_abs_y, NP_abs_z, xyz_file_name, coordinates, output_path, out_file_name):
    """
    To be added...
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(freq, NP_dip_x, 'b', label='Real')
    ax1.plot(freq, NP_abs_x, 'r', label='Imaginary')
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax2.plot(freq, NP_dip_y, 'b')
    ax2.plot(freq, NP_abs_y, 'r')
    ax2.set_title("Y component", fontsize=10)
    ax2.set_ylabel(r"Induced dipole moment [a.u.]", fontsize=15)

    ax3.plot(freq, NP_dip_z, 'b')
    ax3.plot(freq, NP_abs_z, 'r')
    ax3.set_title("Z component", fontsize=10)
    ax3.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_file_name.split('.', 1)[0],
                                        xyz_file_name.split('.', 1)[0],
                                        len(coordinates)), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_NPdip_{2}.png".format(
            output_path, xyz_file_name.split('.', 1)[0], i)
    ):
        i += 1

    plot_file_name = "{0}{1}_NPdip_{2}.png".format(
        output_path, xyz_file_name.split('.', 1)[0], i)

    plt.savefig(plot_file_name, bbox_inches='tight')
    plt.close("all")


def avg_NP_dipole(freq, NP_dip_x, NP_dip_y, NP_dip_z, NP_abs_x, NP_abs_y, NP_abs_z, out_file_name, xyz_file_name, output_path, coordinates):
    avgr = (NP_dip_x + NP_dip_y + NP_dip_z)/3
    avgi = (NP_abs_x + NP_abs_y + NP_abs_z)/3

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(freq, avgr, 'b', label='Real')
    ax1.legend(loc=0, fontsize=10)

    ax2.plot(freq, avgi, 'r', label='Imaginary')
    ax2.legend(loc=0, fontsize=10)

    ax1.set_ylabel(r"Avg. induced dipole moment [a.u.]", fontsize=15)
    ax2.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_file_name.split('.', 1)[0],
                                        xyz_file_name.split('.', 1)[0],
                                        len(coordinates)), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_avgNPdip_{2}.png".format(
            output_path, xyz_file_name.split('.', 1)[0], i)
    ):
        i += 1

    plot_filename = "{0}{1}_avgNPdip_{2}.png".format(
        output_path, xyz_file_name.split('.', 1)[0], i)

    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close("all")


def NP_pol(freq, aNP_xx, aNP_yy, aNP_zz, out_file_name, xyz_file_name, output_path, coordinates):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(freq, np.real(aNP_xx), 'b', label='Real')
    ax1.plot(freq, np.imag(aNP_xx), 'r', label='Imaginary')
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax2.plot(freq, np.real(aNP_yy), 'b')
    ax2.plot(freq, np.imag(aNP_yy), 'r')
    ax2.set_title("Y component", fontsize=10)
    ax2.set_ylabel(r"Polarizability [a.u.]", fontsize=15)

    ax3.plot(freq, np.real(aNP_zz), 'b')
    ax3.plot(freq, np.imag(aNP_zz), 'r')
    ax3.set_title("Z component", fontsize=10)
    ax3.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_file_name.split('.', 1)[0],
                                        xyz_file_name.split('.', 1)[0],
                                        len(coordinates)), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_pol_{2}.png".format(
            output_path, xyz_file_name.split('.', 1)[0], i)
    ):
        i += 1

    plot_file_name = "{0}{1}_pol_{2}.png".format(
        output_path, xyz_file_name.split('.', 1)[0], i)

    plt.savefig(plot_file_name, bbox_inches='tight')
    plt.close("all")


def avg_NP_pol(aNP_xx, aNP_yy, aNP_zz, freq, out_file_name, xyz_file_name,
               output_path, coordinates):
    """
    TO BE ADDED...
    """
    avgr = (np.real(aNP_xx) + np.real(aNP_yy) + np.real(aNP_zz))/3
    avgi = (np.imag(aNP_xx) + np.imag(aNP_yy) + np.imag(aNP_zz))/3

    fig, (ax1) = plt.subplots(
        1, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(freq, avgr, 'b', label='Real')
    ax1.plot(freq, avgi, 'r', label='Imaginary')
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax1.set_ylabel(r"Avg. polarizability [a.u.]", fontsize=15)
    ax1.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_file_name.split('.', 1)[0],
                                        xyz_file_name.split('.', 1)[0],
                                        len(coordinates)), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_avgpol_{2}.png".format(
            output_path, xyz_file_name.split('.', 1)[0], i)
    ):
        i += 1

    plot_file_name = "{0}{1}_avgpol_{2}.png".format(
        output_path, xyz_file_name.split('.', 1)[0], i)

    plt.savefig(plot_file_name, bbox_inches='tight')
    plt.close("all")


def NP_polarizability(aNP_list, freq, out_file_name,
                      xyz_file_name, output_path, coordinates):
    """
    TO BE ADDED...
    """
    fig, (ax1) = plt.subplots(
        1, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(freq, np.real(aNP_list), 'b', label='Real')
    ax1.plot(freq, np.imag(aNP_list), 'r', label='Imaginary')
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax1.set_ylabel(r"Avg. polarizability [a.u.]", fontsize=15)
    ax1.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)

    def g(w, pos):
        return "${}$".format(f._formatSciNotation("%1.10e" % w))

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    plt.suptitle(
        "{0} \n {1} ({2} atoms)".format(out_file_name.split('.', 1)[0],
                                        xyz_file_name.split('.', 1)[0],
                                        len(coordinates)), fontsize=12
    )

    i = 0
    while os.path.exists(
        "{0}{1}_Cpol_{2}.png".format(
            output_path, xyz_file_name.split('.', 1)[0], i)
    ):
        i += 1

    plot_file_name = "{0}{1}_Cpol_{2}.png".format(
        output_path, xyz_file_name.split('.', 1)[0], i)

    plt.savefig(plot_file_name, bbox_inches='tight')
    plt.close("all")
