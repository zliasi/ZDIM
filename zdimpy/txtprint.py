import numpy as np
import os
import psutil
import time


def initialise_file(xyz_filename, out_filename, output_path):
    """
    Initialises a .txt file at the specified location for print() output.

    Parameters
    ----------
    xyz_filename : String variable containing the filename of the .xyz file used
                   for the current run of the code.

    out_filename : String variable containing the filename of the .out file used
                   for the current run of the code.

    output_path : The absolute path of the directory in which the file will be
                  saved.

    Returns
    -------
    txt_output : Automatically generated filename for the .txt file containing
                 the print() output incl. the absolute path of the output file.
    """
    i = 0
    while os.path.exists("{0}{1}_zout_{2}.txt".format(
        output_path, xyz_filename.split('.', 1)[0], str(i)
    )):
        i += 1

    txt_output = "{0}{1}_zout_{2}.txt".format(
        output_path, xyz_filename.split('.', 1)[0], str(i)
    )

    with open(txt_output, "w") as txt_file:
        print("", file=txt_file)
        print("-" * 56, file=txt_file)
        print("zdim_v4", file=txt_file)
        print("Induced dipoles for {0} and {1}".format(
            out_filename, xyz_filename), file=txt_file
        )
        print("-" * 56, file=txt_file)
        print("", file=txt_file)

    return txt_output


def iteration(txt_output, W, g):
    """
    Appends the iteration number, as well as the frequency used for the given
    iteration of the calculations, to an output .txt file specified by the
    txt_output variable.

    Parameters
    ----------
    txt_output : Automatically generated filename for the .txt file containing
                 the print() output incl. the absolute path of the output file.

    W : An np.array() containing the frequencies from the .out file.

    g : Counter variable that is used to run through the np.array() containing
        the frequencies (W).

    Returns
    -------
    A nicely formatted print() output in the specified directory.
    """

    with open(txt_output, "a") as txt_file:
        print("-" * 56, file=txt_file)
        print("# {0} | Frequency = {1} a.u.".format(g+1, W[g]), file=txt_file)
        print("-" * 56, file=txt_file)


def dipole(txt_output, counter, dipole):
    """
    Prints the Cartesian components of both the NP's and molecule's induced
    dipole moments.

    Parameters
    ----------
    dipole : The np.dot() product of A (see A()) and E (see E()).

    txt_output : Automatically generated filename for the .txt file containing
                 the print() output incl. the absolute path of the output file.

    Returns
    -------
    A nicely formatted output in an automatically named .txt file in the
    directory given in the txt_init() function.
    """
    with open(txt_output, "a") as txt_file:
        print("Number of iterations: {0}".format(counter), file=txt_file)
        print("-" * 25, file=txt_file)
        print("", file=txt_file)
        print(
            "Molecule dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
                dipole[0, 0].real, "+-"[dipole[0, 0].imag < 0],
                abs(dipole[0, 0].imag)), file=txt_file
        )
        print(
            "Molecule dipole (y) = {0:.4e} {1} {2:.4e}i a.u.".format(
                dipole[1, 0].real, "+-"[dipole[1, 0].imag < 0],
                abs(dipole[1, 0].imag)), file=txt_file
        )
        print(
            "Molecule dipole (z) = {0:.4e} {1} {2:.4e}i a.u.".format(
                dipole[2, 0].real, "+-"[dipole[2, 0].imag < 0],
                abs(dipole[2, 0].imag)), file=txt_file
        )
        print("", file=txt_file)
        print(
            "Nanoparticle dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
                np.sum(dipole[3:len(dipole):3]).real, "+-"
                [np.sum(dipole[3:len(dipole):3]).imag < 0],
                abs(np.sum(dipole[3:len(dipole):3]).imag)),
            file=txt_file
        )
        print(
            "Nanoparticle dipole (y) = {0:.4e} {1} {2:.4e}i a.u.".format(
                np.sum(dipole[4:len(dipole):3]).real, "+-"
                [np.sum(dipole[4:len(dipole):3]).imag < 0],
                abs(np.sum(dipole[4:len(dipole):3]).imag)),
            file=txt_file
        )
        print(
            "Nanoparticle dipole (z) = {0:.4e} {1} {2:.4e}i a.u.".format(
                np.sum(dipole[5:len(dipole):3]).real, "+-"
                [np.sum(dipole[5:len(dipole):3]).imag < 0],
                abs(np.sum(dipole[5:len(dipole):3]).imag)),
            file=txt_file
        )
        print("", file=txt_file)


def min(txt_output, W, absM_X, absM_Y, absM_Z):
    """
    Prints the index of the minimum value for the absorbtion, i.e. the absM__
    variables.

    Parameters
    ----------
    txt_output : Automatically generated filename for the .txt file containing
                 the print() output incl. the absolute path of the output file.

    absM_X : np.array() containing the x component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Y : np.array() containing the y component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    absM_Z : np.array() containing the z component of the absorbtion values for
             the molecule, i.e. the imaginary values from the np.array() dipole.

    Returns
    -------
    A nicely formatted output in an automatically named .txt file in the
    directory given in the txt_init() function.
    """

    with open(txt_output, "a") as txt_file:
        print("=" * 56,
              file=txt_file)
        print("Imag. minimum is at: Frequency = {0} a.u.(#{1})".format(
            W[absM_X.index(np.min(absM_X))], absM_X.index(np.min(absM_X)
                                                          )), file=txt_file
              )
        print("", file=txt_file)
        print("Imag. minimum = {:.4e}i".format(np.min(absM_X)), file=txt_file)
        print("=" * 56, file=txt_file)
        print("", file=txt_file)


def misc(txt_output, START_TIME):
    """
    Prints miscellaneous information about the past run of the program.

    Parameters
    ----------
    txt_output : Automatically generated filename for the .txt file containing
                 the print() output incl. the absolute path of the output file.

    START_TIME: timer initialiser.

    Returns
    -------
    The elapsed time, and the used memory for the past run of the program.
    """
    process = psutil.Process(os.getpid())

    with open(txt_output, "a") as txt_file:
        print("-" * 56, file=txt_file)
        print("Process time = {:.4f} seconds".format(
            (time.time() - START_TIME)), file=txt_file
        )
        print("", file=txt_file)

        print("Process memory usage = {:.4f} GB".format(
            process.memory_info().rss * (9.31*10**(-10))), file=txt_file
        )
        print("", file=txt_file)
        print("CPU usage = {0}%".format(psutil.cpu_percent()), file=txt_file)
        print("-" * 56, file=txt_file)
        print("", file=txt_file)
