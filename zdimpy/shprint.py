import numpy as np
import os
import psutil
import time


def iteration(W, g):
    """
    Prints the iteration number, as well as the frequency used for the given
    iteration of calculations, to the currently working terminal.

    Parameters
    ----------
    W : np.array() containing the frequencies from the .out file.

    g : Counter variable that is used to run through the np.array() containing
        the frequencies (W).

    Returns
    -------
    A nicely formatted print() output in the currently working terminal.
    """
    print("-" * 56)
    print("# {0} | Frequency = {1} a.u.".format(g+1, W[g]))
    print("-" * 56)


def dipole(counter, dipole):
    """
    Prints the Cartesian components of both the NP's and molecule's induced
    dipole moment.

    Parameters
    ----------
    dipole : The np.dot() product of A (see A()) and E (see E()).

    Returns
    -------
    A nicely formatted output in the currently working terminal.
    """
    print("Number of iterations:", counter)
    print("-" * 25)
    print()
    print(
        "Molecule dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
            dipole[0, 0].real, "+-"[dipole[0, 0].imag < 0],
            abs(dipole[0, 0].imag))
    )
    print(
        "Molecule dipole (y) = {0:.4e} {1} {2:.4e}i a.u.".format(
            dipole[1, 0].real, "+-"[dipole[1, 0].imag < 0],
            abs(dipole[1, 0].imag))
    )
    print(
        "Molecule dipole (z) = {0:.4e} {1} {2:.4e}i a.u.".format(
            dipole[2, 0].real, "+-"[dipole[2, 0].imag < 0],
            abs(dipole[2, 0].imag))
    )
    print()
    print(
        "Nanoparticle dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
            np.sum(dipole[3:len(dipole):3]).real, "+-"
            [np.sum(dipole[3:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[3:len(dipole):3]).imag))
    )
    print(
        "Nanoparticle dipole (y) = {0:.4e} {1} {2:.4e}i a.u.".format(
            np.sum(dipole[4:len(dipole):3]).real, "+-"
            [np.sum(dipole[4:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[4:len(dipole):3]).imag))
    )
    print(
        "Nanoparticle dipole (z) = {0:.4e} {1} {2:.4e}i a.u.".format(
            np.sum(dipole[5:len(dipole):3]).real, "+-"
            [np.sum(dipole[5:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[5:len(dipole):3]).imag))
    )
    print()


def min(W, absM_X, absM_Y, absM_Z):
    """
    Prints the index of the minimum value for the absorbtion, i.e. the absM__
    variables.

    Parameters
    ----------
    W : An np.array() containing the frequencies from the .out file.

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
    print("=" * 56)
    print("Imag. minimum is at: Frequency = {0} a.u. (#{1})".format(
        W[absM_X.index(np.min(absM_X))], absM_X.index(np.min(absM_X)))
    )
    print()
    print("Imag. minimum = {:.4e}i".format(np.min(absM_X)))
    print("=" * 56)
    print()


def misc(START_TIME):
    """
    Prints miscellaneous information about the past run of the program.

    Parameters
    ----------
    START_TIME: timer initialiser.

    Returns
    -------
    The elapsed time, and the used memory for the past run of the program.
    """
    process = psutil.Process(os.getpid())

    print("-" * 56)
    print("Process time =", "{:.4f} seconds".format((time.time() - START_TIME)))
    print()
    print("Process memory usage =", "{:.4f} GB".format(process.memory_info().rss
                                                       * (9.31*10**(-10)))
          )
    print()
    print("CPU usage = {0}%".format(psutil.cpu_percent()))
    print("-" * 56)
    print()
