import numpy as np
import os
import psutil
import time


def header(freq, xyz_file_name, out_file_name, coordinates, aNP_xx, aNP_yy, aNP_zz):
    """
    TO BE ADDED...
    """
    freq_eV = np.array(freq * 27.211396)
    freq_cm = np.array(freq * 219474.6305)
    print("\n" + "="*80 + "\n")
    print("{:^80}".format("ZDIM") + "\n")
    print("{:^80}".format("Computations of response properties") + "\n")
    print("="*80 + "\n"*2)
    print("Input files:")
    print("    .xyz file: {}".format(xyz_file_name))
    print("    .out file: {}".format(out_file_name))
    print("\nTotal no. of atoms: {}".format(len(coordinates)) + "\n"*2)
    print("="*56)
    print("{:^56}".format("FREQUENCIES"))
    print("="*56 + "\n")
    print("-"*56)
    print("       " + " (a.u.) "+"\t    ", " (eV) "+"\t   ", "(cm-1) ")
    print("-"*56)
    print("       ", np.array_str(np.c_[freq, freq_eV, freq_cm], precision=3).replace(" ", "     ").replace(
        '[', '   ').replace(']', '').strip())
    print("-"*56)
    print("Total number of frequencies:", len(freq))
    print("-"*56 + "\n")
    print("*Conversion: 1 a.u. = 27.211396 eV = 219474.6305 cm-1" + "\n"*2)
    print("="*69)
    print("{:^69}".format("POLARIZABILITIES (a.u.)"))
    print("="*69 + "\n")
    print("-"*69)
    print("   " + "XX "+"\t"*3, " YY "+"\t"*3, "ZZ")
    print("-"*69)
    print(
        "   " + np.array_str(np.c_[aNP_xx, aNP_yy, aNP_zz], precision=4).replace(
            '[', '').replace(']', '').replace('j', 'i').strip()
    )
    print("\n"*2 + "="*56)
    print("{:^56}".format("INDUCED DIPOLE MOMENTS"))
    print("="*56 + "\n"*2)


def iteration(freq, i):
    """
    Prints the iteration number, as well as the frequency used for the given
    iteration of calculations, to the currently working terminal.

    Parameters
    ----------
    freq : array containing the frequencies from the .out file.

    i : Counter variable that is used to run through the array containing
        the frequencies (W).

    Returns
    -------
    A nicely formatted print() output in the currently working terminal.
    """
    print("-" * 56)
    print("# {0} | Frequency = {1} a.u.".format(i + 1, freq[i]))
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
    print(
        "\nMolecule dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
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
    print(
        "\nNanoparticle dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
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
        "Nanoparticle dipole (z) = {0:.4e} {1} {2:.4e}i a.u.\n".format(
            np.sum(dipole[5:len(dipole):3]).real, "+-"
            [np.sum(dipole[5:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[5:len(dipole):3]).imag))
    )


def NP_dipole(counter, dipole):
    """
    TO BE ADDED...
    """
    print("Number of iterations:", counter)
    print("-" * 25)
    print(
        "\nNanoparticle dipole (x) = {0:.4e} {1} {2:.4e}i a.u.".format(
            np.sum(dipole[0:len(dipole):3]).real, "+-"
            [np.sum(dipole[0:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[0:len(dipole):3]).imag))
    )
    print(
        "Nanoparticle dipole (y) = {0:.4e} {1} {2:.4e}i a.u.".format(
            np.sum(dipole[1:len(dipole):3]).real, "+-"
            [np.sum(dipole[1:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[1:len(dipole):3]).imag))
    )
    print(
        "Nanoparticle dipole (z) = {0:.4e} {1} {2:.4e}i a.u.\n".format(
            np.sum(dipole[2:len(dipole):3]).real, "+-"
            [np.sum(dipole[2:len(dipole):3]).imag < 0],
            abs(np.sum(dipole[2:len(dipole):3]).imag))
    )


def min_max(freq, abs_x, abs_y, abs_z, dip_x, dip_y, dip_z):
    """
    Prints the index of the minimum value for the absorbtion, i.e. the absM__
    variables.

    Parameters
    ----------
    freq : An array containing the frequencies from the .out file.

    MOL_abs_x : array containing the x component of the absorbtion values for
                the molecule, i.e. the imaginary values from the array dipole.

    MOL_abs_y : array containing the y component of the absorbtion values for
                the molecule, i.e. the imaginary values from the array dipole.

    MOL_abs_z : array containing the z component of the absorbtion values for
                the molecule, i.e. the imaginary values from the array dipole.

    Returns
    -------
    A nicely formatted output in an automatically named .txt file in the
    directory given in the txt_init() function.
    """
    print("=" * 56)
    print()
    print(
        "Real maximum is at: Frequency = {0} a.u. (#{1})".format(
            freq[dip_x.index(np.max(dip_x))], dip_x.index(np.max(dip_x))+1)
    )
    print(
        "\nReal maximum = {:.4e} a.u.\n".format(np.max(dip_x))
    )
    print(
        "Imag. minimum is at: Frequency = {0} a.u. (#{1})".format(
            freq[abs_x.index(np.min(abs_x))], abs_x.index(np.min(abs_x))+1)
    )
    print(
        "\nImag. minimum = {:.4e}i a.u.\n".format(np.min(abs_x))
    )
    print("=" * 56, "\n")


def misc(start_time):
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

    print()
    print("=" * 80)
    print(
        "Process time =", "{:.4f} seconds\n".format(
            (time.time() - start_time))
    )

    print(
        "Process memory usage =", "{:.4f} GB\n".format(
            process.memory_info().rss * (9.31*10**(-10)))
    )
    print(
        "CPU usage = {0}%".format(psutil.cpu_percent())
    )
    print("=" * 80, "\n")
    print()
    print("{:^80}".format("**** END ****"))
