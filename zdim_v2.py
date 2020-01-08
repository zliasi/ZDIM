#==============================================================================#
#                                                                              |
# Induced dipole moment interaction computations for a molecule-NP system.     |
#                                                                              |
# Zacharias Liasi                                                              |
# KU mail: flx527@alumni.ku.dk                                                 |
#                                                                              |
# August/September 2019                                                        |
#                                                                              |
# Version 2                                                                    |
#                                                                              |
# GitHub repository: https://github.com/zliasi/zdim                            |
#                                                                              |
#==============================================================================#
#   IMPORT MODULES                                                             #
#==============================================================================#

import numpy as np  # Array objects, math.operations over arrays, linalg.
import os  # Used for reading and printing the amount of memory used.
import psutil  # -||-
import time  # Used to time the program.
import matplotlib.pyplot as plt  # Used to plot the results.
import matplotlib.ticker as mticker  # Used to add specific ticks to the plots.

#==============================================================================#
#   PARAMETERS                                                                 #
#==============================================================================#

# Prompt asking for the name of the .xyz file to be read.
print("Which .xyz file should be used? (exclude file extention): ")
xyz_filename = input()

# Prompt asking for the name of the .out file to be read.
print("Which .out file should be used? (exclude file extention): ")
out_filename = input()

# Prompt asking for whether or not the results should be plottet.
print("Do you wish to plot the results?")
plot_results = input("(y/n): ")

# Prompt asking for wether or not a LaTeX formattet output is wanted.
print("Do you wish to get a LaTeX formatted output?")
zdim2latex = input("(y/n): ")

# The values for the external electric field vector.
E_EXTERNAL = np.array([5, 0, 0])

#==============================================================================#
#   MISCELLANEOUS                                                              #
#==============================================================================#

# Initialise the start time for the timer.
START_TIME = time.time()

#==============================================================================#
#   READ THE .xyz AND .out FILES                                               #
#==============================================================================#

#===================================# .xyz #===================================#

# Initialise a set of lists to contain the data from the .xyz file.
NP_COORDINATES = []
X_COORDINATES = []
Y_COORDINATES = []
Z_COORDINATES = []

# Read the necessary data from the .xyz file with open.
with open("/home/liasi/Documents/zdim/" + xyz_filename + ".xyz") as xyz:
    n_atoms = xyz.readline()
    title = xyz.readline()
    for line in xyz:
        atom, x, y, z = line.split()
        NP_COORDINATES.append([float(x), float(y), float(z)])
        X_COORDINATES.append([float(x)])
        Y_COORDINATES.append([float(y)])
        Z_COORDINATES.append([float(z)])

# Create np.arrays from the lists
NP_COORDINATES = np.asarray(NP_COORDINATES)
X_COORDINATES = np.asarray(X_COORDINATES)
Y_COORDINATES = np.asarray(Y_COORDINATES)
Z_COORDINATES = np.asarray(Z_COORDINATES)

# Initialise the coordinates for the single-point molecule.
x = float(max(X_COORDINATES)) + 3.25
M_COORDINATES = np.array([x, 0, 0])

# Combine the molecule and NP coordinates by stacking them vertically.
COORDINATES = np.vstack((M_COORDINATES, NP_COORDINATES))
X_COORDINATES = np.vstack((M_COORDINATES[0], X_COORDINATES))
Y_COORDINATES = np.vstack((M_COORDINATES[1], Y_COORDINATES))
Z_COORDINATES = np.vstack((M_COORDINATES[2], Z_COORDINATES))

#===================================# .out #===================================#

# Initialise a set of lists to contain the data from the .out file.
W = []
aMXXI = []
aMYYI = []
aMZZI = []
aMXXR = []
aMYYR = []
aMZZR = []

# Read the necessary data from the .out file with open.
with open("/home/liasi/Documents/zdim/isopol/" + out_filename + ".out") as OUT:
    for line in OUT:
        if "XDIPLEN   XDIPLEN" in line:
            XX = line.strip().split()
            aMXXR.append(float(XX[4]))
            aMXXI.append(float(XX[5]))
            W.append(float(XX[3]))
        if "YDIPLEN   YDIPLEN" in line:
            YY = line.strip().split()
            aMYYR.append(float(YY[4]))
            aMYYI.append(float(YY[5]))
        if "ZDIPLEN   ZDIPLEN" in line:
            ZZ = line.strip().split()
            aMZZR.append(float(ZZ[4]))
            aMZZI.append(float(ZZ[5]))

# Initialise a set of empty lists to contain the complex polarizabilities.
aMXX = []
aMYY = []
aMZZ = []

# For loop to combine the real and imaginary part of the polarizabilities.
for i in range(len(aMXXI)):
    aMXX.append(complex(float(aMXXR[i]), float(aMXXI[i])))
    aMYY.append(complex(float(aMYYR[i]), float(aMYYI[i])))
    aMZZ.append(complex(float(aMZZR[i]), float(aMZZI[i])))

# Create np.arrays from the lists of the frequencies and polarizabilities.
W = np.asarray(W)
aMXX = np.asarray(aMXX)
aMYY = np.asarray(aMYY)
aMZZ = np.asarray(aMZZ)

#==============================================================================#
#  CALCULATE THE SPATIAL DISTANCES                                             #
#==============================================================================#

# Using np.linalg.norm combined with broadcasting for the calculations.

# i[:, None] inserts a new axis into i,
# i - i[:, None] will then do a row by row subtraction due to broadcasting.
# np.linalg.norm calculates np.sqrt(np.sum(np.square(...))) over the last axis.

# Spatial distance between the atoms.
R = np.linalg.norm(COORDINATES - COORDINATES[:, None], axis=-1)

# Spatial distance from the origin (0, 0, 0).
R0 = np.linalg.norm(np.array([0, 0, 0]) - COORDINATES[:, None], axis=-1)

# Replace any 0 in the array with a 1 as a 0 will cause problems in Ex, Ey, Ez.
R0[R0 == 0] = 1

#==============================================================================#
#  CALCULATE THE DIFFERENCE BETWEEN COORDINATES                                #
#==============================================================================#

# Calculate the difference (x2 - x1) using broadcasting.
DIFFERENCE_X = (COORDINATES - COORDINATES[:, None])[..., 0]
DIFFERENCE_Y = (COORDINATES - COORDINATES[:, None])[..., 1]
DIFFERENCE_Z = (COORDINATES - COORDINATES[:, None])[..., 2]

#==============================================================================#
#  CREATE THE T TENSOR COMPONENTS                                              #
#==============================================================================#

# The np.errstate function is used to ignore error message for dividing by zero.
with np.errstate(divide="ignore", invalid="ignore"):
    TXX = ((3 * DIFFERENCE_X**2) / R**5) - (1 / R**3)
    TYY = ((3 * DIFFERENCE_Y**2) / R**5) - (1 / R**3)
    TZZ = ((3 * DIFFERENCE_Z**2) / R**5) - (1 / R**3)
pass

# Change NaN to 0 in order to avoid [non-crucial] runtime warning.
TXX[np.isnan(TXX)] = 0
TYY[np.isnan(TYY)] = 0
TZZ[np.isnan(TZZ)] = 0

#==============================================================================#
#  CREATE THE A MATRIX                                                         #
#==============================================================================#

# The np.einsum function maps the axes of the reshaped A to those of M;
# "jiki -> ijk" means that axis 0 ("j") maps to axis 1,
# axes 1 and 3 ("i") map to axis 0,
# and axis 2 ("k") maps to axis 2.
# Mapping two axes to one (as with "i") takes the diagonal.

# Create the A matrix using einstein summation.
M = np.array((TXX, TYY, TZZ))
i, j, k = M.shape
A = np.zeros((i * j, i * k), M.dtype)
np.einsum("jiki -> ijk", A.reshape(j, i, k, i))[...] = M

# Change all the elements in the A matrix to complex numbers.
A = A.astype(complex)

# Make the T tensors negative.
A = np.negative(A)

# Initialise variables for each NP to calculate the polarizabilities.
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

# else statement uses au_x data.
else:
    Wp = 9.03
    We = np.array([[0.415], [0.830], [2.969], [4.304], [13.32]])
    Fp = np.array([[0.024], [0.010], [0.071], [0.601], [4.384]])
    Te = np.array([[0.241], [0.345], [0.870], [2.494], [2.214]])
    STAT_POL = 31.0400

# Initialise empty lists to contain the data for plotting.
dipM_X = []
dipM_Y = []
dipM_Z = []
absM_X = []
absM_Y = []
absM_Z = []

# Initialise a for loop using enumerate() to loop through for each frequency.
for g, freq in enumerate(W):

    # Print the value and number of the frequency used for the calculations.
    print("-" * 56)
    print("#", g+1, "| Frequency =", W[g], "a.u.")
    print("-" * 56)

    # Initialise the calculations of the polarizabilities for the NP.
    aNP = STAT_POL * ((Fp * Wp**2) / ((We**2 - W[g]**2) - ((W[g] * Te) * 1j)))

    aNP = np.sum(aNP)

    # Insert the complex NP polarizabilities in the A matrix.
    np.fill_diagonal(A, aNP)

    # Insert the complex molecule polarizabilities in the A matrix.
    A[0, 0] = aMXX[g]
    A[1, 1] = aMYY[g]
    A[2, 2] = aMZZ[g]

    # Invert the A matrix.
    A = np.linalg.inv(A)

    #==========================================================================#
    #  CALCULATE THE INDUCED DIPOLE MOMENTS                                    #
    #==========================================================================#

    # Set the initial values for the dipole moments.
    dipole_x = np.full((len(COORDINATES), 1), 1 + 0.j)
    dipole_y = np.full((len(COORDINATES), 1), 1 + 0.j)
    dipole_z = np.full((len(COORDINATES), 1), 1 + 0.j)

    # Initialise a variable to count the number of iterations.
    counter = 0

    # Initialise an infinite loop for the dipole calculations.
    while True:

        # Add 1 to the counter per iteration to track the number of iterations.
        counter += 1

        # Compute the Ex, Ey, and Ez arrays.
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

        # Combine the x, y, and z elements of the electric field into one array.
        E = []
        for element in zip(Ex, Ey, Ez):
            E.extend(element)

        E = np.asarray(E)
        E = E.astype(complex)

        # Create the dipole matrix.
        dipole = np.dot(A, E)

        # Initialise a set of values to check if the dipole values have changed.
        check_x = dipole[0:len(dipole):3]
        check_y = dipole[1:len(dipole):3]
        check_z = dipole[2:len(dipole):3]

        # Initialise an if statement that breaks the otherwise infinite loop.
        if (np.array_equal(dipole_x, check_x) == True
            and np.array_equal(dipole_y, check_y) == True
            and np.array_equal(dipole_z, check_z) == True
                or counter > 999):

            # Print the results for each frequency.
            print("Number of iterations:", counter)
            print("-" * 25)
            print()
            print(
                "Molecule dipole (x) =", "{0:.4e} {1} {2:.4e}i".format(
                    dipole[0, 0].real, "+-"[dipole[0, 0].imag < 0],
                    abs(dipole[0, 0].imag)), "a.u."
            )
            print(
                "Molecule dipole (y) =", "{0:.4e} {1} {2:.4e}i".format(
                    dipole[1, 0].real, "+-"[dipole[1, 0].imag < 0],
                    abs(dipole[1, 0].imag)), "a.u."
            )
            print(
                "Molecule dipole (z) =", "{0:.4e} {1} {2:.4e}i".format(
                    dipole[2, 0].real, "+-"[dipole[2, 0].imag < 0],
                    abs(dipole[2, 0].imag)), "a.u."
            )
            print()
            print(
                "Nanoparticle dipole (x) =", "{0:.4e} {1} {2:.4e}i".format(
                    np.sum(dipole[3:len(dipole):3]).real, "+-"
                    [np.sum(dipole[3:len(dipole):3]).imag < 0],
                    abs(np.sum(dipole[3:len(dipole):3]).imag)), "a.u."
            )
            print(
                "Nanoparticle dipole (y) =", "{0:.4e} {1} {2:.4e}i".format(
                    np.sum(dipole[4:len(dipole):3]).real, "+-"
                    [np.sum(dipole[4:len(dipole):3]).imag < 0],
                    abs(np.sum(dipole[4:len(dipole):3]).imag)), "a.u."
            )
            print(
                "Nanoparticle dipole (z) =", "{0:.4e} {1} {2:.4e}i".format(
                    np.sum(dipole[5:len(dipole):3]).real, "+-"
                    [np.sum(dipole[5:len(dipole):3]).imag < 0],
                    abs(np.sum(dipole[5:len(dipole):3]).imag)), "a.u."
            )
            print()

            # Save the results for each frequency to be plottet afterwards.
            dipM_X.append(np.real(dipole[0, 0]))
            dipM_Y.append(np.real(dipole[1, 0]))
            dipM_Z.append(np.real(dipole[2, 0]))
            absM_X.append(np.imag(dipole[0, 0]))
            absM_Z.append(np.imag(dipole[2, 0]))
            absM_Y.append(np.imag(dipole[1, 0]))
            break

        # Update the dipole values if and only if the if statement is False.
        dipole_x = dipole[0:len(dipole):3]
        dipole_y = dipole[1:len(dipole):3]
        dipole_z = dipole[2:len(dipole):3]

# Print the minimum value for the imaginary part of the polarizabilities.
print("=" * 56)
print("Imag. minimum is at:", "Frequency =", W[absM_X.index(np.min(absM_X))],
      "a.u.", "(", "#", absM_X.index(np.min(absM_X)), ")")
print()
print("Imag. minimum =", "{:.4e}i".format(np.min(absM_X)))
print("=" * 56)

#==============================================================================#
#  MISCELLANEOUS PROCESS INFORMATION                                           #
#==============================================================================#

# Time printout.
print()
print("-" * 56)
print("Process time =", "{:.4f}".format((time.time() - START_TIME)), "seconds")

# Memory usage printout.
process = psutil.Process(os.getpid())
print()
print("Process memory usage =", "{:.4f}".format(process.memory_info().rss
                                                * (9.31*10**(-10))), "GB")
print("-" * 56)

#==============================================================================#
#  CALL zdim2latex.py                                                          #
#==============================================================================#

# LaTeX formatted output.
if zdim2latex == "y":
    import zdim2latex as pltex

    pltex.latex_output(W, dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z)

#==============================================================================#
#  PLOT THE RESULTS                                                            #
#==============================================================================#

if plot_results == "y":

    # Plot settings.
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, sharey=False, figsize=(10, 6)
    )

    ax1.plot(W, dipM_X, "b", label="Real")
    ax1.plot(W, absM_X, "r", label="Imaginary")
    ax1.set_title("X component", fontsize=10)
    ax1.legend(loc=0, fontsize=10)

    ax2.plot(W, dipM_Y, "b")
    ax2.plot(W, absM_Y, "r")
    ax2.set_title("Y component", fontsize=10)
    ax2.set_ylabel(r"Induced dipole moment [a.u.]", fontsize=15)

    ax3.plot(W, dipM_Z, "b")
    ax3.plot(W, absM_Z, "r")
    ax3.set_title("Z component", fontsize=10)
    ax3.set_xlabel(r"External field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    def g(w, pos): return "${}$".format(f._formatSciNotation("%1.10e" % w))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    # if statement that outputs the plot title depending on the given NP.
    if "cu" in xyz_filename:
        plt.suptitle(
            "Cu nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
            + " " + "atoms)", fontsize=15
        )

    elif "ag" in xyz_filename:
        plt.suptitle(
            "Ag nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
            + " " + "atoms)", fontsize=15
        )

    # else statement uses au_x data.
    else:
        plt.suptitle(
            "Au nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
            + " " + "atoms)", fontsize=15
        )

    plt.show()
