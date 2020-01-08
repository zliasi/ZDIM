#==============================================================================#
#                                                                              |
# Induced dipole moment interaction computations for a molecule-NP system      |
#                                                                              |
# Zacharias Liasi                                                              |
# KU mail: flx527@alumni.ku.dk                                                 |
#                                                                              |
# September 2019                                                               |
#                                                                              |
# Version 3                                                                    |
#                                                                              |
# GitHub repository: https://github.com/zliasi/zdim                            |
#                                                                              |
#==============================================================================#
#   IMPORT MODULES                                                             #
#==============================================================================#

import numpy as np  # Array objects, math.operations over arrays, linalg
import os  # Used for reading and printing the amount of memory used
import psutil  # -||-
import time  # Used to time the program
import matplotlib.pyplot as plt  # Used to plot the results
import matplotlib.ticker as mticker  # Used to add ticks to the plots

#==============================================================================#
#   PARAMETERS                                                                 #
#==============================================================================#

# The .xyz file to be read
XYZ_FILE_NAME = "au_particle"

# The .out file to be read
OUT_FILE_NAME = "DHA_iso"

# The values for the external electric field vector
E_EXTERNAL = np.array([5, 0, 0])

# Plot the results of the computations ("y" for plot, "n" for no plot)
PLOT_RESULTS = "y"

# Print LaTeX plottable code from the results ("y" for plot, "n" for no plot)
zdim2latex = "n"

#==============================================================================#
#   MISCELLANEOUS                                                              #
#==============================================================================#

# Initialises the timer
START_TIME = time.time()

# Print name of program file and data files
print()
print("Currently running:", __file__)
print()
print("For:", XYZ_FILE_NAME + ".xyz", "and", OUT_FILE_NAME + ".out")
print()

#==============================================================================#
#   READ THE .xyz AND .out FILES                                               #
#==============================================================================#

#===================================# .xyz #===================================#

# Initialise a set of lists to contain the data from the .xyz file
NP_COORDINATES = []
X_COORDINATES = []
Y_COORDINATES = []
Z_COORDINATES = []

# Read the necessary data from the .xyz file with open
with open("/home/liasi/Documents/zdim/" + XYZ_FILE_NAME + ".xyz") as XYZ:
    n_atoms = XYZ.readline()
    title = XYZ.readline()
    for line in XYZ:
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

# Initialise the coordinates for the single-point molecule
M_COORDINATES = np.array([3.25, 0, 0])

# Combine the molecule and NP coordinates by stacking them vertically
COORDINATES = np.vstack((M_COORDINATES, NP_COORDINATES))
X_COORDINATES = np.vstack((M_COORDINATES[0], X_COORDINATES))
Y_COORDINATES = np.vstack((M_COORDINATES[1], Y_COORDINATES))
Z_COORDINATES = np.vstack((M_COORDINATES[2], Z_COORDINATES))

#===================================# .out #===================================#

# Initialise a set of lists to contain the data from the .out file
W = []
aMXXI = []
aMYYI = []
aMZZI = []
aMXXR = []
aMYYR = []
aMZZR = []

# Read the necessary data from the .out file with open
with open("/home/liasi/Documents/zdim/isopol/" + OUT_FILE_NAME + ".out") as OUT:
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

# Initialise a set of empty lists to contain the complex polarizabilities
aMXX = []
aMYY = []
aMZZ = []

# For loop to combine the real and imaginary part of the polarizabilities
for i in range(len(aMXXI)):
    aMXX.append(complex(float(aMXXR[i]), float(aMXXI[i])))
    aMYY.append(complex(float(aMYYR[i]), float(aMYYI[i])))
    aMZZ.append(complex(float(aMZZR[i]), float(aMZZI[i])))

# Create np.arrays from the lists of the frequencies and polarizabilities
W = np.asarray(W)
aMXX = np.asarray(aMXX)
aMYY = np.asarray(aMYY)
aMZZ = np.asarray(aMZZ)

#==============================================================================#
#  CALCULATE THE SPATIAL DISTANCES                                             #
#==============================================================================#

# Using np.linalg.norm combined with broadcasting for the calculations

# i[:, None] inserts a new axis into i,
# i - i[:, None] will then do a row by row subtraction due to broadcasting.
# np.linalg.norm calculates np.sqrt(np.sum(np.square(...))) over the last axis.

# Spatial distance between the atoms
R = np.linalg.norm(COORDINATES - COORDINATES[:, None], axis=-1)

# Spatial distance from the origin (0, 0, 0)
R0 = np.linalg.norm(np.array([0, 0, 0]) - COORDINATES[:, None], axis=-1)

# Replace any 0 in the array with a 1 as a 0 will cause problems in Ex, Ey, Ez
if "sphere" in XYZ_FILE_NAME:
    R0[R0 == 0] = 1

#==============================================================================#
#  CALCULATE THE DIFFERENCE BETWEEN COORDINATES                                #
#==============================================================================#

# i[:, None] inserts a new axis into i (just like np.newaxis).
# [..., 0] uses axis 0 and the ... sets the proper shape of the output array

# Calculate the difference (a2 - a1) using broadcasting
DIFFERENCE_XX = (X_COORDINATES - X_COORDINATES[:, None])[..., 0]
DIFFERENCE_XY = (X_COORDINATES - Y_COORDINATES[:, None])[..., 0]
DIFFERENCE_XZ = (X_COORDINATES - Z_COORDINATES[:, None])[..., 0]

DIFFERENCE_YX = (Y_COORDINATES - X_COORDINATES[:, None])[..., 0]
DIFFERENCE_YY = (Y_COORDINATES - Y_COORDINATES[:, None])[..., 0]
DIFFERENCE_YZ = (Y_COORDINATES - Z_COORDINATES[:, None])[..., 0]

DIFFERENCE_ZX = (Z_COORDINATES - X_COORDINATES[:, None])[..., 0]
DIFFERENCE_ZY = (Z_COORDINATES - Y_COORDINATES[:, None])[..., 0]
DIFFERENCE_ZZ = (Z_COORDINATES - Z_COORDINATES[:, None])[..., 0]

#==============================================================================#
#  CREATE THE T TENSOR COMPONENTS                                              #
#==============================================================================#

# The np.errstate function is used to ignore error message for dividing by zero
with np.errstate(divide="ignore", invalid="ignore"):
    TXX = ((3 * DIFFERENCE_XX**2) / R**5) - (1 / R**3)
    TXY = ((3 * DIFFERENCE_XY**2) / R**5) - (1 / R**3)
    TXZ = ((3 * DIFFERENCE_XZ**2) / R**5) - (1 / R**3)

    TYX = ((3 * DIFFERENCE_YX**2) / R**5) - (1 / R**3)
    TYY = ((3 * DIFFERENCE_YY**2) / R**5) - (1 / R**3)
    TYZ = ((3 * DIFFERENCE_YZ**2) / R**5) - (1 / R**3)

    TZX = ((3 * DIFFERENCE_ZX**2) / R**5) - (1 / R**3)
    TZY = ((3 * DIFFERENCE_ZY**2) / R**5) - (1 / R**3)
    TZZ = ((3 * DIFFERENCE_ZZ**2) / R**5) - (1 / R**3)
pass

# Change NaN to 0 in order to avoid [non-crucial] runtime warning
TXX[np.isnan(TXX)] = 0
TXY[np.isnan(TXY)] = 0
TXZ[np.isnan(TXZ)] = 0

TYX[np.isnan(TYX)] = 0
TYY[np.isnan(TYY)] = 0
TYZ[np.isnan(TYZ)] = 0

TZX[np.isnan(TZX)] = 0
TZY[np.isnan(TZY)] = 0
TZZ[np.isnan(TZZ)] = 0

#==============================================================================#
#  CREATE THE A MATRIX                                                         #
#==============================================================================#

# Define a function that combines a given set of arrays into one array


def combine_arrays(arrays):
    arrays = np.asarray(arrays)
    n, p, q = arrays.shape
    s = int(round(np.sqrt(n)))
    arrays = arrays.reshape(s, -1, p, q)
    return arrays.transpose(2, 0, 3, 1).reshape(s * p, -1)


# Initialise the matrix A using the combine_arrays function
A = combine_arrays([TXX, TXY, TXZ, TYX, TYY, TYZ, TZX, TYZ, TZZ])

# Change all the elements in the A matrix to complex numbers
A = A.astype(complex)

# Make the T tensors negative
A = np.negative(A)

# Initialise variables for each NP to calculate the polarizabilities
if "cu" in XYZ_FILE_NAME:
    Wp = 10.83
    We = np.array([[0.291], [2.957], [5.300], [11.18]])
    Fp = np.array([[0.061], [0.104], [0.723], [0.638]])
    Te = np.array([[0.378], [1.056], [3.213], [4.305]])
    STAT_POL = 33.7420

elif "ag" in XYZ_FILE_NAME:
    Wp = 9.01
    We = np.array([[0.816], [4.481], [8.185], [9.083], [20.29]])
    Fp = np.array([[0.065], [0.124], [0.011], [0.840], [5.646]])
    Te = np.array([[3.886], [0.452], [0.065], [0.916], [2.419]])
    STAT_POL = 49.9843

# else statement uses au_x data
else:
    Wp = 9.03
    We = np.array([[0.415], [0.830], [2.969], [4.304], [13.32]])
    Fp = np.array([[0.024], [0.010], [0.071], [0.601], [4.384]])
    Te = np.array([[0.241], [0.345], [0.870], [2.494], [2.214]])
    STAT_POL = 31.0400

# Initialise empty lists to contain the data for plotting
dipM_X = []
dipM_Y = []
dipM_Z = []
absM_X = []
absM_Y = []
absM_Z = []

# Initialise a counter that corresponds to the frequency index in the array W
g = -1

# Initialise a while loop to loop over the calculations for each frequency
while g < 20:

    # Increment g to proceed to the next frequency for each iteration
    g += 1

    # Print the value and number of the frequency used for the calculations
    print("--------------------------------------------------------")
    print("#", g+1, "| Frequency =", W[g], "a.u.")
    print("--------------------------------------------------------")

    # Initialise the calculations of the polarizabilities for the NP
    aNP = STAT_POL * ((Fp * Wp**2) / ((We**2 - W[g]**2) - ((W[g] * Te) * 1j)))

    aNP = np.sum(aNP)

    # Insert the complex NP polarizabilities in the A matrix
    np.fill_diagonal(A, aNP)

    # Insert the complex molecule polarizabilities in the A matrix
    A[0, 0] = aMXX[g]
    A[1, 1] = aMYY[g]
    A[2, 2] = aMZZ[g]

    # Invert the A matrix
    A = np.linalg.inv(A)

    #==========================================================================#
    #  CALCULATE THE INDUCED DIPOLE MOMENTS                                    #
    #==========================================================================#

    # Set the initial values for the dipole moments
    dipole_x = np.full((len(COORDINATES), 1), 1 + 0.j)
    dipole_y = np.full((len(COORDINATES), 1), 1 + 0.j)
    dipole_z = np.full((len(COORDINATES), 1), 1 + 0.j)

    # Initialise a variable to count the number of iterations
    counter = 0

    # Initialise an infinite loop for the dipole calculations
    while True:

        # Add one to the counter per iteration to track the number of iterations
        counter += 1

        # Compute the Ex, Ey, and Ez arrays
        Ex = (E_EXTERNAL[0] +
              (dipole_x * (1 / (R0**3) - 3 * ((X_COORDINATES**2) / (R0**5)))
               + dipole_y * ((- 3 * X_COORDINATES * Y_COORDINATES) / R0**5)
               + dipole_z * (-(3 * X_COORDINATES * Z_COORDINATES) / R0**5)))

        Ey = (E_EXTERNAL[1] +
              (dipole_y * (1 / (R0**3) - 3 * ((Y_COORDINATES**2) / (R0**5)))
               + dipole_x * ((- 3 * Y_COORDINATES * X_COORDINATES) / R0**5)
               + dipole_z * (- (3 * Y_COORDINATES * Z_COORDINATES) / R0**5)))

        Ez = (E_EXTERNAL[2] +
              (dipole_z * (1 / (R0**3) - 3 * ((Z_COORDINATES**2) / (R0**5)))
               + dipole_y * ((- 3 * Z_COORDINATES * Y_COORDINATES) / R0**5)
               + dipole_x * (-(3 * Y_COORDINATES * Z_COORDINATES) / R0**5)))

        # Combine the x, y, and z elements of the electric field into one array
        E = []
        for element in zip(Ex, Ey, Ez):
            E.extend(element)

        E = np.asarray(E)
        E = E.astype(complex)

        # Create the dipole matrix
        dipole = np.dot(A, E)

        # Initialise a set of values to check if the dipole values have changed
        check_x = dipole[0:len(dipole):3]
        check_y = dipole[1:len(dipole):3]
        check_z = dipole[2:len(dipole):3]

        # Initialise an if statement that breaks the otherwise infinite loop
        if (np.array_equal(dipole_x, check_x) == True
            and np.array_equal(dipole_y, check_y) == True
            and np.array_equal(dipole_z, check_z) == True
                or counter > 999):

            # Print the results for each frequency
            print("Number of iterations:", counter)
            print("-------------------------")
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

            # Save the results for each frequency to be plottet afterwards
            dipM_X.append(np.real(dipole[0, 0]))
            dipM_Y.append(np.real(dipole[1, 0]))
            dipM_Z.append(np.real(dipole[2, 0]))
            absM_X.append(np.imag(dipole[0, 0]))
            absM_Z.append(np.imag(dipole[2, 0]))
            absM_Y.append(np.imag(dipole[1, 0]))
            break

        # Update the dipole values if and only if the if statement is False
        dipole_x = dipole[0:len(dipole):3]
        dipole_y = dipole[1:len(dipole):3]
        dipole_z = dipole[2:len(dipole):3]

# Print the minimum value for the imaginary part of the polarizabilities
print("========================================================")
print("Imag. minimum is at:", "Frequency =", W[absM_X.index(np.min(absM_X))],
      "a.u.", "(", "#", absM_X.index(np.min(absM_X)), ")")
print()
print("Imag. minimum =", "{:.4e}i".format(np.min(absM_X)))
print("========================================================")

#==============================================================================#
#  MISCELLANEOUS PROCESS INFORMATION                                           #
#==============================================================================#

# Time printout
print()
print("--------------------------------------------------------")
print("Process time =", "{:.4f}".format((time.time() - START_TIME)), "seconds")

# Memory usage printout
process = psutil.Process(os.getpid())
print()
print("Process memory usage =", "{:.4f}".format(process.memory_info().rss
                                                * (9.31*10**(-10))), "GB")
print("--------------------------------------------------------")

# LaTeX formatted output
if zdim2latex == "y":
    import zdim2latex as pltex

    pltex.latex_output(W, dipM_X, dipM_Y, dipM_Z, absM_X, absM_Y, absM_Z)

#==============================================================================#
#  PLOT                                                                        #
#==============================================================================#

if PLOT_RESULTS == "y":

    # Plot settings
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False,
                                        figsize=(10, 6))

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
    ax3.set_xlabel(r"Field frequency [a.u.]", fontsize=15)

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    def g(w, pos): return "${}$".format(f._formatSciNotation("%1.10e" % w))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(g))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(g))

    # if statement that outputs the plot title depending on the given NP
    if "cu" in XYZ_FILE_NAME:
        plt.suptitle("Cu nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
                     + " " + "atoms)", fontsize=15)

    elif "ag" in XYZ_FILE_NAME:
        plt.suptitle("Ag nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
                     + " " + "atoms)", fontsize=15)

    # else statement uses au_x data
    else:
        plt.suptitle("Au nanoparticle" + " " + "(" + str(len(NP_COORDINATES))
                     + " " + "atoms)", fontsize=15)

    plt.show()
