# zacharias liasi's discrete interaction model computations for a nanoparticle in the presence of a molecule
# ku mail: flx527@alumni.ku.dk

# july 2019

# single comments above code explains the directly following line(s)
## double comment following code gives extra information about the use of the above code

# see the following github rep. for any sources and files used in the below code: https://github.com/ZLiasi/ForskPrak

#===============================================================================
#   IMPORT MODULES
#===============================================================================

import sys # used for np.set_printoption() to expand the threshold for np.array size
import numpy as np # used for array objects, math.operations over arrays, linalg
import os # used for reading and printing the memory used when running the program
import psutil # -||-
import time # used to time the program
import os.path

#===============================================================================
#   PREAMPLE
#===============================================================================

# increases the max display size for the printet array in order to avoid truncation of the array

#np.set_printoptions(threshold=sys.maxsize)

# initialise the timer

start_time = time.time()

#===============================================================================
#   READ THE .xyz AND .out FILE
#===============================================================================

# create empty lists (in order to be able to append the wanted information to the corresponding list)

atom_symbols = []
NP_coordinates = []
rx = []
ry =[]
rz = []

# open the given .xyz file and read the necessary data

xyz = open("/home/liasi/Documents/research_practice/au_particle.xyz")
n_atoms = int(xyz.readline())
title = xyz.readline()
for line in xyz:
    atom, x, y, z = line.split()
    NP_coordinates.append([float(x), float(y), float(z)]) # create a list of the coordinates
    atom_symbols.append([str(atom)]) # create a list of the chemical symbols for the particles
    rx.append([float(x)]) # create a set of lists of the x, y, and z coordinates, respectively, for the nanoparticle
    ry.append([float(y)]) # -||-
    rz.append([float(z)]) # -||-
xyz.close() # close the .xyz file

# create np.arrays from the list(s)

NP_coordinates = np.asarray(NP_coordinates) # array containing the coordinates for the given nanoparticle

atom_symbols = np.asarray(atom_symbols) # array containing the chemical symbol for each particle

rx = np.asarray(rx) # array containing the x, y, and z coordinates, respectively, for the given nanoparticle
ry = np.asarray(ry) # -||-
rz = np.asarray(rz) # -||-

M_coordinates = np.array([3.25, 0, 0]) # coordinates for the singlepoint molecule

coordinates = np.vstack((M_coordinates, NP_coordinates)) # combined array of the coordinates for the NP and for the singlepoint molecule

rx = np.vstack((M_coordinates[0], rx)) # combined array of the x, y, and z coordinates, respectively, for the nanoparticle and the singlepoint molecule
ry = np.vstack((M_coordinates[1], ry)) # -||-
rz = np.vstack((M_coordinates[2], rz)) # -||-

# open the given .out file and read the necessary data (frequencies)

w = []
aMXXI = []
aMYYI = []
aMZZI = []
aMXXR = []
aMYYR = []
aMZZR = []

for root, dirs, files in os.walk("/home/liasi/Documents/research_practice/imagpol"):
    for file in files:
        if file.endswith('DHA'+'.out'):
            with open(os.path.join(root, file), 'r') as F:
                for line in F:
                    if 'XDIPLEN   XDIPLEN' in line:
                        XX = line.strip().split()
                        aMXXR.append(float(XX[4]))
                        aMXXI.append(float(XX[5]))
                        w.append(float(XX[3]))
                    if 'YDIPLEN   YDIPLEN' in line:
                        YY = line.strip().split()
                        aMYYR.append(float(YY[4]))
                        aMYYI.append(float(YY[5]))
                    if 'ZDIPLEN   ZDIPLEN' in line:
                        ZZ = line.strip().split()
                        aMZZR.append(float(ZZ[4]))
                        aMZZI.append(float(ZZ[5]))

# create an array from the list

w = np.asarray(w)

aMXX = []
aMYY = []
aMZZ = []
for i in range(len(aMXXI)):
    aMXX.append(complex(float(aMXXR[i]), float(aMXXI[i])))
    aMYY.append(complex(float(aMYYR[i]), float(aMYYI[i])))
    aMZZ.append(complex(float(aMZZR[i]), float(aMZZI[i])))

aMxx = np.asarray(aMXX)
aMyy = np.asarray(aMYY)
aMzz = np.asarray(aMZZ)

#===============================================================================
#  CALCULATE THE SPATIAL DISTANCES
#===============================================================================

# using np.linalg.norm combined with broadcasting for the calculations

R = np.linalg.norm(coordinates - coordinates[:, None], axis = -1)

## i[:, None] insert a new axis into i, i - i[:, None] will then do a row by row subtraction due to broadcasting. np.linalg.norm calculates the np.sqrt(np.sum(np.square(...))) over the last axis

RO = np.linalg.norm(np.array([0, 0, 0]) - coordinates[:, None], axis = -1) # spatial distance from origin

#===============================================================================
#  CALCULATE THE DIFFERENCE BETWEEN COORDINATES
#===============================================================================

# calculate the difference (x2 - x1) using broadcasting

dx = (coordinates - coordinates[:, None])[..., 0]
dy = (coordinates - coordinates[:, None])[..., 1]
dz = (coordinates - coordinates[:, None])[..., 2]

#===============================================================================
#  CREATE THE T TENSOR COMPONENTS
#===============================================================================

# with enviorment for overwriting the [non-crucial] error message for dividing by zero

with np.errstate(divide = 'ignore', invalid = 'ignore'):
    Txx = ( ( 3 * ( dx )**2 ) / R**5 ) - ( 1 / R**3 ) # eq. from "zdim_calculations"
    Tyy = ( ( 3 * ( dy )**2 ) / R**5 ) - ( 1 / R**3 ) # -||-
    Tzz = ( ( 3 * ( dz )**2 ) / R**5 ) - ( 1 / R**3 ) # -||-
pass

# change nan to 0 in order to avoid [non-crucial] runtime warning

Txx[np.isnan(Txx)] = 0
Tyy[np.isnan(Tyy)] = 0 # -||-
Tzz[np.isnan(Tzz)] = 0 # -||-

#===============================================================================
#  CREATE THE A MATRIX
#===============================================================================

# create the A matrix using einstein summation

M = np.array((Txx, Tyy, Tzz)) # combine the three arrays containing the x, y, and z components (Txx, Tyy, Tzz)
M = np.negative(M) # the T tensors has to be negative (see theory)
i, j, k = M.shape # isolate the 3x3 diagonal matrices into axes 1, 3
A = np.zeros((i*j, i*k),M.dtype) # create the A matrix as a np.zeors array based on the parameters of M
np.einsum("jiki -> ijk", A.reshape(j, i, k, i))[...] = M # create a writable view and assign M to it

## the np.einsum function maps the axes of the reshaped A to those of M; "jiki -> ijk" means that axis 0 ("j") maps to axis 1, axes 1 and 3 ("i") map to axis 0, and axis 2 ("k") maps to axis 2. Mapping two axes to one (as with "i") has the special meaning of taking the diagonal.

# change all the elements in the A matrix to complex numbers

A = A.astype(complex)

# initialising variables

wp = 9.03
wn = np.array([[0.415], [0.830], [2.969], [4.304], [13.32]])
fn = np.array([[0.024], [0.010], [0.071], [0.601], [4.384]])
Tn = np.array([[0.241], [0.345], [0.870], [2.494], [2.214]])

# create the polarizability constants for the nanoparticles (aplha NP)

g = -1

while g < 20:

    g += 1
    print(g+1)


    aNP = 40 * ( ( fn * wp**2 ) / ( ( wn**2 - w[g]**2 ) - (( w[g] * Tn ) * 1j) ) )

    aNP = np.sum(aNP)

    np.fill_diagonal(A, aNP) # fill the diagonal in A with aNP

    A[0, 0] = aMxx[g] # insert the x, y, and z components for the polarizability of the molecule
    A[1, 1] = aMyy[g] # -||-
    A[2, 2] = aMzz[g] # -||-

    # invert the A matrix

    A = np.linalg.inv(A)

    #===============================================================================
    #  CALCULATE THE INDUCED DIPOLE MOMENTS
    #===============================================================================

    # initialise the induced dipole array

    dipole = np.ones((len(A), 1))

    # create initial arrays for the Cartesian components of the dipole moment

    init_x = np.full((len(coordinates), 1), 1)
    init_y = np.full((len(coordinates), 1), 1)
    init_z = np.full((len(coordinates), 1), 1)

    # set the dipole elements to equal the inital arrays

    dipole_x = init_x
    dipole_y = init_y
    dipole_z = init_z

    # initialise a variable to count the number of iterations

    counter = 0

    # initialise the conditions for the while loop

    while (np.all(init_x) > np.all(dipole_x) * 0.80 and np.all(init_x) < np.all(dipole_x) * 1.20) and (np.all(init_y) > np.all(dipole_y) * 0.80 and np.all(init_y) < np.all(dipole_y) * 1.20) and (np.all(init_z) > np.all(dipole_z) * 0.80 and np.all(init_z) < np.all(dipole_z) * 1.20):

        # add one to the counter per iteration in order to count the number of iterations

        counter += 1

        # compute the Ex, Ey, and Ez arrays

        Ex = 5 + (dipole_x * ( 1 / ( RO**3 ) - 3 * ( ( rx**2 ) / ( RO**5 ) ) ) + dipole_y * ( ( - 3 * rx * ry ) / RO**5 ) + dipole_z * ( - ( 3 * rx * rz ) / RO**5 ))
        Ey = 0 + (dipole_y * ( 1 / ( RO**3 ) - 3 * ( ( ry**2 ) / ( RO**5 ) ) ) + dipole_x * ( ( - 3 * ry * rx ) / RO**5 ) + dipole_z * ( - ( 3 * ry * rz ) / RO**5 ))
        Ez = 0 + (dipole_z * ( 1 / ( RO**3 ) - 3 * ( ( rz**2 ) / ( RO**5 ) ) ) + dipole_y * ( ( - 3 * rz * ry ) / RO**5 ) + dipole_x * ( - ( 3 * ry * rz ) / RO**5 ))

        # combine the x, y, and z elements of the electric field into one array

        E = [] # create and empty list
        for element in zip(Ex, Ey, Ez): # insert the elements from Ex, Ey, and Ez into the list in element wise order (i.e. Ex1, Ey1, Ez1, Ex2, Ey2, Ez2, etc.)
            E.extend(element)

        E = np.asarray(E) # create an np.array from the list
        E = E.astype(complex)

        # create the dipole matrix

        dipole = np.dot(A, E)

        dipole_x = dipole[0:len(dipole):3]
        dipole_y = dipole[1:len(dipole):3]
        dipole_z = dipole[2:len(dipole):3]

        if counter == 1000:
            print(dipole[0:3])
            print()
            print(np.sum(dipole[3:len(dipole):3]))
            print()
            print(np.sum(dipole[4:len(dipole):3]))
            print()
            print(np.sum(dipole[5:len(dipole):3]))
            print()
            print("----------------------------------------")
            print("Number of iterations:", counter) # print the number of iterations used to calculate the induced dipole moment matrix
            print("----------------------------------------")
            break

    else:
        # print()
        # print("Induced dipole moments for", str(len(coordinates)), "atoms:") # print some text for cosmetics
        # print()
        # print(dipole) # print the induced dipole matrix
        print("working")
        print(dipole[0:3])
        print()
        print(np.sum(dipole[3:len(dipole):3]))
        print()
        print(np.sum(dipole[4:len(dipole):3]))
        print()
        print(np.sum(dipole[5:len(dipole):3]))
        print()
        # print("----------------------------------------")
        # print("Number of iterations:", counter) # print the number of iterations used to calculate the induced dipole moment matrix
        # print("----------------------------------------")

#===============================================================================
#  MISCELLANEOUS PROCESS INFORMATION
#===============================================================================

# time printout

print()
print("----------------------------------------")
print("Process time: %s seconds" % (time.time() - start_time))

# memory usage printout

process = psutil.Process(os.getpid())
print()
print("Process memory usage =" ,process.memory_info().rss*(9.31*10**(-10)), "GB")  # in bytes
print("----------------------------------------")
