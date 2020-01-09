# ==============================================================================
#   MODULES
# ==============================================================================

import argparse
import os
import sys
import numpy as np
import time
import psutil
from zdimpy import (
    fread as f,
    calc,
    shprint as sh,
    txtprint as txt,
    plot
)

# ==============================================================================
#   ARGPARSE CONFIGURATIONS
# ==============================================================================


def convert_arg_line_to_args(arg_line):
    """
    Redefinition of the default arg.parse function used to read arguments from
    a file. More specifically, this redefines the way lines are read
    inorder to make multiple value arguments possible, e.g. -e, --external.
    """
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def file_path(path):
    """
    Custom "type" for argparse arguments which checks if the given file exists.

    Parameters
    ----------
    path : The given file path.

    Returns
    -------
    path : The given path is returned if valid.

    Raises
    ------
    argparse.ArgumentTypeError()
        If the given file path is invalid.
    """
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("{} does not exist".format(path))

    return path


class StoreAsArray(argparse._StoreAction):
    """
    Custom "action" for argparse arguments that converts the input values to an
    array.

    Parameters
    ----------
    values : Input values.

    Returns
    -------
    values : Input values as an array.

    Raises
    ------
    argparse.ArgumentTypeError()
        If the number of input values is not equal to 3.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                "{0} is not a valid vector".format(values)
            )
        return super().__call__(parser, namespace, values, option_string)


parser = argparse.ArgumentParser(
    fromfile_prefix_chars='@',
    description="computations of response properties",
)
parser.convert_arg_line_to_args = convert_arg_line_to_args  # Kludge replacement

parser.add_argument(
    "-m", "--mode",
    metavar="",
    type=int,
    required=True,
    help="choose mode"
)
parser.add_argument(
    "-x", "--xyz",
    metavar="",
    dest="xyz_file_path",
    type=file_path,
    required=True,
    help="absolute path of .xyz file"
)
parser.add_argument(
    "-o", "--out",
    metavar="",
    dest="out_file_path",
    type=file_path,
    required=True,
    help="absolute path of .out file"
)
parser.add_argument(
    "-e", "--external",
    metavar="",
    dest="E_external",
    type=int,
    nargs="+",
    action=StoreAsArray,
    default=np.array([5, 0, 0]),
    help="cartesian values for the external field vector"
)
parser.add_argument(
    "-p", "--path",
    metavar="",
    dest="output_path",
    type=file_path,
    default=os.path.dirname(os.path.abspath(sys.argv[0])),
    help="absolute path for the output files"
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="give more output"
)
group.add_argument(
    "-q", "--quiet",
    action="store_true",
    help="give less output"
)

parser.add_argument(
    "-V", "--version",
    action="version",
    version="%(prog)s v5.1",
    help="show version and exit"
)

args = parser.parse_args()

assert isinstance(args.E_external, np.ndarray), "invalid type for external"
assert args.mode == 1 or args.mode == 2 or args.mode == 3, "invalid mode"

# ==============================================================================
#   MISCELLANEOUS
# ==============================================================================

start_time = time.time()

# ==============================================================================
#   READ FILES
# ==============================================================================

coordinates, x_coordinates, y_coordinates, z_coordinates = f.xyz(
    args.xyz_file_path
)

if args.mode == 1:
    freq = f.freq(args.out_file_path)

if args.mode == 3:
    freq, aNP_xx, aNP_yy, aNP_zz = f.out_NP(args.out_file_path)

# ==============================================================================
#   OUTPUT
# ==============================================================================

if args.mode == 3:
    sh.header(
        freq,
        os.path.basename(args.xyz_file_path),
        os.path.basename(args.out_file_path),
        coordinates,
        aNP_xx,
        aNP_yy,
        aNP_zz
    )

# ==============================================================================
#   COMPUTATIONS
# ==============================================================================

if args.mode == 1:
    pl_freq, ex_freq, pl_str, ex_lftm, stat_pol = calc.pol_par(
        os.path.basename(args.xyz_file_path)
    )

p_dist, o_dist = calc.spatial_dist(
    coordinates
)

x_diff, y_diff, z_diff = calc.point_diff(
    coordinates
)

T_xx, T_yy, T_zz, T_xy, T_xz, T_yz = calc.T(
    x_diff,
    y_diff,
    z_diff,
    p_dist
)

temp_A = calc.tensor_stack(
    [
        T_xx,
        T_xy,
        T_xz,
        T_xy,
        T_yy,
        T_yz,
        T_xz,
        T_xy,
        T_zz
    ]
)

temp_A = calc.tensor_gentrificator(temp_A)

# ==============================================================================
#   FOR LOOP THROUGH FREQUENCIES
# ==============================================================================

if args.mode == 1 or args.mode == 3:
    NP_dip_x = []
    NP_dip_y = []
    NP_dip_z = []
    NP_abs_x = []
    NP_abs_y = []
    NP_abs_z = []

    if args.mode == 1:
        aNP_list = []

for i, frequency in enumerate(freq):

    sh.iteration(freq, i)

    if args.mode == 1:
        A, aNP = calc.nano_pol(
            temp_A,
            pl_freq,
            ex_freq,
            pl_str,
            ex_lftm,
            stat_pol,
            freq,
            i
        )
        aNP_list.append(aNP)

    if args.mode == 3:
        A = calc.nano_diag(
            aNP_xx,
            aNP_yy,
            aNP_zz,
            temp_A,
            i
        )

    dipole_x = np.full((len(coordinates), 1), 1 + 0.j)
    dipole_y = np.full((len(coordinates), 1), 1 + 0.j)
    dipole_z = np.full((len(coordinates), 1), 1 + 0.j)

    counter = 0

    while True:

        counter += 1

        E = calc.E(o_dist, args.E_external, dipole_x, dipole_y, dipole_z,
                   coordinates, x_coordinates, y_coordinates, z_coordinates)

        dipole = np.dot(A, E)

        check_x = dipole[0:len(dipole):3]
        check_y = dipole[1:len(dipole):3]
        check_z = dipole[2:len(dipole):3]

        if (np.array_equal(dipole_x, check_x) == True
            and np.array_equal(dipole_y, check_y) == True
            and np.array_equal(dipole_z, check_z) == True
                or counter > 999):

            if args.mode == 1 or args.mode == 3:
                sh.NP_dipole(counter, dipole)

                plot.NP_dipole_append(
                    NP_dip_x,
                    NP_dip_y,
                    NP_dip_z,
                    NP_abs_x,
                    NP_abs_y,
                    NP_abs_z,
                    dipole
                )

            break

        dipole_x = dipole[0:len(dipole):3]
        dipole_y = dipole[1:len(dipole):3]
        dipole_z = dipole[2:len(dipole):3]

if args.mode == 1 or args.mode == 3:
    sh.min_max(freq, NP_abs_x, NP_abs_y, NP_abs_z, NP_dip_x, NP_dip_y, NP_dip_z)

# ==============================================================================
#   PLOTS
# ==============================================================================

if not args.quiet == True:
    if args.mode == 1 or args.mode == 3:
        NP_dip_x = np.asarray(NP_dip_x)
        NP_dip_y = np.asarray(NP_dip_y)
        NP_dip_z = np.asarray(NP_dip_z)
        NP_abs_x = np.asarray(NP_abs_x)
        NP_abs_y = np.asarray(NP_abs_y)
        NP_abs_z = np.asarray(NP_abs_z)

        plot.NP_dipole(
            freq,
            NP_dip_x,
            NP_dip_y,
            NP_dip_z,
            NP_abs_x,
            NP_abs_y,
            NP_abs_z,
            os.path.basename(args.xyz_file_path),
            coordinates,
            args.output_path,
            os.path.basename(args.out_file_path)
        )

    if args.verbose == True:
        plot.avg_NP_dipole(
            freq,
            NP_dip_x,
            NP_dip_y,
            NP_dip_z,
            NP_abs_x,
            NP_abs_y,
            NP_abs_z,
            os.path.basename(args.out_file_path),
            os.path.basename(args.xyz_file_path),
            args.output_path,
            coordinates
        )

        if args.mode == 3:
            plot.NP_pol(
                freq,
                aNP_xx,
                aNP_yy,
                aNP_zz,
                os.path.basename(args.out_file_path),
                os.path.basename(args.xyz_file_path),
                args.output_path,
                coordinates
            )
            plot.avg_NP_pol(
                aNP_xx,
                aNP_yy,
                aNP_zz,
                freq,
                os.path.basename(args.out_file_path),
                os.path.basename(args.xyz_file_path),
                args.output_path,
                coordinates
            )

        elif args.mode == 1:
            aNP_list = np.asarray(aNP_list)

            plot.NP_polarizability(
                aNP_list,
                freq,
                os.path.basename(args.out_file_path),
                os.path.basename(args.xyz_file_path),
                args.output_path,
                coordinates
            )

            
# ==============================================================================
#   MISCELLANEOUS OUTPUT
# ==============================================================================

sh.misc(start_time)
