import argparse
import os
import numpy as np


def file_path(string):
    """
    Custom "type" for argparse arguments that checks if the given file exists.

    Parameters
    ----------
    inputf : Input containing the absolute file path of the .xyz or .out file.

    Returns
    -------
    inputf : The given string constant is returned if valid.

    Raises
    ------
    argparse.ArgumentTypeError()
        If the given file path is invalid.
    """
    if not os.path.exists(string):
        raise argparse.ArgumentTypeError("{0} does not exist".format(string))

    return string


class StoreAsArray(argparse._StoreAction):
    """
    Custom "action" for argparse arguments that conforms the input values to a
    np.array().

    Parameters
    ----------
    values : Input values.

    Returns
    -------
    args._ : Input values as a np.array().

    Raises
    ------
    argparse.ArgumentTypeError()
        If the number of input values is not equal to 3.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                "{0} is not a valid vector".format(values))
        return super().__call__(parser, namespace, values, option_string)


parser = argparse.ArgumentParser(
    description="classical computation of the complex molecular induced dipole moments of a given molecule-nanoparticle system",
    epilog="NOTE: mode: 1 for ONLY a nanoparticle; 2 for BOTH a nanoparticle and a molecule"
)
parser.add_argument(
    "-m", "--mode",
    metavar="",
    type=int,
    required=False,
    help="choose mode"
)
parser.add_argument(
    "-x", "--xyz",
    metavar="",
    dest="xyz_file_path",
    type=file_path,
    required=False,
    help="absolute path of .xyz file"
)
parser.add_argument(
    "-o", "--out",
    metavar="",
    dest="out_file_path",
    type=file_path,
    required=False,
    help="absolute path of .out file"
)
parser.add_argument(
    "-e", "--external",
    metavar="",
    type=int,
    nargs="+",
    action=StoreAsArray,
    default=np.array([5, 0, 0]),
    help="cartesian values for the external field vector"
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
    version="%(prog)s v5",
    help="show version and exit"
)

args = parser.parse_args()

assert isinstance(args.external, np.ndarray), "invalid type for external"
assert args.mode == 1 or args.mode == 2, "invalid mode"

xyz_file_path = args.xyz_file_path
out_file_path = args.out_file_path

if args.mode == 1:
    print("1")
elif args.mode == 2:
    print("2")
