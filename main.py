"""Main Module"""
import argparse
from module.core import main

def parse_args():
    """Parse and stores the provided terminal args

    Parameters
    ----------

    Returns
    -------
    args : Parsed args

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', help="Output file to inspect",
                        default=None, type=str, action='store')

    parser.add_argument('-t', '--test', help="Runs the test of the computed model",
                        default=False, action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
