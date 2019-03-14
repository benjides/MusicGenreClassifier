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

    parser.add_argument('config', help="Config file to use", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
