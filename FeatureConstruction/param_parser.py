import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Feature Extraction (gpickle2feature) based on graph theory.")

    parser.add_argument('--input-path',
                        nargs='?',
                        default='../../../b10704118/OpcodeTask/Benignware_FCG/',
                        help='input binary file.')

    parser.add_argument('--output-path',
                        nargs='?',
                        default='.',
                        help='input binary file.')

    return parser.parse_args()
