import sys, argparse
from src.predict import build_library

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fasta',
                        type=str,
                        help='path to input dlib')
    parser.add_argument('output_dlib',
                        type=str,
                        help='Directory to save model (default is Chronologer/models)',
                        default='output.dlib')
    
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])
    build_library( args.input_fasta, args.output_dlib, )
    return 0


if __name__ == "__main__":
    main()
    sys.exit()
