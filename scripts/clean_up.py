import argparse
from amhelpers.experiment import Postprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    args = parser.parse_args()
    Postprocessing(args.experiment_path).remove_files()
