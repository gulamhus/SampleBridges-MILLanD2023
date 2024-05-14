import argparse
import importlib
import pathlib


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-t', '--task',
    help="Which task to start; choices are 'rating' and 'keypoints'"
)
arg_parser.add_argument(
    '-d', '--data_path',
    help="Path from where to read the task data"
)


if __name__ == '__main__':
    args, _ = arg_parser.parse_known_args()

    task = importlib.import_module(args.task)
    window = task.LabelWindow(pathlib.Path(args.data_path))
