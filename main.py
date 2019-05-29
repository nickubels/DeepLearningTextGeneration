import argparse
import logging
import os
import sys

from fastai.text import *

logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

def get_args():
    parser = argparse.ArgumentParser(description='Script to generate Trump tweets')
    parser.add_argument('--data_path', '-d', metavar='STRING', default='./tweets',
                        help="Where the CSV with the tweets is stored")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='./output',
                        help="Where the output will be stored")

    return parser.parse_args()


class TextGeneration:
    def __init__(self):
        # Here we init stuff
        self.args = get_args()
        logger.info(self.args)
        print("Init")

    def load_data(self):
        logger.info("Start loading data")
        data = TextDataBunch.from_csv(self.args.data_path,'tweets.csv')

    def run(self):
        self.load_data()


def main():
    generation = TextGeneration()
    generation.run()


if __name__ == '__main__':
    main()
