import argparse
import logging
import os
import sys

# Dit is ff copypasta
import json
import itertools
import numpy as np
import os
import pandas as pd
import random
import torch

from fastai.text import TextLMDataBunch, URLs
from fastai.text import language_model_learner
from sklearn.model_selection import train_test_split
from torch import nn, optim

random.seed(2)
# einde copypasta

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
        # TODO: move to args
        self.dropout = 0.5 
        self.epochs = 8
        self.batch_size = 32


        self.trained = False
        logger.info(self.args)
        print("Init")

    def load_data(self):
        logger.info("Start loading data")
        with open(os.path.join(self.args.data_path,'tweets.csv'),'r') as file:
            reader = csv.reader(file)
            data = []
            for row in reader:
                data.append(row[0])
        train_data, validation_data = train_test_split(
            list(map(lambda x: x.lower(), data)),
            test_size=0.05,
            random_state=1
        )
        self.train_df = pd.DataFrame({'tweet': train_data})
        self.validation_df = pd.DataFrame({'tweet': validation_data})

    def train(self, epochs=1, batch_size=32):
        self.data_lm = TextLMDataBunch.from_df(
            'data',
            self.train_df,
            self.validation_df,
            text_cols='tweet',
            bs=batch_size
        )
        if not self.trained:
            self.model = language_model_learner(self.data_lm, arch=AWD_LSTM, drop_mult=self.dropout)
            self.model.fit_one_cycle(1, 1e-2)
            self.model.unfreeze()
            self.model.fit_one_cycle(1, 1e-3)
            self.trained = True
        self.model.fit(epochs, lr=1e-3, wd=1e-7)


    def generate(self, count=10, max_words=70):
        generated_tweets = []
        while len(generated_tweets) < count:
            raw_generated = self.model.predict("xxbos", n_words=max_words, temperature=0.8)
            raw_tweets = raw_generated.split("xxbos ")[1:]
            for tweet in raw_tweets:
                tweet = tweet.replace('hyperlink', '')[:-1]
                if tweet:
                    generated_tweets.append(tweet)
        return generated_tweets


def main():
    generation = TextGeneration()
    generation.load_data()
    generation.train()
    generation.model.save("trained_model",return_path=True)
    generated_tweets = generation.generate(5)
    print('\n'.join(generated_tweets))


if __name__ == '__main__':
    main()
