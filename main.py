import argparse
import logging
import sys
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from fastai.text import TextLMDataBunch, URLs, language_model_learner, csv, AWD_LSTM, load_learner, Path

# import os
#
# # Dit is ff copypasta
# import json
# import itertools
# import numpy as np

# import random
# import torch
#
# from torch import nn, optim
#
# # random.seed(2)
# # einde copypasta
# # from fastai.text import *


logger = logging.getLogger()
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Script to generate Trump tweets')
    parser.add_argument('--job_id', '-j', metavar='STRING', default="", help="Job_id used for saving files")
    parser.add_argument('--data', '-d', metavar='STRING', default='./tweets/tweets.csv',
                        help="The CSV with the tweets")
    parser.add_argument("--train", '-t', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Do we need to train a model?")
    parser.add_argument('--output_path', '-o', metavar='STRING', default='./output',
                        help="Where the output will be stored")
    parser.add_argument('--use_pretrained', '-p', metavar='BOOL', default=True,
                        help="If we use a pretrained model")
    parser.add_argument('--model_path', '-mp', metavar='STRING',
                        default=os.path.join(os.getcwd(), 'model/'),
                        help="Where the model is or should be stored")
    parser.add_argument('--model', '-m', metavar='STRING',
                        default='finetune_trump.pkl', help="Which model to load, if any")

    return parser.parse_args()

    


class TextGeneration:
    def __init__(self):
        # Here we init stuff
        self.args = get_args()
        self.trained = False

        self.dropout = 0.5
        self.epochs = 8
        self.batch_size = 32

        self.train_df = None
        self.validation_df = None
        self.model = None
        self.data_lm = None
        logger.info(self.args)

        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def load_data(self):
        logger.info("Start loading data")
        with open(self.args.data, 'r') as file:
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
            logger.info("Using a pretrained_model to finetune: " + str(self.args.use_pretrained))
            self.model = language_model_learner(self.data_lm, arch=AWD_LSTM,
                                                pretrained=self.args.use_pretrained, drop_mult=self.dropout)
            self.model.fit_one_cycle(1, 1e-2)
            self.model.unfreeze()
            self.model.fit_one_cycle(1, 1e-3)
            self.trained = True
        self.model.fit(epochs, lr=1e-3, wd=1e-7)


    def generate(self, count=10, max_words=70):
        logger.info("Generating tweets")
        generated_tweets = []
        while len(generated_tweets) < count:
            raw_generated = self.model.predict("xxbos", n_words=max_words, temperature=0.8)
            raw_tweets = raw_generated.split("xxbos ")[1:]
            for tweet in raw_tweets:
                tweet = tweet.replace('hyperlink', '')[:-1]
                if tweet:
                    generated_tweets.append(tweet)
        return generated_tweets

    def run(self):
        if self.args.train:
            logger.info("Start training the model")
            self.load_data()
            # self.train()

            self.train(epochs=3, batch_size=32)
            self.train(epochs=2, batch_size=64)
            
            if self.args.job_id == "":
                modelname = self.args.model
            else:
                modelname = str(self.args.job_id)+".pkl"
            self.model.export(Path(os.path.join(self.args.model_path, modelname)))
        else:
            logger.info("Loading a pretrained model")
            self.model = load_learner(Path(self.args.model_path), self.args.model)
        generated_tweets = self.generate(10)
        print(generated_tweets)
        print('\n'.join(generated_tweets))


def main():
    generation = TextGeneration()
    generation.run()


if __name__ == '__main__':
    main()
