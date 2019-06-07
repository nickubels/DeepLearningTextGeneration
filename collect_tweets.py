import argparse
import codecs
import csv
import json
import os
import re


def get_args():
    parser = argparse.ArgumentParser(description='Obtain tweets from json files')
    parser.add_argument('--data_path', '-d', metavar='STRING', default=os.path.join(os.path.curdir, 'data'),
                        help="Where the json files containing tweets are stored")
    parser.add_argument('--output_path', '-o', metavar='STRING', default=os.path.join(os.path.curdir, 'tweets'),
                        help="Where the output will be stored")

    return parser.parse_args()


class Tweets(object):
    def __init__(self):
        self.args = get_args()

    @staticmethod
    def clean_tweet(tweet):
        """
        This function cleans the tweet from its emojjitions, urls and other redundant characters
        :param tweet: the tweet that must be cleaned
        :return: the cleaned tweet
        """
        # Remove emojis
        tweet["text"] = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE).sub(r'', tweet["text"])

        # Replace &
        tweet["text"] = tweet["text"].replace('&amp;', '&')

        # Remove newline
        tweet["text"] = re.sub("\n", "", tweet["text"])

        # Remove urls
        tweet["text"] = re.sub(r'http\S+', '', tweet["text"])  # remove urls
        # tweet["text"] = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet["text"], flags=re.MULTILINE)
        # tweet["text"] = re.sub(r'https?:\/\/.*[\r\n]', '', tweet["text"], flags=re.MULTILINE) # remove urls

        # Remove redundant white space
        tweet["text"] = " ".join(tweet["text"].split())

        return tweet

    def collect_tweets(self):
        """
        This function loads all the tweets in the given directory, cleans them and exports the tweets as a csv file.
        """
        # Obtain all json files in directory
        json_files = os.listdir(self.args.data_path)

        all_tweets = []
        # Obtain all tweets in each json file
        for json_file in json_files:
            if json_file[-4:] != 'json':
                continue
            with open(os.path.join(self.args.data_path, json_file), 'r') as fp:
                try:
                    print("load this file: ", os.path.join(self.args.data_path, json_file))
                    tweets = json.loads(fp.read())
                except json.decoder.JSONDecodeError:
                    # It should also be able to load json files with UTF-8 BOM header
                    tweets = json.load(codecs.open(os.path.join(self.args.data_path, json_file), 'r', 'utf-8-sig'))
                except:
                    print('I skip this file: ', os.path.join(self.args.data_path, json_file))
                    continue
                for tweet in tweets:
                    # We don't want any retweets in the data set
                    if not tweet["is_retweet"]:
                        # Clean the tweet
                        tweet = self.clean_tweet(tweet)
                        # If the cleaned tweet contains any text, it should be added to the data set
                        if len(tweet) > 0:
                            all_tweets.append(tweet["text"])
            fp.close()

        # Check if the directory in which the csv file should be stored exists, if not create it
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

        # Export tweets as a csv file
        with open(os.path.join(self.args.output_path, 'tweets_after_2011.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for tweet in all_tweets:
                writer.writerow([tweet])
                # writer.writerow([tweet.encode("utf-8")])


def main():
    tweets = Tweets()
    tweets.collect_tweets()


if __name__ == '__main__':
    main()
