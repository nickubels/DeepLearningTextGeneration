import json
import os
import re


def clean_tweet(tweet):
	tweet["text"] = tweet["text"].replace('&amp;', '&')

	tweet["text"] = re.sub("\n", "", tweet["text"]) # remove newline
	# tweet["text"] = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet["text"], flags=re.MULTILINE)
	# tweet["text"] = re.sub(r'https?:\/\/.*[\r\n]', '', tweet["text"], flags=re.MULTILINE) # remove urls
	tweet["text"] = re.sub(r'http\S+', '', tweet["text"]) # remove urls
	tweet["text"] = " ".join(tweet["text"].split()) # remove redundant white space
	# tweet = re.sub(r'[_"\-;%()|.,+&=*%]', '', tweet)
	# tweet = re.sub(r'\.', ' . ', tweet)
	# tweet = re.sub(r'\!', ' !', tweet)
	# tweet = re.sub(r'\?', ' ?', tweet)
	# tweet = re.sub(r'\,', ' ,', tweet)
	# tweet = re.sub(r':', ' : ', tweet)
	# tweet = re.sub(r'#', ' # ', tweet)
	# tweet = re.sub(r'@', ' @ ', tweet)
	# tweet = re.sub(r'd .c .', 'd.c.', tweet)
	# tweet = re.sub(r'u .s .', 'd.c.', tweet)
	# tweet = re.sub(r' amp ', ' and ', tweet)
	# tweet = re.sub(r'pm', ' pm ', tweet)
	# tweet = re.sub(r'news', ' news ', tweet)
	# tweet = re.sub(r' . . . ', ' ', tweet)
	# tweet = re.sub(r' .  .  . ', ' ', tweet)
	# tweet = re.sub(r' ! ! ', ' ! ', tweet)
	# tweet = re.sub(r'&amp', 'and', tweet)
	return tweet


def collect_tweets(data_path):
	"""
	This function analyzes the stored json tweet data to create the long sample text
	It opens each json file and adds the text of every tweet to a resulting string
	and then returns that string.
	This function is called by generate_text.py
	"""

	json_files = os.listdir(data_path)

	all_tweets = []

	for json_file in json_files:
		path = 'data/' + json_file
		with open(path, 'r') as fp:
			tweets = json.loads(fp.read())
			for tweet in tweets:
				if not tweet["is_retweet"]:
					tweet = clean_tweet(tweet)
					if len(tweet) > 0: # after removing url there should still be text
						all_tweets.append(tweet["text"])
		fp.close()

	return all_tweets


def main():
	data_path = './data'
	tweets = collect_tweets(data_path)

main()