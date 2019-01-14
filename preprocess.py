"""
Name:           preprocess.py
Function:       Extract every sentence from the data.
                Split the sentence into three parts:
                Target, Tweet and Attitude.

Author:         LiXu
Create Date:    2019.01.11
Modify Date:    2019.01.11
Version:        0.1
"""

import re
import random

import numpy as np

random.seed(0)
np.random.seed(0)


class StcPreprocess(object):
    """
    Class Name:     StcPreprocess
    Function:       Split the sentence in the data file into
                    three parts: Target, tweet and Attitude.
    Parameter:      The name of the file you want to preprocess.
    """

    def __init__(self, file_name=None):

        self.path = "./data/initial/"
        self.data_name = ["dev.sd", "test.sd", "train.sd"]

        self.name_map = {
            'Climate Change is a Real Concern':     'climate',
            'Hillary Clinton':                      'clinton',
            'Atheism':                              'atheism',
            'Feminist Movement':                    'feminist',
            'Legalization of Abortion':             'abortion',
            'Donald Trump':                         'trump'
        }
        self.attitude_map = {
            'AGAINST':  0,
            'NONE':     1,
            'FAVOR':    2
        }
        self.stc_info = self.preprocess(file_name)
        self.write_to_csv(file_name)

    def preprocess(self, file_name):

        if file_name not in self.data_name:
            assert "Your input file name isn't in train, dev and test!"
            exit(1)

        target_tweet_label = list()

        with open(self.path + file_name, encoding='utf-8') as corpus:
            for line in corpus:
                line = line.rstrip()

                target_info, span = self.target(line)
                tweet_info = self.tweet(line, span)
                attitude_info = self.attitude(line)

                info_dict = {'target': target_info,
                             'tweet': tweet_info,
                             'attitude': attitude_info}

                target_tweet_label.append(info_dict)

        return target_tweet_label

    def target(self, sentence):
        """
        :param sentence:        One sentence in data file.
        :return:                Target form name mapping.
        """

        # Map the start of the sentence in the name mapping.
        for topic in self.name_map:
            topic_match = re.match(topic, sentence)

            # Return the short form and the span of target
            # if the start of the sentence in the name mapping.
            if topic_match:
                return self.name_map[topic_match.group()], topic_match.span()

        return None

    def tweet(self, sentence, span):

        # If there isn't the target in the sentence,
        # the sentence wouldn't be processsed.
        if not span:
            return None

        # Set the start location of the sentence without target.
        char_start = span[1] + 1

        # Remove the target and mood to get the tweet sentence.
        remove_target = sentence[char_start::]
        after_token = remove_target.split()
        after_token.pop(-1)
        tweet = ' '.join(after_token)

        return tweet

    def attitude(self, sentence):
        """
        :param sentence:        One sentence in the data file.
        :return:                The valuable attitude of the tweet.
        """
        sentence_words = sentence.split()
        attitude_word = sentence_words[-1]
        label_val = self.attitude_map[attitude_word]

        return label_val

    def write_to_csv(self, file_name):

        write_name = file_name[0:-3] + '.csv'
        write_path = './data/csvdata/' + write_name

        with open(write_path, 'w') as csv_f:
            for item in self.stc_info:
                content = item['target'] + '\t' + item['tweet'] + '\t' + str(item['attitude']) + '\n'

                csv_f.write(content)

        print("Wirte %s to csv file." % file_name)




