"""
Name:           batchsize.py
Function:       Corresponding every input data set,
                build its vocabulary and batch iterator.

Author:         LiXu
Create Date:    2019.01.12
Modify Date:    2019.01.12
Version:        0.1
"""


from torchtext.data import Field, Example, Dataset, Iterator


class CreateBatch(object):

    def __init__(self, stc_info, args):

        self.stc_info = stc_info
        self.args = args

        # Build fields, dataset, word dictionary and batch iterator.
        self.three_fields = self.build_fields()
        self.datasets = self.create_dataset()
        self.vocab_num = self.build_dict()
        self.batch_iter = self.create_iterator()

    def build_fields(self):

        TAEGET = Field(sequential=False)
        TWEET = Field(sequential=True, tokenize=lambda x: x.split())
        ATTITUDE = Field(sequential=False, use_vocab=False)

        three_filds = [
            ('Target', TAEGET),
            ('Tweet', TWEET),
            ('Attitude', ATTITUDE)
        ]

        return three_filds

    def create_one_example(self, one_sample):
        """
        :param one_sample:      One sample from sentence information
                                it contained three fields that are
                                target, tweet and attitude and it's
                                corresponding value.
        :return:                An Example object to be consisted of
                                the every batch.
        """

        # Extract value of its corresponding field.
        target_val = one_sample['target']
        tweet_val = one_sample['tweet']
        attitude_val = one_sample['attitude']

        # Merge the three parts value.
        val_list = [target_val, tweet_val, attitude_val]

        example = Example.fromlist(val_list, self.three_fields)

        return example

    def create_dataset(self):

        # Collect every example into a list,
        examples = list()
        for one_sample in self.stc_info:
            example_obj = self.create_one_example(one_sample)
            examples.append(example_obj)

        # Build datasets with an example list.
        datasets = Dataset(examples, self.three_fields)

        return datasets

    def build_dict(self):

        # Build dictionary for target field and tweet field.
        self.three_fields[0][1].build_vocab(self.datasets)
        self.three_fields[1][1].build_vocab(self.datasets)

        target_vocab = len(self.three_fields[0][1].vocab)
        tweet_vocab = len(self.three_fields[1][1].vocab)

        vocab_num = {'Target': target_vocab,
                     'Tweet': tweet_vocab}

        return vocab_num

    def create_iterator(self):

        iterator = Iterator(self.datasets,
                            self.args.batch_size,
                            shuffle=self.args.shuffle,
                            device=self.args.device)

        return iterator


