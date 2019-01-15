"""
Name:           batchsize.py
Function:       Corresponding every input data set,
                build its vocabulary and batch iterator.

Author:         LiXu
Create Date:    2019.01.12
Modify Date:    2019.01.12
Version:        0.1
"""


from torchtext.data import Field, Example
from torchtext.data import Dataset, TabularDataset
from torchtext.data import Iterator, BucketIterator


class CreateBatch(object):

    def __init__(self, obj_name, stc_info, args, vocab=None):

        self.obj_name = obj_name
        self.stc_info = stc_info
        self.univocab = vocab
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

        # Only build vocabulary dictionary for train dataset.
        if self.obj_name == 'train':
            # Build dictionary for target field and tweet field.
            self.three_fields[0][1].build_vocab(self.datasets)
            self.three_fields[1][1].build_vocab(self.datasets)

            target_vocab = len(self.three_fields[0][1].vocab)
            tweet_vocab = len(self.three_fields[1][1].vocab)

            vocab_num = {'Target': target_vocab,
                         'Tweet': tweet_vocab}

            self.univocab = {'target': self.three_fields[0][1].vocab,
                             'tweet': self.three_fields[1][1].vocab}

            return vocab_num
        else:
            self.three_fields[0][1].vocab = self.univocab['target']
            self.three_fields[1][1].vocab = self.univocab['tweet']
            return None

    def create_iterator(self):

        # Be sure the batch size according the dataset name.
        batch_size = None
        if self.obj_name == 'train':
            batch_size = self.args.batch_size
        if self.obj_name == 'dev':
            batch_size = len(self.datasets)
        if self.obj_name == 'test':
            batch_size = len(self.datasets)

        iterator = Iterator(self.datasets,
                            batch_size,
                            shuffle=self.args.shuffle,
                            device=self.args.device)

        return iterator


class CreateTriIter(object):

    def __init__(self, args):

        self.args = args

        # Build fields, dataset, word dictionary and batch iterator.
        self.three_fields = self.build_fields()
        self.datasets = self.create_datasets()
        self.vocab_num = self.build_dict()

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

    def create_datasets(self):
        train_datasets, dev_datasets, test_datasets = TabularDataset.splits(
                    path=self.args.datafile_path,
                    train='train.csv', validation='dev.csv', test='test.csv',
                    format='csv',
                    skip_header=False,
                    fields=self.three_fields)

        return {'train': train_datasets,
                'dev': dev_datasets,
                'test': test_datasets}

    def build_dict(self):

        # Only build vocabulary dictionary for train dataset.
        if self.obj.name[0:-3] == 'train':
            self.three_fields[0][1].build_vocab(self.datasets['train'])
            self.three_fields[1][1].build_vocab(self.datasets['train'])
            '''
            self.three_fields[0][1].build_vocab(self.datasets['dev'])
            self.three_fields[1][1].build_vocab(self.datasets['dev'])

            self.three_fields[0][1].build_vocab(self.datasets['test'])
            self.three_fields[1][1].build_vocab(self.datasets['test'])

            '''
            target_vocab = len(self.three_fields[0][1].vocab)
            tweet_vocab = len(self.three_fields[1][1].vocab)

            vocab_num = {'Target': target_vocab,
                         'Tweet': tweet_vocab}

            return vocab_num

        else:
            return None

    def create_iterators(self):

        '''

        train_iter, dev_iter, test_iter = BucketIterator.splits(
            (self.datasets['train'], self.datasets['dev'], self.datasets['test']),
            # batch_sizes=(self.args.batch_size, len(self.datasets['dev']), len(self.datasets['test'])),
            batch_sizes=(self.args.batch_size, self.args.batch_size, self.args.batch_size),
            device=self.args.device)
        '''

        train_iter = Iterator.splits(
            self.datasets['train'],
            batch_size=self.args.batch_size,
            device=self.args.device
        )

        dev_iter = Iterator.splits(
            self.datasets['dev'],
            batch_size=len(self.datasets['dev']),
            device=self.args.device
        )

        test_iter = Iterator.splits(
            self.datasets['test'],
            batch_size=len(self.datasets['test']),
            device=self.args.device
        )

        return train_iter, dev_iter, test_iter
