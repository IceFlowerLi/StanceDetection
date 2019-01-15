
from config.config import MyConfig
from preprocess import StcPreprocess
from batchsize import CreateBatch, CreateTriIter
from networks.nnLSTM import TwoBiLSTM
from trainLSTM import train


if __name__ == '__main__':

    cfg_dict = MyConfig()
    print(cfg_dict.Adadelta)

    train_obj = StcPreprocess("train.sd")

    train_batch = CreateBatch(train_obj.name, train_obj.stc_info, cfg_dict)
    train_iter = train_batch.batch_iter
    train_vocab_num = train_batch.vocab_num

    cfg_dict.target_embed_num = train_vocab_num['Target']
    cfg_dict.tweet_embed_num = train_vocab_num['Tweet']

    dev_obj = StcPreprocess("dev.sd")
    dev_iter = CreateBatch(dev_obj.name, dev_obj.stc_info,
                           cfg_dict, train_batch.univocab).batch_iter

    test_obj = StcPreprocess("test.sd")
    test_iter = CreateBatch(test_obj.name, test_obj.stc_info,
                            cfg_dict, train_batch.univocab).batch_iter

    '''
    # Create train iter, dev iter and test iter.
    train_info = StcPreprocess("train.sd")
    dev_info = StcPreprocess("dev.sd")
    test_info = StcPreprocess("test.sd")

    iters = CreateTriIter(cfg_dict)
    train_iter, dev_iter, test_iter = iters.create_iterators()
    '''

    lstm_model = TwoBiLSTM(cfg_dict)

    train(train_iter, dev_iter, test_iter, lstm_model, cfg_dict)

    print("Success!")
