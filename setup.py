
from config.config import MyConfig
from preprocess import StcPreprocess
from batchsize import CreateBatch
from networks.nnLSTM import TwoBiLSTM
from trainLSTM import train


if __name__ == '__main__':

    cfg_dict = MyConfig()
    print(cfg_dict.Adadelta)

    data_info = StcPreprocess("train.sd").stc_info

    train_batch = CreateBatch(data_info, cfg_dict)
    train_iter = train_batch.batch_iter
    train_vocab_num = train_batch.vocab_num

    cfg_dict.target_embed_num = train_vocab_num['Target']
    cfg_dict.tweet_embed_num = train_vocab_num['Tweet']

    lstm_model = TwoBiLSTM(cfg_dict)

    train(train_iter, None, None, lstm_model, cfg_dict)

    print("Success!")
