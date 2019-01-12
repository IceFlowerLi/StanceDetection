
from config.config import MyConfig
from preprocess import StcPreprocess
from batchsize import CreateBatch

if __name__ == '__main__':

    cfg_dict = MyConfig()
    print(cfg_dict.Adadelta)

    data_info = StcPreprocess("train.sd").stc_info

    args = {'batch_size': 64,
            'shuffle': True,
            'device': -1}

    train_iter = CreateBatch(data_info, cfg_dict).batch_iter

    a = iter(train_iter)
    b = next(a)

    print("Success!")
