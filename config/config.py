
import os
from configparser import ConfigParser


class MyConfig(ConfigParser):

    def __init__(self):
        super(MyConfig, self).__init__()

        config_obj = ConfigParser()
        # XXX: The current path is the absolute name
        #      of the project that is
        #      D:\Document\Program\Python\StanceDetection.
        #      The config content would fail to be read
        #      if the python file in its sub folder and
        #      not set the right reading path.
        config_obj.read('./config/configfile.cfg')
        self._cfg_dict = config_obj
        print("Load config file successfully.\n")

        # Examine and display the property read
        # from the configfile.
        for section in config_obj.sections():
            print(section, ':')
            for k, v in config_obj.items(section):
                print(k, ':', v)
            print()

    """
    Function:   Transfer the string of the the section
                keys into its corresponding value.
                Put the key and its value into class property
                for getting easily by instance indexing 
                not section dictionary indexing in the future.
                
    Use:        value = instance.property
    """

    # Data
    @property
    def word_Embedding(self):
        return self._cfg_dict.getboolean('Data', 'word_Embedding')

    @property
    def freq_1_unk(self):
        return self._cfg_dict.getboolean('Data', 'freq_1_unk')

    @property
    def word_Embedding_Path(self):
        return self._cfg_dict.get('Data', 'word_Embedding_Path')

    @property
    def datafile_path(self):
        return self._cfg_dict.get('Data', 'datafile_path')

    @property
    def name_trainfile(self):
        return self._cfg_dict.get('Data', 'name_trainfile')

    @property
    def name_devfile(self):
        return self._cfg_dict.get('Data', 'name_devfile')

    @property
    def name_testfile(self):
        return self._cfg_dict.get('Data', 'name_testfile')

    @property
    def min_freq(self):
        return self._cfg_dict.getint('Data', 'min_freq')

    @property
    def shuffle(self):
        return self._cfg_dict.getboolean('Data', 'shuffle')

    @property
    def epochs_shuffle(self):
        return self._cfg_dict.getboolean('Data', 'epochs_shuffle')

    @property
    def FIVE_CLASS_TASK(self):
        if self._cfg_dict.getboolean('Data', 'FIVE_CLASS_TASK'):
            return 5
        else:
            return False

    @property
    def TWO_CLASS_TASK(self):
        if self._cfg_dict.getboolean('Data', 'TWO_CLASS_TASK'):
            return 2
        else:
            return False

    # Save
    @property
    def snapshot(self):
        value = self._cfg_dict.get('Save', 'snapshot')
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def predict(self):
        value = self._cfg_dict.get('Save', 'predict')
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def test(self):
        return self._cfg_dict.getboolean('Save', 'test')

    @property
    def save_dir(self):
        return self._cfg_dict.get('Save', 'save_dir')

    @save_dir.setter
    def save_dir(self, value):
        self._cfg_dict.set('Save', 'save_dir', str(value))

    # Model
    @property
    def static(self):
        return self._cfg_dict.getboolean("Model", "static")

    @property
    def wide_conv(self):
        return self._cfg_dict.getboolean("Model", "wide_conv")

    @property
    def LSTM(self):
        return self._cfg_dict.getboolean("Model", "LSTM")

    @property
    def BiLSTM(self):
        return self._cfg_dict.getboolean("Model", "BiLSTM")

    @property
    def embed_dim(self):
        return self._cfg_dict.getint("Model", "embed_dim")

    @property
    def lstm_hidden_dim(self):
        return self._cfg_dict.getint("Model", "lstm_hidden_dim")

    @property
    def lstm_num_layers(self):
        return self._cfg_dict.getint("Model", "lstm_num_layers")

    @property
    def target_hidden_dim(self):
        return self._cfg_dict.getint("Model", "target_hidden_dim")

    @property
    def target_num_layers(self):
        return self._cfg_dict.getint("Model", "target_num_layers")

    @property
    def tweet_hidden_dim(self):
        return self._cfg_dict.getint("Model", "tweet_hidden_dim")

    @property
    def tweet_num_layers(self):
        return self._cfg_dict.getint("Model", "target_num_layers")

    @property
    def batch_normalizations(self):
        return self._cfg_dict.getboolean("Model", "batch_normalizations")

    @property
    def bath_norm_momentum(self):
        return self._cfg_dict.getfloat("Model", "bath_norm_momentum")

    @property
    def batch_norm_affine(self):
        return self._cfg_dict.getboolean("Model", "batch_norm_affine")

    @property
    def dropout(self):
        return self._cfg_dict.getfloat("Model", "dropout")

    @property
    def dropout_embed(self):
        return self._cfg_dict.getfloat("Model", "dropout_embed")

    @property
    def max_norm(self):
        value = self._cfg_dict.get("Model", "max_norm")
        if value == "None" or value == "none":
            return None
        else:
            return value

    @property
    def clip_max_norm(self):
        return self._cfg_dict.getint("Model", "clip_max_norm")

    @property
    def init_weight(self):
        return self._cfg_dict.getboolean("Model", "init_weight")

    @property
    def init_weight_value(self):
        return self._cfg_dict.getfloat("Model", "init_weight_value")

    # Optimizer
    @property
    def learning_rate(self):
        return self._cfg_dict.getfloat("Optimizer", "learning_rate")

    @property
    def Adam(self):
        return self._cfg_dict.getboolean("Optimizer", "Adam")

    @property
    def SGD(self):
        return self._cfg_dict.getboolean("Optimizer", "SGD")

    @property
    def Adadelta(self):
        return self._cfg_dict.getboolean("Optimizer", "Adadelta")

    @property
    def momentum_value(self):
        return self._cfg_dict.getfloat("Optimizer", "optim_momentum_value")

    @property
    def weight_decay(self):
        return self._cfg_dict.getfloat("Optimizer", "weight_decay")

    # Train
    @property
    def num_threads(self):
        return self._cfg_dict.getint("Train", "num_threads")

    @property
    def device(self):
        return self._cfg_dict.getint("Train", "device")

    @property
    def cuda(self):
        return self._cfg_dict.getboolean("Train", "cuda")

    @property
    def epochs(self):
        return self._cfg_dict.getint("Train", "epochs")

    @property
    def batch_size(self):
        return self._cfg_dict.getint("Train", "batch_size")

    @property
    def log_interval(self):
        return self._cfg_dict.getint("Train", "log_interval")

    @property
    def test_interval(self):
        return self._cfg_dict.getint("Train", "test_interval")

    @property
    def save_interval(self):
        return self._cfg_dict.getint("Train", "save_interval")






