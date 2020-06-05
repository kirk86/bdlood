# coding: utf-8

import os
import sys
import logging


class Config(object):
    """
    Config object to setup dirs and logger
    """
    def __init__(self):
        super(Config, self).__init__()
        pathname = os.path.dirname(sys.argv[0])
        self.root_dir = os.path.dirname(os.path.abspath(pathname))
        self.logger_path = self.root_dir + '/log/'
        self.data_path = self.root_dir + '/data/'
        self.out_path = self.root_dir + '/out/'
        self.chkpt_path = self.root_dir + '/out/chkpt'
        self.setup_dirs()

    def setup_dirs(self):
        for dirname in ['/log', '/out']:
            if not os.path.exists(self.root_dir + dirname) and \
               not os.path.isdir(self.root_dir + dirname):
                os.mkdir(self.root_dir + dirname)

    def setup_logger(self):
        # set up logging
        path = self.logger_path + 'experiment.log'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)s: %(levelname)-2s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=path,
            filemode='a'
        )
        # define handler writing messages to sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format for console
        formatter = logging.Formatter('%(name)s: %(levelname)-2s %(message)24s')
        console.setFormatter(formatter)
        # add handler to the root logger
        logger = logging.getLogger('OoD')
        logger.addHandler(console)
        return logger
