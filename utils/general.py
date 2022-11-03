import datetime
import os
import contextlib
import time
import tensorflow as tf

def get_save_dir_name(opt, mode='train', makedir=False):
    path = opt.save_dir
    now_date = datetime.datetime.now()
    date = '%dy_%dm_%dd_%dh_%dm'%(now_date.year, now_date.month, now_date.day, now_date.hour, now_date.minute)

    save_count = 0
    if os.path.exists(path) and len(os.listdir(path)) >= 1:
        save_count = max(save_count, int(sorted(os.listdir(path))[-1].split('_')[0])+1)

    title = ''
    title += '%05d_'%save_count
    title += '%s_'%mode
    title += date

    save_dir_name = os.path.join(path, title)

    if makedir:
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name, exist_ok=True)
            os.mkdir(os.path.join(save_dir_name, 'ckpt'))

    return save_dir_name 



class Timer(contextlib.ContextDecorator):
    def __init__(self, time=0.0):
        self.init_time(time)
        self.cuda = tf.test.is_gpu_available()

    def __enter__(self):
        self.init_time()
        self.start = self.get_time()
        return self

    def __exit__(self, type, value, traceback):
        self.time += self.get_time() - self.start

    def init_time(self, time=0.0):
        self.time = time
        
    def get_time(self):
        return time.time()

        




