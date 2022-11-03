import tensorflow as tf
from models.models import BaseModel
import os
import argparse
import logging



import numpy as np

from utils import seed, logger, general
# tensorboard --logdir=.\results\00006_train_2022y_11m_3d_18h_33m\tensorboard

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.join('.', 'results'), help='dir path to save resutls')
    parser.add_argument('--dataset_path', type=str, default='dataset', help='dataset path(need to train, valid and test)')
    #parser.add_argument('--model_weights', type=str, default='results\\00001_train_2022y_10m_27d_14h_22m\\best_loss.pt', help='to load weights')
    parser.add_argument('--model_weights', type=str, default='', help='to load weights')
    parser.add_argument('--ckpt_term', type=int, default=5, help='checkpoint term to save')
    parser.add_argument('--device', type=str, default='cuda:0', help='your device : cpu or cuda:0')
    parser.add_argument('--seed', type=int, default=1234, help='fix seed')

    ###    
    parser.add_argument('--batch_size', type=int, default=32, help='set batch size')
    parser.add_argument('--lr', type=float, default=1E-3, help='set learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='set epoch')
    parser.add_argument('--scheduler', type=bool, default=False, help='use scheduler')
    parser.add_argument('--loss', type=str, default='l1', help='set loss : [l1 l2 ce(notworking)]')
    opt = parser.parse_args()
    return opt


def set_logger(path):
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(os.path.join(path, 'log_train.log'))
    #formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
    formatter = logging.Formatter('%(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    return logger


def train(): 
    opt = get_opt()
    mode = 'train'
    device = opt.device
    save_dir = general.get_save_dir_name(opt, mode=mode, makedir=True)
    
    # memo
    memo = 'Code 화질 개선 AI 모델(U-net V1) \
        \nResnet 구조 채택'

    # log
    logger = set_logger(save_dir)
    logger.info('< memo >')
    logger.info(memo)
    logger.info('< info >')
    logger.info('dir path : %s'%save_dir)
    logger.info('< option >')
    for k, v in opt._get_kwargs():
        logger.info('%s : %s'%(k, v))


    # tensorboard
    writer = tf.summary.create_file_writer(os.path.join(save_dir, 'tensorboard'))



    # load data
    x_train = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.rand(10000, 32, 32, 3)
    # data preprocessing
    x_train = x_train.astype(np.float)/255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_dataset = valid_dataset.shuffle(buffer_size=1024).batch(64)

    
    # load model
    model = BaseModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    


    # timer ( total time, now train time, now valid time )
    timer = (general.Timer(), general.Timer(), general.Timer())
    
    

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)

        valid_loss(loss)


    start_epoch = 1

    logger.info('---------- train start ----------')
    best_loss = 1000
    best_loss_epoch = 0
    with timer[0]:
        for epoch in range(start_epoch, opt.epochs+1):
            with timer[1]: # Train
                train_loss.reset_states()
                for iter, x_batch_train in enumerate(train_dataset):
                    x = x_batch_train[0]
                    y = x_batch_train[1]
                    train_step(x, y)

            with writer.as_default():
                tf.summary.scalar('Loss/train', train_loss.result(), epoch)


            with timer[2]:
                valid_loss.reset_states()
                for iter, x_batch_valid in enumerate(valid_dataset):
                    x = x_batch_valid[0]
                    y = x_batch_valid[1]
                    valid_step(x, y)
                    
            with writer.as_default():
                tf.summary.scalar('Loss/valid', valid_loss.result(), epoch)

            logger.info('%d/%d << train loss : %.5f | valid loss: %.5f >>  train time: %.1f | valid time: %.1f sec'%(epoch, opt.epochs, train_loss.result(), valid_loss.result(), timer[1].time, timer[2].time))

            
            # best loss model save
            if best_loss > valid_loss.result():
                best_loss = valid_loss.result()
                best_loss_epoch = epoch
                model.save(filepath=os.path.join(save_dir, 'best_loss'))

            #ckpt save  
            if  epoch % opt.ckpt_term == 0: 
                model.save(filepath=os.path.join(save_dir, 'ckpt', 'epoch{%d}'%epoch))



    writer.flush()
    writer.close()
    logger.info('---------- result ----------')
    logger.info('train epoch : %d'%opt.epochs)
    logger.info('total time : %.1f sec'%(timer[0].time))
    logger.info('best loss epoch : %d '%(best_loss_epoch))
    logger.info('best loss : %f '%(best_loss))




def main():
    train()



if __name__=='__main__':
    main()

"""
def train_fit():
    # load data
    x_train = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.rand(10000, 32, 32, 3)
    # data preprocessing
    x_train = x_train.astype(np.float)/255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    model = BaseModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    metric = tf.keras.metrics.Mean()

    model.build(input_shape=(10000,32,32,3))
    model.compile(optimizer,loss_fn,metric)
    model.summary()
"""