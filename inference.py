import tensorflow as tf



def main():
    print(tf.__version__)
    print(tf.test.is_gpu_available())



if __name__=='__main__':
    main()