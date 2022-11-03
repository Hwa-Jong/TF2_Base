import tensorflow as tf
import models
from  tensorflow.keras import Model, layers
import numpy as np


def train_fit():
    # load data
    x_train = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.rand(10000, 32, 32, 3)
    # data preprocessing
    x_train = x_train.astype(np.float)/255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    model = models.BaseModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    metric = tf.keras.metrics.Mean()

    model.build(input_shape=(10000,32,32,3))
    model.compile(optimizer,loss_fn,metric)
    model.summary()

def train():
    # load data
    x_train = np.zeros((10000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.rand(10000, 32, 32, 3)
    # data preprocessing
    x_train = x_train.astype(np.float)/255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    model = models.BaseModel()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()

    metric = tf.keras.metrics.Mean()

    epochs = 2
    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for iter, x_batch_train in enumerate(train_dataset):
            x = x_batch_train[0]
            y = x_batch_train[1]
            with tf.GradientTape() as tape:
                pred = model(x)
                # Compute reconstruction loss
                loss_batch = loss_fn(y, pred)
                loss_batch += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss_batch, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            metric(loss_batch)

            if iter % 100 == 0:
                print("iter %d: mean loss = %.4f" % (iter, metric.result()))



def main():
    train()



if __name__=='__main__':
    main()