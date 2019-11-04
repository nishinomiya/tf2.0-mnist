import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import os

# Tensorflow 2.0 Eager execution 
# mnist

class Trainer(object):
  def __init__(self, inputs, outputs):
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    self.optimizer = tf.keras.optimizers.Adam()
    self.model_path = './my_model'
    self.create_model(inputs, outputs)
    self.checkpoint_prefix = os.path.join(self.model_path, "ckpt")
    self.ckptroot = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

  def create_model(self, inputs, outputs):
    inputs_ = layers.Input((inputs,))
    x = layers.Dense(128, activation="relu")(inputs_)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(outputs, activation="softmax")(x)
    self.model = tf.keras.models.Model(inputs_, x)
    return self.model

  @tf.function
  def train_on_batch(self, X, y):
    with tf.GradientTape() as tape:
      prediction = self.model(X, training=True)
      loss = self.loss(y, prediction)
    graidents = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(graidents, self.model.trainable_weights))
    self.accuracy.update_state(y, prediction)
    return loss

  def train(self, trainset, epochs):
    for epoch in range(epochs):
      self.accuracy.reset_states()
      for step, (X, y) in enumerate(trainset):
        loss_val = self.train_on_batch(X, y)
        if step % 100 == 0:
          print("epoch = {} step = {} loss = {} accuracy = {}".format(epoch, step, loss_val, self.accuracy.result()))

  def save(self):
    self.ckptroot.save(self.checkpoint_prefix)
  
  def load(self):
    self.ckptroot.restore(tf.train.latest_checkpoint(self.model_path))
