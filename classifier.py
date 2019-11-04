import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import os

# Tensorflow 2.0 Eager execution 
# mnist / using tf.keras

class Trainer(object):
  def __init__(self, inputs, outputs):
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    self.optimizer = tf.keras.optimizers.Adam()
    self.model_path = './my_model'
    self.build_model(inputs, outputs)
    self.checkpoint_prefix = os.path.join(self.model_path, "ckpt")
    self.ckptroot = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

  def build_model(self, inputs, outputs) -> None:
    inputs_ = layers.Input((inputs,))
    x = layers.Dense(128, activation="relu")(inputs_)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(outputs, activation="softmax")(x)
    self.model = tf.keras.models.Model(inputs_, x)

  @tf.function
  def train_step(self, x, y) -> float:
    with tf.GradientTape() as tape:
      prediction = self.model(x, training=True)
      loss = self.loss(y, prediction)
    graidents = tape.gradient(loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(graidents, self.model.trainable_weights))
    self.accuracy.update_state(y, prediction)
    return loss

  def train(self, dataset, epochs) -> None:
    for epoch in range(epochs):
      self.accuracy.reset_states()
      for step, (x, y) in enumerate(dataset):
        loss = self.train_step(x, y)
        if step % 100 == 0:
          print("epoch = {} step = {} loss = {} accuracy = {}".format(epoch, step, loss, self.accuracy.result()))

  def save(self) -> None:
    self.ckptroot.save(self.checkpoint_prefix)
  
  def load(self) -> None:
    self.ckptroot.restore(tf.train.latest_checkpoint(self.model_path))
