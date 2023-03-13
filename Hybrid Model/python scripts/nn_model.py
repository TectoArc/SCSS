import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model

class NN:
  def __init__(self, x_train, x_test, y_train, y_test, df_pred, inp_shape, activation:str=None, lr:float=None):
    self.df_pred = df_pred
    self.x_train, self.x_test = np.asarray(x_train).astype(np.float32), np.asarray(x_test).astype(np.float32)
    self.y_train, self.y_test = np.asarray(y_train).astype(np.float32), np.asarray(y_test).astype(np.float32)
    self.inp_shape = inp_shape
    self.activation = tf.keras.activations.get(activation)
    self.lr = lr

  def parameterize_model(self):
    x_in = tf.keras.layers.Input(shape=self.inp_shape,)
    x_1 = tf.keras.layers.Dense(units='128', activation=self.activation)(x_in)
    x_2 = tf.keras.layers.Dense(units='128', activation=self.activation)(x_1)
    x_3 = tf.keras.layers.Dense(units='64', activation=self.activation)(x_2)
    x_4 = tf.keras.layers.Dense(units='8', activation=self.activation)(x_3)
    x_out = tf.keras.layers.Dense(units='1', name='x_out')(x_4)

    model_nn = tf.keras.models.Model(inputs = x_in, outputs = x_out)
    return model_nn, model_nn.summary()

  def optimize_model(self):
    opt = tf.keras.optimizers.Adam(lr=self.lr)
    set_model, _ = self.parameterize_model()
    set_model.compile(optimizer=opt,
                loss="mse",
                metrics=tf.keras.metrics.RootMeanSquaredError())
    return set_model

  def fit_nn(self, model, epochs=100, batch_size=5, evaluation=False):
    model.fit(self.x_train, self.y_train, epochs=epochs, batch_size = batch_size, verbose=0, validation_data=(self.x_test, self.y_test))
    nn_loss, nn_rmse = model.evaluate(self.x_test, self.y_test)
    if evaluation:
      history = model.fit(self.x_train, self.y_train, epochs=epochs, batch_size = batch_size, verbose=0, validation_data=(self.x_test, self.y_test))
      return history, nn_loss, nn_rmse
    else:
      return nn_loss, nn_rmse

  def evaluate_model(self, evaluation=False):
    set_model = self.optimize_model()
    _, _ = self.fit_nn(model=set_model, evaluation=evaluation)
    nn_pred = set_model.predict(self.x_test)
    nn_pred_df = set_model.predict(self.df_pred)
    return nn_pred, nn_pred_df

  def plot_results(self, epochs = 100):
    _, _, _, y_pred,_ = self.evaluate_model(epochs=epochs)
    plt.figure(figsize=(11,7))
    plt.scatter(self.y_test, y_pred)
    plt.xlabel("True C test values")
    plt.ylabel("Predictions on the test set")
    plt.title("NN model results")
    plt.grid(False)
    plt.show()

class NNP(NN):
  def __init__(self, x_train, x_test, y_train, y_test, df_pred, inp_shape,  activation:str=None, lr=None):
    NN.__init__ (self, x_train, x_test, y_train, y_test, df_pred, inp_shape, activation, lr)
    
  def predict_nn(self, df_test):
    model = self.optimize_model()
    _, _ = self.fit_nn(model=model, evaluation=False)
    y_pred = model.predict(df_test)
    return y_pred

if __name__ == '__main__':
    print('ANN model called successfully')
