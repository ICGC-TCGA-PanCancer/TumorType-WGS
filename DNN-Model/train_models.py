import tensorflow as tf
import random as rn
import numpy as np 
import os 
os.environ['PYTHONHASHSEED'] = '0'
# Setting the seed for numpy-generated random numbers
np.random.seed(45)

# Setting the graph-level random seed.
tf.set_random_seed(1337)

rn.seed(73)

from keras import backend as K 

session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)

#Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
import math 
import pandas as pd

import keras
from keras import backend as K 
from keras.models import Sequential 
from keras.layers import InputLayer, Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import TensorBoard, EarlyStopping 
from keras.optimizers import Adam, Adamax
from keras.models import load_model 
from keras import regularizers

import skopt 
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer 
from skopt.plots import plot_convergence 
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import sys


dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')

dim_weight_decay = Real(low=1e-3, high = 0.5, prior = 'log-uniform', name='weight_decay')

dim_num_dense_layers = Integer(low=0, high = 5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=1024, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'softplus'], name = 'activation')

dim_dropout = Real(low=1e-6, high=0.5, prior = 'log-uniform', name = 'dropout')

dimensions= [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]

default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100, 'relu']

def log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation): 
	s = "./crossvalidation%s_logs/lr_{0:.0e}_wd_{0:.0e}_layers_{2}_nodes{3}_{4}/"%fold
	log_dir = s.format(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
	return log_dir 

### Make train test and validaiton here 




def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation): 
	###Define model here 
	model = Sequential()
	model.add(InputLayer(input_shape = (input_size,)))
	for i in range(num_dense_layers): 
		name = 'layer_dense_{0}'.format(i+1)
		model.add(Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(Dropout(dropout))
	model.add(Dense(num_classes, activation='softmax'))
	#optimizer = Adam(lr=learning_rate)
	optimizer = Adam(lr=learning_rate)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	#callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
	return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation): 
	global best_accuracy
	#best_accuracy = 0.0
	print('learning rate: ',learning_rate)
	print('weight_decay: ', weight_decay)
	print('dropout', dropout)
	print('num_dense_layers: ', num_dense_layers)
	print('num_dense_nodes: ', num_dense_nodes)
	print('activation: ', activation)
	print() 
	model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout, num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation)
	log_dir = log_dir_name(learning_rate, weight_decay, num_dense_layers,
                           num_dense_nodes, activation)
	callback_log = TensorBoard(
							log_dir=log_dir,
							histogram_freq=0,
							batch_size=32,
							write_graph=True,
							write_grads=True,
							write_images=False)
	callbacks = [callback_log]
	history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=32, validation_data=validation_data, callbacks=callbacks)
	accuracy = history.history['val_acc'][-1]
	print()
	print('Accuracy: {0:.2%}'.format(accuracy))
	if accuracy > best_accuracy: 
		model.save(path_best_model)
		best_accuracy = accuracy 
	del model 
	K.clear_session() 
	return -accuracy 


if __name__ == '__main__': 
	fold = int(sys.argv[1])
	feature_type = sys.argv[2]

	path_best_model = './crossvalidation%s_best_model.keras'%fold
	best_accuracy = 0.0 
	if feature_type == '1':
		data = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/pcawg_mutations_complete.csv', index_col = [0])
	elif feature_type == '2': 
		data = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/pcawg_data_genes_complete.csv', index_col = [0])
	elif feature_type == '3': 
		data = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/pcawg_genes_only.csv', index_col = [0])
	elif feature_type == '4': 
		data = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/pcawg_mutation_distribution.csv', index_col = [0])
	elif feature_type == '5': 
		data = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/pcawg_mutations_types.csv', index_col = [0])

	### Making training, test, validation data 
	training_samples = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/training_idx_pcawg.csv', index_col=[0])
	training_samples.columns = ['guid', 'split']
	training_samples = training_samples[training_samples.split == fold]
	training_data = data[data['guid'].isin(training_samples.guid)]
	validation_samples = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/validation_idx_pcawg.csv', index_col=[0])
	validation_samples.columns = ['guid', 'split']
	validation_samples = validation_samples[validation_samples.split == fold]
	validation_data = data[data['guid'].isin(validation_samples.guid)]
	test_samples = pd.read_csv('/Users/gatwal/Desktop/SpatialSigs/test_idx_pcawg.csv', index_col=[0])
	test_samples.columns = ['guid', 'split']
	test_samples = test_samples[test_samples.split == fold]
	test_data = data[data['guid'].isin(test_samples.guid)]

	training_data = training_data.drop(['guid'], axis = 1)
	validation_data = validation_data.drop(['guid'], axis = 1)
	test_data = test_data.drop(['guid'], axis = 1)

	x_train = training_data.values
	y_train = training_data.index
	x_val = validation_data.values
	y_val = validation_data.index
	x_test = test_data.values
	y_test = test_data.index 

	encoder = LabelEncoder()
	test_labels_names = y_test
	y_test = encoder.fit_transform(y_test)
	test_labels = y_test
	y_test = keras.utils.to_categorical(y_test, 24)
	y_train = encoder.fit_transform(y_train)
	y_train = keras.utils.to_categorical(y_train, 24)
	y_val = encoder.fit_transform(y_val)
	y_val = keras.utils.to_categorical(y_val, 24)

	validation_data = (x_val, y_val)


	input_size = x_train.shape[1]
	num_classes = 24

	### Run Bayesian optimization 
	search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=200, x0=default_paramaters, random_state=7, n_jobs=-1)

	# Save Best Hyperparameters
	hyps = np.asarray(search_result.x)
	np.save('./crossvalidation_results/fold%s_hyperparams'%fold, hyps, allow_pickle=False)
	model = load_model(path_best_model)
	# Evaluate best model on test data
	result = model.evaluate(x=x_test, y=y_test)
	# Save best model 
	model.save('./crossvalidation_results/fold_%s_model.keras'%fold)

	Y_pred = model.predict(x_test)
	y_pred = np.argmax(Y_pred, axis = 1)


	a = pd.Series(test_labels_names)
	b = pd.Series(test_labels)
	d = pd.DataFrame({'Factor':b, 'Cancer':a})
	d = d.drop_duplicates()
	d = d.sort_values('Factor') 

	## Create array of prediction probabilities 
	p = model.predict_proba(x_test)
	p_df = pd.DataFrame(data = p, columns = d.Cancer, index = test_labels_names)

	## Generate Confusion Matrix
	c_matrix = confusion_matrix(test_labels, y_pred)
	c_df = pd.DataFrame(data=c_matrix, index=d.Cancer, columns = d.Cancer)

	## Generate Class Report

	c_report = classification_report(test_labels, y_pred, digits=3)
		r = to_table(c_report)
	cols = r[0]
	r = pd.DataFrame(data=r[1:-1], columns = cols, index = d.Cancer)

	samples = []
	for i in r.index: 
		l = len(data[data.index == i])
		samples.append(l)

	r['sample_size'] = samples
	r.columns = [x.replace('-', '_') for x in r.columns]
	r['f1_score'] = [float(x) for x in r.f1_score]
	#r.to_csv('./class_report_fold%s.csv'%fold)
	#c_df.to_csv('./confusion_matrix_fold_%s.csv'%fold)
	#p_df.to_csv('./probability_classification_%s.csv'%fold)







