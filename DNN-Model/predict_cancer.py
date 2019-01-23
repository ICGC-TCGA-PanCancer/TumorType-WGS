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
import argparse 


import keras
from keras import backend as K 
from keras.models import Sequential 
from keras.layers import InputLayer, Input
from keras.layers import Dropout
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model 
from keras import regularizers


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Predict cancer type from mutation topology and mutation types', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input_file', dest='input_file', required=True, type=str)
	parser.add_argument('--output', required=True)

	d = pd.read_csv('/Users/gatwal/Desktop/crossvalidation_results/fold1_d.csv')
	#factor = d.Factor 
	#cancer = d.Cancer 
	#factor_dict = dict(zip(factor, cancer))
	factor_dict = {0: 'Bone-Osteosarc',
 1: 'Breast-AdenoCA',
 2: 'CNS-GBM',
 3: 'CNS-Medullo',
 4: 'CNS-PiloAstro',
 5: 'ColoRect-AdenoCA',
 6: 'Eso-AdenoCA',
 7: 'Head-SCC',
 8: 'Kidney-ChRCC',
 9: 'Kidney-RCC',
 10: 'Liver-HCC',
 11: 'Lung-AdenoCA',
 12: 'Lung-SCC',
 13: 'Lymph-BNHL',
 14: 'Lymph-CLL',
 15: 'Myeloid-MPN',
 16: 'Ovary-AdenoCA',
 17: 'Panc-AdenoCA',
 18: 'Panc-Endocrine',
 19: 'Prost-AdenoCA',
 20: 'Skin-Melanoma',
 21: 'Stomach-AdenoCA',
 22: 'Thy-AdenoCA',
 23: 'Uterus-AdenoCA'}
	model = load_model('/Users/gatwal/Desktop/ensemble/ensemble_model.keras')
	args = parser.parse_args()
	input_file = args.input_file
	output = args.output	
	data = pd.read_csv(input_file, index_col = [0])
	x_input = data.values 
	if x_input.shape[-1] != 3047:
		raise IOerror('Input file not of correct dimensionality. See README for properly formatted file')
	predictions = model.predict(x_input)
	class_predictions = np.argmax(predictions, axis = 1)
	### make predictions dataframe
	predictions_df = pd.DataFrame(data = predictions, index = data.index, columns = d.Cancer)
	predictions_df.to_csv('%s/cancer_prediction_probability.txt'%output, sep = '\t')
	cancer_classes = [factor_dict[i] for i in class_predictions]
	class_df = pd.DataFrame({'cancer_prediction':cancer_classes}, index = data.index)
	class_df.to_csv('%s/cancer_predictions.txt'%output, sep = '\t')


