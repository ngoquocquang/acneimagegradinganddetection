from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from numpy import argmax

# load models from file
def load_all_models(n_models, load_model_path):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = load_model_path + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			# make not trainable
			layer.trainable = False
			# rename to avoid 'unique layer name' issue
			layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = Concatenate()(ensemble_outputs)
	hidden = Dense(8, activation='relu')(merge)
	output = Dense(4, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def multiple_generator_input(generator):
	while True:
		x = generator.next()
		yield [x[0],x[0],x[0]], x[1]

# fit a stacked model
def fit_stacked_model(model, train_generator, val_generator, epochs, path_model, checkpoint_filepath):
	# prepare input data
	X=multiple_generator_input(train_generator)
	val=multiple_generator_input(val_generator)
	# load wight_model
	try:
		model.load_weights(checkpoint_filepath)
		print('Loaded weights!')
	except:
		pass
	# fit model
	history = model.fit_generator(X, steps_per_epoch=60, epochs= epochs,
											validation_data= val, validation_steps=30,
											verbose= 1, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor= 0.5,
											patience=5, min_lr=0.000001), ModelCheckpoint(filepath=checkpoint_filepath, verbose = 1,
																						  save_weights_only=True,monitor='val_accuracy',
																						  mode='max',save_best_only=True)], use_multiprocessing= False, shuffle=True)
	model.save(path_model)
	return history

# make a prediction with a stacked model
def predict_stacked_model(model, test_generator, count):
	# prepare input data
	X = multiple_generator_input(test_generator)
	# make prediction
	return model.predict_generator(X, steps = count)

