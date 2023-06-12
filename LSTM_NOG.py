## LSTM Implementation with following
# Input Gate
# Output gate
# Forget gate
# PeepHole Connections


import keras
from keras import backend
from keras import activations
from keras import constraints
from keras import initializers
from keras import regularizers
import tensorflow as tf




class LSTM_NOG(keras.layers.Layer):

	def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
			   
               **kwargs):
		self.units = units		
		self.state_size = [self.units, self.units]
		self.unit_forget_bias = unit_forget_bias

		super(LSTM_NOG, self).__init__(**kwargs)

		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)

		self.kernel_initializer = initializers.get(kernel_initializer)

		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.kernel_constraint  = constraints.get(kernel_constraint)


		self.recurrent_initializer = initializers.get(recurrent_initializer)
		self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		self.recurrent_constraint  = constraints.get(recurrent_constraint)

		self.bias_initializer = initializers.get(bias_initializer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.bias_constraint  = constraints.get(bias_constraint)
        
        

	def build(self, input_shape):

		input_dim = input_shape[-1]

        # Initialize input weight tensor
		self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        # Initialize recurrent weight tensor
		self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                        name='recurrent_kernel',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)


		if self.unit_forget_bias:
			def bias_initializer(_, *args, **kwargs):
				return backend.concatenate([
				  self.bias_initializer((self.units,), *args, **kwargs),
				  initializers.get('ones')((self.units,), *args, **kwargs),
				  self.bias_initializer((self.units * 2,), *args, **kwargs),
				])
		else:
			bias_initializer = self.bias_initializer

		self.bias = self.add_weight(shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          )


		print("built")

		self.built = True

	def _compute_carry_and_output(self, x, h_tm1, c_tm1):
		"""Computes carry and output using split kernels."""
		x_i, x_f, x_c, x_o = x
		h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
		i = self.recurrent_activation(
		    x_i + backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))

		f = self.recurrent_activation(x_f + backend.dot(
		    h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
		
		
		
		c = f * c_tm1 + i * self.activation(x_c + backend.dot(
			h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
		
		o = self.recurrent_activation(
		    x_o + backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))

		return c, o

	def call(self, inputs, states, training=None):

		#print("------",states)
		h_tm1 = states[0]  # previous memory state
		c_tm1 = states[1]  # previous carry state

		

	    ##
	   	## No Dropout is considered for now
	    ##

	  	## if self.implementation == 1: This is Larger Number of small operations, consider this mode default 
      
		inputs_i = inputs
		inputs_f = inputs
		inputs_c = inputs
		inputs_o = inputs


		k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
		x_i = backend.dot(inputs_i, k_i)
		x_f = backend.dot(inputs_f, k_f)
		x_c = backend.dot(inputs_c, k_c)
		x_o = backend.dot(inputs_o, k_o)

		#if self.use_bias: Default Use_bias
		b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
		x_i = backend.bias_add(x_i, b_i)
		x_f = backend.bias_add(x_f, b_f)
		x_c = backend.bias_add(x_c, b_c)
		x_o = backend.bias_add(x_o, b_o)

		# recurrent dropout is not considered
		h_tm1_i = h_tm1
		h_tm1_f = h_tm1
		h_tm1_c = h_tm1
		h_tm1_o = h_tm1

		x = (x_i, x_f, x_c, x_o)
		h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
		#print("Line 172 in Vanilla LSTM","x")
		c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

		# No Output gate
		h = self.activation(c)  


		return h, [h, c]
