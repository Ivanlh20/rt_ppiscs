#-*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:30:14 1615

__author__ = "Ivan Lobato"
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import io
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#########################################################################################
import tensorflow as tf

#########################################################################################
N_X = 9
N_Y = 87

MODV = 10
LR_0 = 1e-5
BATCH_SIZE = 256
N_LAYS = 13
FTRS = 160
ACT_STR = 'swish'

SCS_MIN = 1e-7
CSTR_DR_DET = 0.001 # mrad

lambda_1_n = 1.0e2
lambda_2_n = 2.0e4
lambda_1_log = 1.6e1
lambda_1_n_cstr = 1.0e2

#  0      1       2    3      4               5                   6               7         8
# [Z, zone_axis, E_0, c_30, c_10, cond_lens_outer_aper_ang, det_inner_ang, det_outer_ang, rmsd]
X_SFT = np.array([50.0, 0.5, 143.0, 0.00, 0.00, 26.0, 99.0, 177.0, 1.1e-1], np.float32).reshape((9))
X_SC = np.array([27.00, 0.5, 91.00, 6e-4, 58.0, 5.00, 52.0, 52.00, 2.3e-2], np.float32).reshape((9))

#########################################################################################
#########################################################################################
def fcn_plot_to_image(figure):
	buf = io.BytesIO()
	plt.savefig(buf, format='png')
	plt.close(figure)
	buf.seek(0)
	image = tf.image.decode_png(buf.getvalue(), channels=3)
	image = tf.expand_dims(image, 0)
	return image

#########################################################################################
###################################### database #########################################
#########################################################################################
def fcn_ds_parse_sgl_ex_string_string(record_string):
	features = {
		'x': tf.io.FixedLenFeature([], tf.string),
		'y': tf.io.FixedLenFeature([], tf.string)
	}

	return tf.io.parse_single_example(record_string, features)

def fcn_string_dc(x, x_typ_d, x_typ_c):
	x = tf.io.decode_raw(x, x_typ_d)
	x = tf.cast(x, x_typ_c)
	return x

@tf.function
def fcn_ds_map_gen(record_string, bb_norm, bb_aug):
	data = fcn_ds_parse_sgl_ex_string_string(record_string)

	x = fcn_string_dc(data['x'], tf.float64, tf.float32)
	x = tf.reshape(x, [N_X])

	y = fcn_string_dc(data['y'], tf.float64, tf.float32)
	y = tf.reshape(y, [N_Y])

	if bb_norm:
		x = (x - X_SFT)/X_SC

	return x, y

def fcn_ds_train_dataset(filename, is_training, fcn_map, parm):
	dataset = tf.data.TFRecordDataset(filename)
	dataset = dataset.map(fcn_map, num_parallel_calls=2)
	dataset = dataset.repeat(None)
	
	if is_training:
		dataset = dataset.shuffle(64*BATCH_SIZE, reshuffle_each_iteration=True)

	dataset = dataset.batch(BATCH_SIZE, drop_remainder=is_training)
	dataset = dataset.prefetch(buffer_size=64)

	return dataset

def fcn_ds_map_train(fcn_map):
	def fcn_map_t(record_string):
		return fcn_map(record_string, tf.constant(True), tf.constant(True))

	return fcn_map_t

def fcn_ds_map_val(fcn_map):
	def fcn_map_t(record_string):
		return fcn_map(record_string, tf.constant(True), tf.constant(False))

	return fcn_map_t

def fcn_load_datasets(fcn_ds_map, parm):
	train_dataset = fcn_ds_train_dataset(parm['train_db_path'], True, fcn_ds_map_train(fcn_ds_map), parm)
	val_dataset = fcn_ds_train_dataset( parm['val_db_path'], False, fcn_ds_map_val(fcn_ds_map), parm)

	return train_dataset, val_dataset

#########################################################################################
####################################### losses ##########################################
#########################################################################################
def fcn_x_cstr(x):
	r = tf.random.uniform((tf.shape(x)[0], 1), 0.0, 1.0, dtype=tf.float32)

	inner = tf.expand_dims(x[:, 6], axis=-1)
	outer = tf.expand_dims(x[:, 7], axis=-1)

	inner_m = inner*X_SC[6] + X_SFT[6] + CSTR_DR_DET
	outer_m = outer*X_SC[7] + X_SFT[7] - CSTR_DR_DET
	det_mrad = inner_m + r*(outer_m - inner_m)
	inner_m = (det_mrad - X_SFT[6])/X_SC[6]
	outer_m = (det_mrad - X_SFT[7])/X_SC[7]

	rmsd = tf.expand_dims(x[:, -1], axis=-1)

	x_1 = tf.concat([x[:, 0:6], inner, outer_m, rmsd], axis = -1)
	x_2 = tf.concat([x[:, 0:6], inner_m, outer, rmsd], axis = -1)

	x_cstr = tf.concat([x, x_1, x_2], axis = 0)

	return x_cstr

def fcn_g_y_pred_y_cstr(y_pred):
	y_p, y_cstr_a, y_cstr_b = tf.split(y_pred, 3, axis=0)

	y_cstr = y_cstr_a + y_cstr_b

	return y_p, y_cstr

def fcn_g_mk(y_true):
	return tf.math.greater(y_true, SCS_MIN)

def fcn_norm(y_true, y_pred):
	sc = tf.reduce_max(y_true, axis=-1, keepdims=True)
	y_true = y_true/sc
	y_pred = y_pred/sc
	return y_true, y_pred

def fcn_l1(y_true, y_pred, mk):
	loss = tf.where(mk, tf.math.abs(y_true-y_pred), 0)
	loss = tf.reduce_sum(loss, axis=-1)
	return tf.reduce_mean(loss)

def fls_l1_n(y_true, y_pred):
	mk = fcn_g_mk(y_true)
	y_true, y_pred = fcn_norm(y_true, y_pred)
	return fcn_l1(y_true, y_pred, mk)

def fls_l2_n(y_true, y_pred):
	mk = fcn_g_mk(y_true)
	y_true, y_pred = fcn_norm(y_true, y_pred)

	loss = tf.where(mk, tf.math.squared_difference(y_true, y_pred), 0)
	loss = tf.reduce_sum(loss, axis=-1)
	return tf.reduce_mean(loss)

def fls_l1_log(y_true, y_pred):
	mk = fcn_g_mk(y_true)
	y_true = tf.math.log(y_true)
	y_pred = tf.math.log(y_pred)
	return fcn_l1(y_true, y_pred, mk)

def fls_l1_n_cstr(y_true, y_pred):
	mk = fcn_g_mk(y_true)
	y_true, y_pred = fcn_norm(y_true, y_pred)
	return fcn_l1(y_true, y_pred, mk)

#########################################################################################
#########################################################################################
def fls_t_w(y_t, y_p, y_cstr):
	loss_l1_n = fls_l1_n(y_t, y_p)
	loss_l1_n_w = lambda_1_n*loss_l1_n

	loss_l2_n = fls_l2_n(y_t, y_p)
	loss_l2_n_w = lambda_2_n*loss_l2_n

	loss_l1_log = fls_l1_log(y_t, y_p)
	loss_l1_log_w = lambda_1_log*loss_l1_log

	loss_l1_n_cstr = fls_l1_n_cstr(y_t, y_cstr)
	loss_l1_n_cstr_w = lambda_1_n_cstr*loss_l1_n_cstr

	loss_t = loss_l1_n_w + loss_l2_n_w + loss_l1_log_w + loss_l1_n_cstr

	return {'loss_t': loss_t, \
			'loss_l1_n_w': loss_l1_n_w, 'loss_l1_n': loss_l1_n, \
			'loss_l2_n_w': loss_l2_n_w, 'loss_l2_n': loss_l2_n, \
			'loss_l1_log_w': loss_l1_log_w, 'loss_l1_log': loss_l1_log, \
			'loss_l1_n_cstr_w': loss_l1_n_cstr_w, 'loss_l1_n_cstr': loss_l1_n_cstr}

#########################################################################################
######################################### metrics #######################################
#########################################################################################
def fmet_l1_n(y_true, y_pred):
	mk = fcn_g_mk(y_true)
	y_true, y_pred = fcn_norm(y_true, y_pred)
	return fcn_l1(y_true, y_pred, mk)

#########################################################################################
######################################## model ##########################################
#########################################################################################
class Cb_Lr_schedule_base:
	def __init__(self, lr_min, lr_max, m_min, m_max, decay_steps, decay_rate, steps_per_cycle, warmup_steps, cooldown_steps, lr_0=0.0, warmup_steps_0=-1, cooldown_steps_0=-1, decay_steps_0=0, lrs_m_pow=1.0, lrs_lr_pow=1.0):
		self.lrs_lr_min = lr_min
		self.lrs_lr_max = lr_max

		self.lrs_m_min = m_min
		self.lrs_m_max = m_max

		self.lrs_decay_steps = decay_steps
		self.lrs_decay_rate = decay_rate

		self.lrs_warmup_steps_c = warmup_steps
		self.lrs_cooldown_steps_c = cooldown_steps

		self.lrs_warmup_steps_0 = self.lrs_cycle_steps//3 if warmup_steps_0<0 else warmup_steps_0
		self.lrs_cooldown_steps_0 = self.lrs_cycle_steps//3 if cooldown_steps_0<0 else cooldown_steps_0
		self.lrs_decay_steps_0 = decay_steps if decay_steps_0<=0 else decay_steps_0
		self.lrs_decay_steps_0 = np.maximum(self.lrs_warmup_steps_0 + self.lrs_cooldown_steps_0, self.lrs_decay_steps_0)

		self.lrs_cycle_steps = steps_per_cycle
		self.lrs_cycle_steps_0 = self.lrs_decay_steps_0

		self.lrs_m_itf_0 = 0
		self.lrs_m_itf_e = 0
		self.lrs_m_dt_itf = 0

		self.lrs_m_pow = lrs_m_pow

		self.lrs_lr_0 = np.minimum(self.lrs_lr_min/100.0, 1e-8) if lr_0<1e-12 else lr_0
		self.lrs_lr_pow = lrs_lr_pow

		self.lrs_lr_itf_0 = 0
		self.lrs_lr_itf_e = 0
		self.lrs_lr_dt_itf = 0

		self.lrs_global_ib = 0
		self.lrs_cycle_ib = 0

		self.lrs_warmup_steps = 0
		self.lrs_cooldown_steps = 0

		self.lrs_cycle_ib_wup_ul = 0 # upper limit
		self.lrs_cycle_ib_cdw_ll = 0 # lower limit

	@classmethod
	def from_parm(cls, parm):
		return cls(
		lr_min=parm['lr_min'],
		lr_max=parm['opt_lr'],
		m_min=parm['m_0'],
		m_max=parm['opt_m'],
		decay_steps=parm['decay_steps'],
		decay_rate=parm['decay_rate'],
		steps_per_cycle=parm['decay_steps'],
		warmup_steps=parm['warmup_steps'],
		cooldown_steps=parm['cooldown_steps'],
		lr_0=parm['lr_0'], 
		warmup_steps_0=parm['warmup_steps_0'], 
		cooldown_steps_0=parm['cooldown_steps_0'])

	def fcn_set_lr_tf_parm(self, lr_0, lr_e, n_iter):
		self.lrs_lr_itf_0 = np.power(lr_0, 1.0/self.lrs_lr_pow)
		self.lrs_lr_itf_e = np.power(lr_e, 1.0/self.lrs_lr_pow)
		self.lrs_lr_dt_itf = (self.lrs_lr_itf_e - self.lrs_lr_itf_0)/n_iter

	def fcn_set_m_tf_parm(self, m_0, m_e, n_iter):
		self.lrs_m_itf_0 = np.power(m_0, 1.0/self.lrs_m_pow)
		self.lrs_m_itf_e = np.power(m_e, 1.0/self.lrs_m_pow)
		self.lrs_m_dt_itf = (self.lrs_m_itf_e - self.lrs_m_itf_0)/n_iter

	def fcn_set_lr_steps_parm(self):
		if self.lrs_global_ib == 0:
			self.lrs_warmup_steps = self.lrs_warmup_steps_0
			self.lrs_cooldown_steps = self.lrs_cooldown_steps_0
		else:
			self.lrs_warmup_steps = self.lrs_warmup_steps_c
			self.lrs_cooldown_steps = self.lrs_cooldown_steps_c

		self.lrs_cycle_ib_wup_ul = self.lrs_warmup_steps
		self.lrs_cycle_ib_cdw_ll = np.maximum(0, self.lrs_cycle_steps - self.lrs_cooldown_steps)

	def fcn_lr_min_exp_decay_eval(self, ig):
		p = np.floor(ig/self.lrs_decay_steps)

		if ig == 0:
			lr_min = self.lrs_lr_0
		else:
			lr_min = self.lrs_lr_min if self.lrs_warmup_steps_c>0 else self.lrs_lr_max
			lr_min = lr_min*np.power(self.lrs_decay_rate, p)

		return lr_min

	def fcn_lr_exp_decay_eval(self, ig):
		p = np.floor(ig/self.lrs_decay_steps)

		lr = self.lrs_lr_max*np.power(self.lrs_decay_rate, p)

		return lr

	def fcn_lr_tf_eval(self, ib):
		lr = np.power(self.lrs_lr_itf_0 + ib*self.lrs_lr_dt_itf, self.lrs_lr_pow)

		return lr

	def fcn_m_tf_eval(self, ib):
		lr = np.power(self.lrs_m_itf_0 + ib*self.lrs_m_dt_itf, self.lrs_m_pow)

		return lr

	def set_opt_iter(self, opt, iter=0):
		tf.keras.backend.set_value(opt.iterations, iter)

	def get_lr_m(self):
		lr = 0
		m = 0

		if self.lrs_cycle_ib % self.lrs_cycle_steps == 0:
			self.lrs_cycle_ib = 0
			self.fcn_set_lr_steps_parm()

		if (self.lrs_warmup_steps > 0) and (self.lrs_cycle_ib <= self.lrs_cycle_ib_wup_ul):
			if self.lrs_cycle_ib == 0:
				lr_min = self.fcn_lr_min_exp_decay_eval(self.lrs_global_ib)
				lr_max = self.fcn_lr_exp_decay_eval(self.lrs_global_ib)

				self.fcn_set_lr_tf_parm(lr_min, lr_max, self.lrs_warmup_steps)
				self.fcn_set_m_tf_parm(self.lrs_m_min, self.lrs_m_max, self.lrs_warmup_steps)

				lr = lr_min

				print('\n warming up the lr from {:.4e} to {:.4e} and m from {} to {:.4f} for {:.4f} batches\n'.format(
					lr_min, lr_max, self.lrs_m_min, self.lrs_m_max, self.lrs_warmup_steps))
			else:
				lr = self.fcn_lr_tf_eval(self.lrs_cycle_ib)
				m = self.fcn_m_tf_eval(self.lrs_cycle_ib)

		elif self.lrs_cycle_ib < self.lrs_cycle_ib_cdw_ll:
			lr = self.fcn_lr_exp_decay_eval(self.lrs_global_ib)
			m = self.lrs_m_max

			if self.lrs_cycle_ib == self.lrs_cycle_ib_wup_ul+1:
				print('\n training with lr = {:.4e} and m = {:.4f} for {} batches\n'.format(
					lr, m, self.lrs_cycle_ib_cdw_ll-self.lrs_cycle_ib_wup_ul))

		elif self.lrs_cooldown_steps > 0:
			if self.lrs_cycle_ib == self.lrs_cycle_ib_cdw_ll:
				lr_min = self.fcn_lr_min_exp_decay_eval(self.lrs_global_ib + self.lrs_cooldown_steps + 1)
				lr_max = self.fcn_lr_exp_decay_eval(self.lrs_global_ib)

				self.fcn_set_lr_tf_parm(lr_max, lr_min, self.lrs_cooldown_steps)

				lr = lr_max
				m = self.lrs_m_max

				print('\n cooling down the lr from {:.4e} to {:.4e} with m = {:.4f} for {:.4f} batches\n'.format(
					lr_max, lr_min, m, self.lrs_cooldown_steps))
			else:
				lr = self.fcn_lr_tf_eval(self.lrs_cycle_ib - self.lrs_cycle_ib_cdw_ll)
				m = self.lrs_m_max

		return lr, m

	def set_opt_lr_m(self, opt, lr, m):
		# set learning rate and momentum values
		nfop.fcn_set_opt_lr(opt, lr)
		nfop.fcn_set_opt_m(opt, m)

	def get_set_opt_lr_m(self, opt):
		lr, m = self.get_lr_m()
		self.set_opt_lr_m(opt, lr, m)

	def inc_counter(self):
		self.lrs_global_ib += 1
		self.lrs_cycle_ib += 1

class My_model(tf.keras.Model):
	def __init__(self, *args, **kwargs):
		super(My_model, self).__init__(*args, **kwargs)

		self.loss_t = tf.keras.metrics.Mean(name="loss_t")
		self.val_loss_t = tf.keras.metrics.Mean(name="loss_t")

		self.met_l1_n = tf.keras.metrics.Mean(name="met_l1_n")
		self.val_met_l1_n = tf.keras.metrics.Mean(name="met_l1_n")

		self.loss_l1_n_w = tf.keras.metrics.Mean(name="loss_l1_n_w")
		self.loss_l1_n = tf.keras.metrics.Mean(name="loss_l1_n")
		self.val_loss_l1_n_w = tf.keras.metrics.Mean(name="loss_l1_n_w")
		self.val_loss_l1_n = tf.keras.metrics.Mean(name="loss_l1_n")

		self.loss_l2_n_w = tf.keras.metrics.Mean(name="loss_l2_n_w")
		self.loss_l2_n = tf.keras.metrics.Mean(name="loss_l2_n")
		self.val_loss_l2_n_w = tf.keras.metrics.Mean(name="loss_l2_n_w")
		self.val_loss_l2_n = tf.keras.metrics.Mean(name="loss_l2_n")

		self.loss_l1_log_w = tf.keras.metrics.Mean(name="loss_l1_log_w")
		self.loss_l1_log = tf.keras.metrics.Mean(name="loss_l1_log")
		self.val_loss_l1_log_w = tf.keras.metrics.Mean(name="loss_l1_log_w")
		self.val_loss_l1_log = tf.keras.metrics.Mean(name="loss_l1_log")

		self.loss_l1_n_cstr_w = tf.keras.metrics.Mean(name="loss_l1_n_cstr_w")
		self.loss_l1_n_cstr = tf.keras.metrics.Mean(name="loss_l1_n_cstr")
		self.val_loss_l1_n_cstr_w = tf.keras.metrics.Mean(name="loss_l1_n_cstr_w")
		self.val_loss_l1_n_cstr = tf.keras.metrics.Mean(name="loss_l1_n_cstr")

	def compile(self, parm):
		super(My_model, self).compile()

		self.opt = tf.keras.optimizers.Adam(learning_rate=parm['opt_lr'], beta_1=0.9, beta_2=0.099, epsilon=1e-7)

		self.lr_schedule_gen = Cb_Lr_schedule_base(
			lr_min=1e-8,
			lr_max=parm['opt_lr'],
			m_min=0.0,
			m_max=0.9,
			decay_steps=parm['decay_steps'],
			decay_rate=parm['decay_rate'],
			steps_per_cycle=parm['decay_steps'],
			warmup_steps=0,
			cooldown_steps=0,
			lr_0=parm['opt_lr'], 
			warmup_steps_0=parm['warmup_steps'], 
			cooldown_steps_0=0,
			decay_steps_0=parm['decay_steps'],
			lrs_m_pow=1.0,
			lrs_lr_pow=1.0)

	def reset_opt_iter(self):
		tf.keras.backend.set_value(self.opt.iterations, 0)

	def set_opt_lr_m(self):
		self.lr_schedule_gen.get_set_opt_lr_m(self.opt)

	def inc_opt_counter(self):
		self.lr_schedule_gen.inc_counter()

	@tf.function
	def train_step(self, data):
		x, y = data

		x = fcn_x_cstr(x)

		with tf.GradientTape() as tape:
			# Forward pass
			y_pred = self(x, training=True)

			# y_p and y_cstr
			y_pred, y_cstr = fcn_g_y_pred_y_cstr(y_pred)

			# Compute our own loss
			loss_dict = fls_t_w(y, y_pred, y_cstr)

			# regularization loss
			loss_reg = tf.reduce_sum(self.losses)

			loss_t = loss_dict['loss_t'] + loss_reg		

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss_t, trainable_vars)

		# Update weights
		self.opt.apply_gradients(zip(gradients, trainable_vars))

		# Compute own metrics
		self.loss_t.update_state(loss_t)
		metrics_out = {'loss_t': self.loss_t.result()}

		self.met_l1_n.update_state(fmet_l1_n(y, y_pred))
		metrics_out.update({'met_l1_n': self.met_l1_n.result()})

		self.loss_l1_n_w.update_state(loss_dict['loss_l1_n_w'])
		self.loss_l1_n.update_state(loss_dict['loss_l1_n'])
		metrics_out.update({'loss_l1_n_w': self.loss_l1_n_w.result(), 'loss_l1_n': self.loss_l1_n.result()})

		self.loss_l2_n_w.update_state(loss_dict['loss_l2_n_w'])
		self.loss_l2_n.update_state(loss_dict['loss_l2_n'])
		metrics_out.update({'loss_l2_n_w': self.loss_l2_n_w.result(), 'loss_l2_n': self.loss_l2_n.result()})

		self.loss_l1_log_w.update_state(loss_dict['loss_l1_log_w'])
		self.loss_l1_log.update_state(loss_dict['loss_l1_log'])
		metrics_out.update({'loss_l1_log_w': self.loss_l1_log_w.result(), 'loss_l1_log': self.loss_l1_log.result()})

		self.loss_l1_n_cstr_w.update_state(loss_dict['loss_l1_n_cstr_w'])
		self.loss_l1_n_cstr.update_state(loss_dict['loss_l1_n_cstr'])
		metrics_out.update({'loss_l1_n_cstr_w': self.loss_l1_n_cstr_w.result(), 'loss_l1_n': self.loss_l1_n_cstr.result()})

		return metrics_out

	@tf.function
	def test_step(self, data):
		x, y = data

		x = fcn_x_cstr(x)

		# Forward pass
		y_pred = self(x, training=False)

		# y_p and y_cstr
		y_pred, y_cstr = fcn_g_y_pred_y_cstr(y_pred)

		# Compute our own loss
		loss_dict = fls_t_w(y, y_pred, y_cstr)

		# regularization loss
		loss_reg = tf.reduce_sum(self.losses)

		loss_t = loss_dict['loss_t'] + loss_reg

		# Compute own metrics
		self.val_loss_t.update_state(loss_t)
		metrics_out = {'loss_t': self.val_loss_t.result()}

		self.val_met_l1_n.update_state(fmet_l1_n(y, y_pred))
		metrics_out.update({'met_l1_n': self.val_met_l1_n.result()})

		self.val_loss_l1_n_w.update_state(loss_dict['loss_l1_n_w'])
		self.val_loss_l1_n.update_state(loss_dict['loss_l1_n'])
		metrics_out.update({'loss_l1_n_w': self.val_loss_l1_n_w.result(), 'loss_l1_n': self.val_loss_l1_n.result()})

		self.val_loss_l2_n_w.update_state(loss_dict['loss_l2_n_w'])
		self.val_loss_l2_n.update_state(loss_dict['loss_l2_n'])
		metrics_out.update({'loss_l2_n_w': self.val_loss_l2_n_w.result(), 'loss_l2_n': self.val_loss_l2_n.result()})

		self.val_loss_l1_log_w.update_state(loss_dict['loss_l1_log_w'])
		self.val_loss_l1_log.update_state(loss_dict['loss_l1_log'])
		metrics_out.update({'loss_l1_log_w': self.val_loss_l1_log_w.result(), 'loss_l1_log': self.val_loss_l1_log.result()})

		self.val_loss_l1_n_cstr_w.update_state(loss_dict['loss_l1_n_cstr_w'])
		self.val_loss_l1_n_cstr.update_state(loss_dict['loss_l1_n_cstr'])
		metrics_out.update({'loss_l1_n_cstr_w': self.val_loss_l1_n_cstr_w.result(), 'loss_l1_n_cstr': self.val_loss_l1_n_cstr.result()})

		return metrics_out

	@property
	def metrics(self):
		metrics_out = [self.loss_t, self.val_loss_t, self.met_l1_n, self.val_met_l1_n]

		metrics_out.extend([self.loss_l1_n_w, self.loss_l1_n, self.val_loss_l1_n_w, self.val_loss_l1_n])

		metrics_out.extend([self.loss_l2_n_w, self.loss_l2_n, self.val_loss_l2_n_w, self.val_loss_l2_n])

		metrics_out.extend([self.loss_l1_log_w, self.loss_l1_log, self.val_loss_l1_log_w, self.val_loss_l1_log])

		metrics_out.extend([self.loss_l1_n_cstr_w, self.loss_l1_n_cstr, self.val_loss_l1_n_cstr_w, self.val_loss_l1_n_cstr])

		return metrics_out

def fcn_net_v1(x, ftrs_i, ftrs_g, ftrs_o, n_lays, act_str, name, parm_norm=None, parm_init=None, parm_reg=None, parm_cstr=None):
	# https://arxiv.org/pdf/2304.08863.pdf
	init = tf.keras.initializers.HeUniform()

	name_dn = name + str(1)	
	x = tf.keras.layers.Dense(ftrs_i, activation=act_str, use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)

	n_lays_base = n_lays - 2
	for il in range(n_lays_base):
		name_dn = name + str(il+2)
		x_g = tf.keras.layers.Dense(ftrs_g, activation=act_str, use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)
		x = tf.keras.layers.Concatenate(axis=1, name=name_dn + '_concat')([x, x_g])

	name_dn = name + str(n_lays)
	x = tf.keras.layers.Dense(ftrs_o, activation='softplus', use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)

	return x

def fcn_net_v2(x, ftrs_i, ftrs_g, ftrs_o, n_lays, act_str, name, parm_norm=None, parm_init=None, parm_reg=None, parm_cstr=None):
	init = tf.keras.initializers.HeUniform()

	name_dn = name + str(1)	
	x = tf.keras.layers.Dense(ftrs_i, activation=act_str, use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)

	n_lays_base = n_lays - 2
	for il in range(n_lays_base):
		name_dn = name + str(il+2)
		x = tf.keras.layers.Dense(ftrs_g, activation=act_str, use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)

	name_dn = name + str(n_lays)
	x = tf.keras.layers.Dense(ftrs_o, activation='softplus', use_bias=True, kernel_initializer=init, bias_initializer='zeros', name=name_dn+'_den')(x)

	return x

def nn_model_vx(input_shape):
	modv = MODV
	n_lays = int(N_LAYS)
	ftrs_g = int(FTRS)
	ftrs_i = int(FTRS)
	ftrs_o = N_Y
	act_str = ACT_STR

	x_i = tf.keras.layers.Input(shape=input_shape, dtype='float32')
	
	if modv==10:
		x = fcn_net_v1(x_i, ftrs_i, ftrs_g, ftrs_o, n_lays, act_str, 'seq_')
	elif modv==20:
		x = fcn_net_v2(x_i, ftrs_i, ftrs_g, ftrs_o, n_lays, act_str, 'seq_')

	model = My_model(inputs=x_i, outputs=x, name='nn_model')

	return model

#########################################################################################
####################################### write image #####################################
#########################################################################################
class Cb_Test_Plot(tf.keras.callbacks.Callback):
	def __init__(self, parm): # add other arguments to __init__ if you need
		super().__init__()

		self.wi_log_dir = parm['log_dir']
		self.wi_update_freq = np.maximum(8, parm['test_update_freq'])
		self.wi_path_mat = parm['test_path']
		self.n_spl = parm['test_n_samples']
		self.parm = parm

		# spacing for plotting
		self.wspace = 0.14
		self.vspace = 0.13
		self.wsize_fig = 19.0

		self.file_writer_cm = tf.summary.create_file_writer(logdir=self.wi_log_dir, max_queue=1)

		matfile = sio.loadmat(self.wi_path_mat)
		x_mat = matfile['x'].astype(np.float32)
		y_mat = matfile['y'].astype(np.float32)

		# normalize x image
		self.n_spl =  np.minimum(self.n_spl, x_mat.shape[1])
		x_mat = x_mat[..., 0:self.n_spl].copy()
		x_mat = np.moveaxis(x_mat, 1, 0)
		x_mat = tf.convert_to_tensor(x_mat, tf.float32)
		x_mat = (x_mat - X_SFT[None, ...])/X_SC[None, ...]

		# normalize y image
		y_mat = y_mat[..., 0:self.n_spl].copy()
		y_mat = np.moveaxis(y_mat, 1, 0)
		y_mat = tf.convert_to_tensor(y_mat, tf.float32)

		self.wi_ds_x = tf.data.Dataset.from_tensor_slices(x_mat).batch(self.n_spl)

		self.wi_x = x_mat
		self.wi_y = y_mat

		self.wi_ibg = np.array(0).astype(np.int64)
		self.wi_ickp = np.array(0).astype(np.int64)

	def fcn_y_image_gen(self, y_t, y_p, opt_bb=True):
		if type(y_t) is not np.ndarray:
			y_t = y_t.numpy()

		if type(y_p) is not np.ndarray:
			y_p = y_p.numpy()

		y_p[y_t < SCS_MIN] = 0

		n_row = 3
		n_col = self.n_spl//n_row
		n_x = y_t.shape[1]
		x = np.arange(1, n_x+1, dtype=np.float32)

		hsize_fig = (self.wsize_fig-(n_col-1)*self.wspace)*n_row/(1.5*n_col)

		figure = plt.figure(figsize=(self.wsize_fig, hsize_fig))
		for iy in range(n_row):
			for ix in range(n_col):
				ixy = iy*n_col+ ix
				y_t_ixy = y_t[ixy, :]
				y_rt_ixy = y_p[ixy, :]
				ax = plt.subplot(n_row, n_col, ixy + 1)
				if iy==0:
					ax.set_title('scs vs # atoms', fontsize=12)

				if opt_bb:
					ax.plot(x, y_t_ixy, color='red', linestyle='-', linewidth=1.5)
					ax.plot(x, y_rt_ixy, color='blue', linestyle='-', linewidth=1.5)
					ax.set_xlim(0, n_x)
					ax.set_ylim(0, 1.1*np.max(y_t_ixy))
				else:
					y_ik = np.ones_like(x)
					y_ik[y_t_ixy < SCS_MIN] = 0.0
					ax.plot(x, y_ik, color='red', linestyle='-', linewidth=1.5)
					y_t_ixy[y_t_ixy < SCS_MIN] = 1.0
					ax.plot(x, y_rt_ixy/y_t_ixy, color='blue', linestyle='-', linewidth=1.5)
					ax.set_xlim(0, n_x)
					ax.set_ylim(0.925, 1.075)

				if iy<n_row-1:
					ax.set_xticklabels([])
					ax.set_xticks([])

				ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
				ax.grid(False)

		figure.subplots_adjust(hspace=self.vspace, wspace=self.wspace)

		return fcn_plot_to_image(figure)
	
	def on_train_begin(self, logs=None):
		self.model.reset_opt_iter()

	def on_train_batch_begin(self, batch, logs=None):
		self.model.set_opt_lr_m()

	def on_train_batch_begin(self, batch, logs=None):
		# saving learning rate and momentum
		if self.wi_ibg % self.parm['log_update_freq'] == 0:	
			with self.file_writer_cm.as_default():
				lr = tf.keras.backend.get_value(self.model.opt.learning_rate)
				m = tf.keras.backend.get_value(self.model.opt.beta_1)			
				tf.summary.scalar(name='batch_gen_lr', data=np.float32(lr), step=self.wi_ibg)
				tf.summary.scalar(name='batch_gen_m', data=np.float32(m), step=self.wi_ibg)

		# saving test data
		if self.wi_ibg % self.wi_update_freq == 0:
			y_p = self.model.predict(self.wi_ds_x)
			with self.file_writer_cm.as_default():
				tf.summary.image(name='y', data=self.fcn_y_image_gen(self.wi_y, y_p, True), step=self.wi_ibg)
				tf.summary.image(name='y_t', data=self.fcn_y_image_gen(self.wi_y, y_p, False), step=self.wi_ibg)
		
		# saving weights
		if self.wi_ibg % self.parm['log_checkpoint_freq'] == 0:
			self.model.save_weights(self.parm['checkpoint_path'].format(self.wi_ickp))

			self.wi_ickp += 1

		# reset metrics
		if (self.wi_ibg > 0) and (self.wi_ibg % self.parm['reset_metric_freq'] == 0):
			self.model.reset_metrics()

		self.wi_ibg += 1
		self.model.inc_opt_counter()

#########################################################################################
################################### compile and fit #####################################
#########################################################################################
def fcn_compile_and_fit_custom(parm):
	# clear session
	tf.keras.backend.clear_session()

	model = nn_model_vx(parm['input_shape'])
		
	if parm['weigths_load_path'] != '':
		model.load_weights(parm['weigths_load_path'], by_name=True, skip_mismatch=True)

	# print summary
	print(model.summary())

	# compile
	model.compile(parm)

	# set datasets
	train_dataset, val_dataset = fcn_load_datasets(fcn_ds_map_gen, parm)

	#fit
	model.fit(train_dataset,
		epochs=parm['epochs'],
		steps_per_epoch=parm['train_steps'],
		callbacks=parm['callbacks'],
		validation_data=val_dataset,
		validation_steps=parm['validation_steps'],
		validation_freq=1,
		initial_epoch=0,
		verbose=2)

#########################################################################################
#################################### initialization #####################################
#########################################################################################
def fcn_read_inputs():
	global MODV, LR_0, N_LAYS, FTRS, ACT_STR, BATCH_SIZE

	parser = argparse.ArgumentParser()
	parser.add_argument('modv', default=1, type=int, help='model version id')
	parser.add_argument('opt_lr', default=1e-3, type=float, help='optimizer learning rate')
	parser.add_argument('act_str', default='relu', type=str, help='activation string')
	parser.add_argument('bz', default=32, type=int, help='batch size')
	parser.add_argument('n_lays', default=13, type=int, help='# of layers')
	parser.add_argument('ftrs', default=160, type=float, help='# of filters')
	args = vars(parser.parse_args())

	parm = {'input_shape':(N_X, )}

	parm['modv'] = int(args['modv'])
	parm['opt_lr'] = args['opt_lr']
	parm['n_lays'] = int(args['n_lays'])
	parm['ftrs'] = int(args['ftrs'])
	parm['act_str'] = args['act_str']
	parm['batch_size'] = int(args['bz'])

	MODV = parm['modv']
	LR_0 = parm['opt_lr']
	N_LAYS = parm['n_lays']
	FTRS = parm['ftrs']
	ACT_STR = parm['act_str']
	BATCH_SIZE = parm['batch_size']

	return parm

def fcn_read_shape_parm(path):
	matfile = sio.loadmat(path)
	x_shape = matfile['x_shape'].squeeze().astype(np.int64)
	y_shape = matfile['y_shape'].squeeze().astype(np.int64)
	return x_shape, y_shape

def fcn_init():
	dir_db = 'dataset_scs'

	parm = fcn_read_inputs()

	parm['weigths_load_path'] = ""
	parm['root'] = r'E:\Neural_network\nt_scs_xtl_fcc\tensorflow'
	parm['train_db_dir'] = os.path.join(parm['root'], dir_db, 'train')
	parm['val_db_dir'] = os.path.join(parm['root'], dir_db, 'val')
	parm['train_db_path'] = os.path.join(parm['train_db_dir'], 'data.tfrecords')
	parm['val_db_path'] = os.path.join(parm['val_db_dir'], 'data.tfrecords')

	parm['test_path'] = os.path.join(parm['root'], dir_db, 'test', 'test.mat')
	parm['test_n_samples'] = 64

	parm['log_dir'] = os.path.join(parm['root'], 'scs_xtl_fcc', 'tf_training')
	parm['checkpoint_path'] = os.path.join(parm['log_dir'], 'cp-{:04d}.h5')

	# load train parameters
	path_parm = os.path.join(parm['train_db_dir'], 'parameters.mat')
	x_shape_train, y_shape_train = fcn_read_shape_parm(path_parm)

	path_parm = os.path.join(parm['val_db_dir'], 'parameters.mat')
	x_shape_val, y_shape_val = fcn_read_shape_parm(path_parm)

	parm['train_n_data'] = x_shape_train[0]
	parm['val_n_data']= x_shape_val[0]

	return	parm

def fcn_run(parm):
	parm['train_steps'] = parm['train_n_data']//parm['batch_size']
	parm['validation_steps'] = parm['val_n_data']//parm['batch_size']
	parm['decay_every_epochs'] = 2
	parm['decay_steps'] = round(parm['decay_every_epochs']*parm['train_steps'])
	parm['decay_rate'] = 0.95
	parm['epochs'] = 1000
	parm['test_update_freq'] = parm['validation_steps']//4

	parm['log_update_freq'] = 32
	parm['log_checkpoint_freq'] = parm['train_steps'] //4

	parm['reset_metric_freq'] = parm['log_update_freq']

	parm['warmup_steps'] = parm['train_steps'] //8

	if False:
		parm['weigths_load_path'] = ''	

	#################################### callbacks ##########################################
	cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(parm['checkpoint_path'], verbose=0, save_weights_only=False, save_freq='epoch')
	parm['callbacks'] = [cb_checkpoint, Cb_Test_Plot(parm)]

	#########################################################################################
	fcn_compile_and_fit_custom(parm)

if __name__ == '__main__':
	parm = fcn_init()
	fcn_run(parm)