"""Utility functions."""

import os
#import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn
import sys
# import matplotlib.pyplot as plt
import pandas
import random

EPS = 1e-3

np.random.seed(42)
random.seed(42)

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.xavier_uniform_(m.weight)
		nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
	"""Save dictionary containing numpy arrays to h5py file."""

	# Ensure directory exists
	directory = os.path.dirname(fname)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with h5py.File(fname, 'w') as hf:
		for key in array_dict.keys():
			hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
	"""Restore dictionary containing numpy arrays from h5py file."""
	array_dict = dict()
	with h5py.File(fname, 'r') as hf:
		for key in hf.keys():
			array_dict[key] = hf[key][:]
	return array_dict


def save_list_dict_h5py(array_dict, fname):
	"""Save list of dictionaries containing numpy arrays to h5py file."""

	# Ensure directory exists
	directory = os.path.dirname(fname)
	if not os.path.exists(directory):
		os.makedirs(directory)

	with h5py.File(fname, 'w') as hf:
		for i in range(len(array_dict)):
			grp = hf.create_group(str(i))
			for key in array_dict[i].keys():
				grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
	"""Restore list of dictionaries containing numpy arrays from h5py file."""
	array_dict = list()
	with h5py.File(fname, 'r') as hf:
		for i, grp in enumerate(hf.keys()):
			array_dict.append(dict())
			for key in hf[grp].keys():
				array_dict[i][key] = hf[grp][key][:]
	return array_dict


def get_colors(cmap='Set1', num_colors=9):
	"""Get color array from matplotlib colormap."""
	cm = plt.get_cmap(cmap)

	colors = []
	for i in range(num_colors):
		colors.append((cm(1. * i / num_colors)))

	return colors


def pairwise_distance_matrix(x, y):
	num_samples = x.size(0)
	dim = x.size(1)

	x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
	y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

	return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
	if act_fn == 'relu':
		return nn.ReLU()
	elif act_fn == 'leaky_relu':
		return nn.LeakyReLU()
	elif act_fn == 'elu':
		return nn.ELU()
	elif act_fn == 'sigmoid':
		return nn.Sigmoid()
	elif act_fn == 'softplus':
		return nn.Softplus()
	else:
		raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
	"""Get one-hot encoding of index tensors."""
	zeros = torch.zeros(
		indices.size()[0], max_index, dtype=torch.float32,
		device=indices.device)
	return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
	"""Convert numpy array to float32."""
	return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
	"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
	result_shape = (num_segments, tensor.size(1))
	result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
	segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
	result.scatter_add_(0, segment_ids, tensor)
	return result


class StateTransitionsDataset(data.Dataset):
	"""Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

	def __init__(self, hdf5_file):
		"""
		Args:
			hdf5_file (string): Path to the hdf5 file that contains experience
				buffer
		"""
		self.experience_buffer = load_list_dict_h5py(hdf5_file)

		# Build table for conversion between linear idx -> episode/step idx
		self.idx2episode = list()
		step = 0
		for ep in range(len(self.experience_buffer)):
			num_steps = len(self.experience_buffer[ep]['action'])
			idx_tuple = [(ep, idx) for idx in range(num_steps)]
			self.idx2episode.extend(idx_tuple)
			step += num_steps

		self.num_steps = step

	def __len__(self):
		return self.num_steps

	def __getitem__(self, idx):
		ep, step = self.idx2episode[idx]

		obs = to_float(self.experience_buffer[ep]['obs'][step])
		action = self.experience_buffer[ep]['action'][step]
		next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

		return obs, action, next_obs


class PathDataset(data.Dataset):
	"""Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
	"""

	def __init__(self, hdf5_file, path_length=5):
		"""
		Args:
			hdf5_file (string): Path to the hdf5 file that contains experience
				buffer
		"""
		self.experience_buffer = load_list_dict_h5py(hdf5_file)
		self.path_length = path_length

	def __len__(self):
		return len(self.experience_buffer)

	def __getitem__(self, idx):
		observations = []
		actions = []
		for i in range(self.path_length):
			obs = to_float(self.experience_buffer[idx]['obs'][i])
			action = self.experience_buffer[idx]['action'][i]
			observations.append(obs)
			actions.append(action)
		obs = to_float(
			self.experience_buffer[idx]['next_obs'][self.path_length - 1])
		observations.append(obs)
		return observations, actions


def load_steam_dataset(path):

	data = np.loadtxt(path)    
	mean = np.nanmean(data, axis = 0)
	var = np.nanvar(data, axis =0)

	data = (data-mean)/np.sqrt(var)
	data = np.transpose(data)    
	return data

def getStatistics(windows):

	mean = np.nanmean(windows, axis=0)
	var = np.nanvar(windows, axis=0)
	return mean, var


def load_sarcos_dataset(path, window_length, horizon, shift=1, shuffle_data=True, steps = 1):

	df = pandas.read_csv(path,header=None)
	data = np.array(df)

	chunks = [data[6300:6735,:],data[6741:7409,:],data[7415:8084,:],data[8090:8757,:],data[8763:9431,:],
	data[9437:10105,:],data[10111:10780,:],data[10786:11454,:], data[11460:12126,:],data[12132:12802,:],
	data[12808:13475,:],data[13481:14149,:],data[14155:14824,:],data[14830:15364,:]]
	print(len(chunks))

	stateList = []
	contList = []
	fut_contList = []
	fut_stateList = []

	for i in range(len(chunks)):
		state,cont,fut_cont,fut_state=getWindows(np.transpose(chunks[i]), window_length, horizon, shift, steps, shuffle_data=True)
		stateList.append(state)
		contList.append(cont)
		fut_contList.append(fut_cont)
		fut_stateList.append(fut_state)
	state = np.concatenate(stateList,axis=0)
	cont = np.concatenate(contList,axis=0)
	fut_cont = np.concatenate(fut_contList,axis=0)
	fut_state = np.concatenate(fut_stateList,axis=0)

	if shuffle_data:
		perm=np.random.permutation(len(state))
		state, cont, fut_cont, fut_state = state[perm], cont[perm], fut_cont[perm], fut_state[perm]
	print(state.shape)
	num_windows = state.shape[0]
	print(num_windows)
	train_state, train_control, train_futcontrol, train_futstate = state[:int(num_windows*0.6),:,:], cont[:int(num_windows*0.6),:,:], fut_cont[:int(num_windows*0.6),:,:], fut_state[:int(num_windows*0.6),:,:]
	valid_state, valid_control, valid_futcontrol, valid_futstate = state[int(0.6*num_windows):int(0.8*num_windows),:,:],cont[int(0.6*num_windows):int(0.8*num_windows),:,:],fut_cont[int(0.6*num_windows):int(0.8*num_windows),:,:],fut_state[int(0.6*num_windows):int(0.8*num_windows),:,:]
	test_state, test_control, test_futcontrol, test_futstate = state[int(num_windows*0.8):,:,:], cont[int(num_windows*0.8):,:,:], fut_cont[int(num_windows*0.8):,:,:], fut_state[int(num_windows*0.8):,:,:]

	train_Stats_state, train_Stats_control = np.concatenate([train_state,train_futstate],axis=2),np.concatenate([train_control,train_futcontrol],axis=2)
	valid_Stats_state, valid_Stats_control = np.concatenate([valid_state,valid_futstate],axis=2),np.concatenate([valid_control,valid_futcontrol],axis=2)
	test_Stats_state, test_Stats_control = np.concatenate([test_state,test_futstate],axis=2),np.concatenate([test_control,test_futcontrol],axis=2)

	mean, var = getStatistics(train_Stats_state)
	train_Stats_state = (train_Stats_state-mean)/np.sqrt(var)
	valid_Stats_state = (valid_Stats_state-mean)/np.sqrt(var)
	test_Stats_state = (test_Stats_state-mean)/np.sqrt(var)
	train_state, train_futstate = train_Stats_state[:,:,:window_length],train_Stats_state[:,:,window_length:]
	valid_state, valid_futstate = valid_Stats_state[:,:,:window_length],valid_Stats_state[:,:,window_length:]
	test_state, test_futstate = test_Stats_state[:,:,:window_length],test_Stats_state[:,:,window_length:]

	mean, var = getStatistics(train_Stats_control)
	train_Stats_control = (train_Stats_control-mean)/np.sqrt(var)
	valid_Stats_control = (valid_Stats_control-mean)/np.sqrt(var)
	test_Stats_control = (test_Stats_control-mean)/np.sqrt(var)
	train_control, train_futcontrol = train_Stats_control[:,:,:window_length],train_Stats_control[:,:,window_length:]
	valid_control, valid_futcontrol = valid_Stats_control[:,:,:window_length],valid_Stats_control[:,:,window_length:]
	test_control, test_futcontrol = test_Stats_control[:,:,:window_length],test_Stats_control[:,:,window_length:]

	return train_state, train_control, train_futcontrol, train_futstate, valid_state, valid_control, valid_futcontrol, valid_futstate, test_state, test_control, test_futcontrol, test_futstate

def getWindows(data, window_length, horizon, shift, steps, shuffle_data):

	length = data.shape[1]
	num_variables = data.shape[0]
	
	if(steps > 1):
		horizon = steps*horizon
	
	if(shift==0):
		print('shift cannot be 0.')
		sys.exit(0)

	total = window_length + horizon
	num_windows = ((length-total)//shift) + 1

	control = data[21:]
	state = data[0:7]

	state_time_series = []
	control_time_series = []
	future_control_time_series = []
	future_state_time_series = []

	for i in range(num_windows):
		state_time_series.append(state[:,i*shift:i*shift+window_length])
		control_time_series.append(control[:,i*shift:i*shift+window_length])
		future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
		future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])

	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)
	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	return state_time_series, control_time_series, future_control_time_series, future_state_time_series

def createSarcosDataset(path, window_length, horizon, shift=1, shuffle_data=True, steps = 1):
	
	return load_sarcos_dataset(path, window_length, horizon, shift=1, shuffle_data=True, steps=steps)   


def createSteamDataset(path, window_length, horizon):

	data = load_steam_dataset(path)
	
	length = data.shape[1]
	num_variables = data.shape[0]

	total = window_length + horizon
	num_windows = length//total

	time = np.expand_dims(data[0], axis=0)
	control = data[1:5]
	state = data[5:]

	time_series = []
	state_time_series = []
	control_time_series = []
	future_time_series = []
	future_control_time_series = []
	future_state_time_series = []

	for i in range(num_windows):
		time_series.append(time[:,i*total:i*total+window_length])
		state_time_series.append(state[:,i*total:i*total+window_length])
		control_time_series.append(control[:,i*total:i*total+window_length])

		future_time_series.append(time[:,i*total+window_length:(i+1)*total])
		future_state_time_series.append(state[:,i*total+window_length:(i+1)*total])
		future_control_time_series.append(control[:,i*total+window_length:(i+1)*total])

	time_series = np.array(time_series)
	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)

	future_time_series = np.array(future_time_series)
	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	return time_series, state_time_series, control_time_series, future_time_series, future_state_time_series, future_control_time_series


class SteamDataset(data.Dataset):

	def __init__(self, time_series, state_time_series, control_time_series, future_time_series, future_state_time_series, future_control_time_series):
		super(SteamDataset, self).__init__()
		self.time_series = time_series
		self.state_time_series = state_time_series
		self.control_time_series = control_time_series
		self.future_time_series = future_time_series
		self.future_state_time_series = future_state_time_series
		self.future_control_time_series = future_control_time_series

	def __getitem__(self, index):
		return self.time_series[index], self.state_time_series[index], self.control_time_series[index], self.future_time_series[index], 
		self.future_state_time_series[index], self.future_control_time_series[index]


class SarcosDataset(data.Dataset):

	def __init__(self, state_past, cont_past, action, next_obs):
		super(SarcosDataset, self).__init__()
		self.state_past = state_past
		self.cont_past = cont_past
		self.action = action
		self.next_obs = next_obs
		self.length = self.state_past.shape[0]

	def __getitem__(self, index):
		return self.state_past[index], self.cont_past[index], self.action[index], self.next_obs[index]

	def __len__(self):
		return self.length

def hasChanged(control):

	num_variables = control.shape[0]
	flag = False
	for i in range(num_variables):
		if flag:
			break
		if(np.unique(control[i]).size>1):
			flag=True
	return flag

def load_swatFull_dataset(path, mean=None, var=None):
	
	df = pandas.read_csv(path)
	c = 0
	s = 0
	data1 = np.array([])
	data2 = np.array([])
	for col in df.columns:
		#Note: P301 takes value 2 only 11 times in entire data after 82767.
		if(col=='P101' or col == 'MV101' or col=='MV201'or col=='P203'
			or col=='P205'or col=='MV301'or col=='MV302'or col=='MV303'
			or col=='MV304' or col=='P302' or col=='P602'):
			if(c==0):
				data1 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32), axis=1) 
				c+=1
			else:
				data1 = np.concatenate((data1,np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32),axis=1)),axis=1)
		elif(col=='FIT101' or col=='LIT101' or col=='FIT201' or col=='DPIT301'
			or col == 'FIT301' or col=='LIT301' or col=='LIT401' or col=='FIT601'):
			if(s==0):
				data2 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32), axis=1) 
				s+=1
			else:
				data2 = np.concatenate((data2,np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32),axis=1)),axis=1) 

	print(data1.shape)
	print(data2.shape)

	train_data1 = data1[0:int(data1.shape[0]*0.8)]
	test_data1 = data1[int(data1.shape[0]*0.8):]

	train_data2 = data2[0:int(data2.shape[0]*0.8)]
	test_data2 = data2[int(data2.shape[0]*0.8):]

	if(mean==None):
		min_ = np.nanmin(train_data1, axis=0)
		max_ = np.nanmax(train_data1, axis=0)
		train_data1 = (train_data1-min_)/(max_ - min_)
		test_data1 = (test_data1-min_)/(max_ - min_)

		'''mean = np.nanmean(data2, axis=0)
		var = np.nanvar(data2, axis=0)
		data2 = (data2-mean)/(var-mean)'''
		min_ = np.nanmin(train_data2, axis=0)
		max_ = np.nanmax(train_data2, axis=0)
		train_data2 = (train_data2-min_)/(max_ - min_)
		test_data2 = (test_data2-min_)/(max_ - min_)
		

	train_data = np.concatenate((train_data1, train_data2), axis=1)
	test_data = np.concatenate((test_data1, test_data2), axis=1)

	train_data = np.transpose(train_data)
	test_data = np.transpose(test_data)

	#return data, mean, var
	return train_data, test_data#min_, max_

def createSwatFullDataset(path, window_length, horizon, shift=1, shuffle_data=True, train=True, steps=1):

	train_data, test_data = load_swatFull_dataset(path)
	data = train_data if train else test_data
	# np.set_printoptions(suppress=True)
	# print("mean",np.round(mean,5))
	# print("stdev",np.round(np.sqrt(var),5))
	# print("max",np.round(np.max(data,axis=1),5),len(np.max(data,axis=1)))
	# print("min",np.round(np.min(data,axis=1),5))
	#sys.exit()
	length = data.shape[1]
	num_variables = data.shape[0]

	if(not train and steps > 1):
		horizon = steps*horizon 

	if(shift==0):
		print('shift cannot be 0.')
		sys.exit(0)
	'''
	control = data[:12]
	state = data[12:]
	'''
	control = data[:11]
	state = data[11:]

	total = window_length + horizon
	num_windows = ((length-total)//shift) + 1

	state_time_series = []
	control_time_series = []
	future_control_time_series = []
	future_state_time_series = []
	c = 0
	for i in range(num_windows):
		#if(hasChanged(control[:,i*shift:i*shift+window_length]) and hasChanged(control[:,i*shift+window_length:i*shift+total])):
		#if(hasChanged(control[:,i*shift+window_length-5:i*shift+window_length])):
		# if(steps==1):
		#     if(hasChanged(control[:,i*shift+window_length:i*shift+window_length+10])):
		#         c+=1
		#         state_time_series.append(state[:,i*shift:i*shift+window_length])
		#         control_time_series.append(control[:,i*shift:i*shift+window_length])
		#         future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
		#         future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])
		# else:
		flag = True
		for k in range(steps):
			flag = flag and hasChanged(control[:,i*shift+k*(horizon//steps)+window_length:i*shift+k*(horizon//steps)+window_length+10])
		if flag:
			c+=1
			state_time_series.append(state[:,i*shift:i*shift+window_length])
			control_time_series.append(control[:,i*shift:i*shift+window_length])
			future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
			future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])

	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)

	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	if shuffle_data:
		perm=np.random.permutation(len(state_time_series))
		return state_time_series[perm], control_time_series[perm], future_control_time_series[perm], future_state_time_series[perm]
	
	return state_time_series, control_time_series, future_control_time_series, future_state_time_series


class SwatFullDataset(data.Dataset):

	def __init__(self, state_past, cont_past, action, next_obs):
		super(SwatFullDataset, self).__init__()
		self.state_past = state_past
		self.cont_past = cont_past
		self.action = action
		self.next_obs = next_obs
		self.length = self.state_past.shape[0]

	def __getitem__(self, index):
		return self.state_past[index], self.cont_past[index], self.action[index], self.next_obs[index]

	def __len__(self):
		return self.length


def load_swat_dataset(path, mean=None, var=None, setup=1):
	
	df = pandas.read_csv(path)
	c = 0
	s = 0
	data1 = np.array([])
	data2 = np.array([])
	if(setup==1):
		for col in df.columns:
			if(col=='P101' or col == 'MV101'):
				if(c==0):
					data1 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32), axis=1) 
					c+=1
				else:
					data1 = np.concatenate((data1,np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32),axis=1)),axis=1)
			elif(col=='FIT101' or col=='LIT101'):
				if(s==0):
					data2 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32), axis=1) 
					s+=1
				else:
					data2 = np.concatenate((data2,np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32),axis=1)),axis=1) 
	if(setup==2):
		for col in df.columns:
			if(col=='P101' or col == 'MV101' or col=='P602'):
				if(c==0):
					data1 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32), axis=1) 
					c+=1
				else:
					data1 = np.concatenate((data1,np.expand_dims(np.array(df[col][82767:336000],dtype=np.int32),axis=1)),axis=1)
			elif(col=='FIT101' or col=='LIT101' or col=='FIT601'):
				if(s==0):
					data2 = np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32), axis=1) 
					s+=1
				else:
					data2 = np.concatenate((data2,np.expand_dims(np.array(df[col][82767:336000],dtype=np.float32),axis=1)),axis=1) 

	print(data1.shape)
	print(data2.shape)

	train_data1 = data1[0:int(data1.shape[0]*0.8)]
	test_data1 = data1[int(data1.shape[0]*0.8):]

	train_data2 = data2[0:int(data2.shape[0]*0.8)]
	test_data2 = data2[int(data2.shape[0]*0.8):]

	if(mean==None):
		min_ = np.nanmin(train_data1, axis=0)
		max_ = np.nanmax(train_data1, axis=0)
		train_data1 = (train_data1-min_)/(max_ - min_)
		test_data1 = (test_data1-min_)/(max_ - min_)

		min_ = np.nanmin(train_data2, axis=0)
		max_ = np.nanmax(train_data2, axis=0)
		train_data2 = (train_data2-min_)/(max_ - min_)
		test_data2 = (test_data2-min_)/(max_ - min_)
		

	train_data = np.concatenate((train_data1, train_data2), axis=1)
	test_data = np.concatenate((test_data1, test_data2), axis=1)
	train_data = np.transpose(train_data)
	test_data = np.transpose(test_data)

	return train_data, test_data

def createSwatDataset(path, window_length, horizon, shift=1, shuffle_data=True, train=True, steps=1,setup=1):

	print(path)#train_data, test_data = load_swatNew_dataset(path) if setup==3 else load_swat_dataset(path=path,setup=setup)
	train_data, test_data = load_swatNew_dataset(path) if setup==3 else load_swat_dataset(path=path,setup=setup)
	data = train_data if train else test_data
	length = data.shape[1]
	num_variables = data.shape[0]

	if(not train and steps > 1):
		horizon = steps*horizon 

	if(shift==0):
		print('shift cannot be 0.')
		sys.exit(0)

	if(setup==2):
		control = data[:3]
		state = data[3:]
	else:
		control = data[:2]
		state = data[2:]

	total = window_length + horizon
	num_windows = ((length-total)//shift) + 1

	state_time_series = []
	control_time_series = []
	future_control_time_series = []
	future_state_time_series = []
	c = 0

	for i in range(num_windows):
		state_time_series.append(state[:,i*shift:i*shift+window_length])
		control_time_series.append(control[:,i*shift:i*shift+window_length])
		future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
		future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])


	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)

	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	if shuffle_data:
		perm=np.random.permutation(len(state_time_series))
		return state_time_series[perm], control_time_series[perm], future_control_time_series[perm], future_state_time_series[perm]
	
	return state_time_series, control_time_series, future_control_time_series, future_state_time_series


class SwatDataset(data.Dataset):

	def __init__(self, state_past, cont_past, action, next_obs):
		super(SwatDataset, self).__init__()
		self.state_past = state_past
		self.cont_past = cont_past
		self.action = action
		self.next_obs = next_obs
		self.length = self.state_past.shape[0]

	def __getitem__(self, index):
		return self.state_past[index], self.cont_past[index], self.action[index], self.next_obs[index]

	def __len__(self):
		return self.length

def addNoise(signal, noise_ratio, pert=None):
	# signal is supposed to be of form (M, L), where is M is bsz and L is total length
	#object list is the list of objects on which noise will be injected. 
	if noise_ratio==0:
		return signal 

	M = signal.size(0)
	num_obj = signal.size(1)
	L = signal.size(2)
	# here we will have a nice mask (M, num_obj, L) where only are 1 where to inject noise 

	# pow_sig  = (signal**2).mean(dim=-1).squeeze()
	stdev =  noise_ratio #torch.sqrt(noise_ratio*pow_sig)
	noise=[]
	for i in range(M):
		if pert is None :
			noise.append(torch.stack([torch.empty(L).normal_(mean=0, std=stdev) for j in range(num_obj)], dim=0))
		else:
			noise.append(torch.stack([torch.empty(L).normal_(mean=0, std=stdev) if j==pert else torch.zeros(L) for j in range(num_obj)], dim=0))

	noise = torch.stack(noise, dim=0)
	noise = noise.type(signal.dtype).to(signal.device)
	return (noise + signal)

def addnoise_ratio(dataloader, noise_ratio, pert=None):
	# dataset would be of dataset class
	val_loader = []
	# tot = 0
	for batch_idx, data_batch in enumerate(dataloader):
		# To add noise in the dependent variables, uncomment below
		#data_batch_perturbed = [addNoise(data_batch[0], noise_ratio, pert)] + data_batch[1:]  # perturbing the state readings obtained so far!!
		# To add noise in the control variables, uncomment below
		data_batch_perturbed = data_batch[:2]+[addNoise(data_batch[2], noise_ratio, pert)] + [data_batch[3]]  
		val_loader.append(data_batch_perturbed)
		# tot+=data_batch[0].size(0)

	return val_loader #, tot


def load_swatNew_dataset(path , mean=None, var=None):
	df = pandas.read_csv(path)
	c = 0
	s = 0
	data1 = np.array([])
	data2 = np.array([])
	for col in df.columns:
		if(col=='P602' or col == 'MV101'):
			if(c==0):
				data1 = np.expand_dims(np.array(df[col],dtype=np.int32), axis=1)
				c+=1
			else:
				data1 = np.concatenate((data1,np.expand_dims(np.array(df[col],dtype=np.int32),axis=1)),axis=1)
		elif(col=='FIT101' or col=='FIT601'):
			if(s==0):
				data2 = np.expand_dims(np.array(df[col],dtype=np.float32), axis=1)
				s+=1
			else:
				data2 = np.concatenate((data2,np.expand_dims(np.array(df[col],dtype=np.float32),axis=1)),axis=1)
	print(data1.shape)
	print(data2.shape)
	train_data1 = data1[0:int(data1.shape[0]*0.8)]
	test_data1 = data1[int(data1.shape[0]*0.8):]
	train_data2 = data2[0:int(data2.shape[0]*0.8)]
	test_data2 = data2[int(data2.shape[0]*0.8):]
	if(mean==None):
		min_ = np.nanmin(train_data1, axis=0)
		max_ = np.nanmax(train_data1, axis=0)
		train_data1 = (train_data1-min_)/(max_ - min_)
		test_data1 = (test_data1-min_)/(max_ - min_)
		min_ = np.nanmin(train_data2, axis=0)
		max_ = np.nanmax(train_data2, axis=0)
		train_data2 = (train_data2-min_)/(max_ - min_)
		test_data2 = (test_data2-min_)/(max_ - min_)
	train_data = np.concatenate((train_data1, train_data2), axis=1)
	test_data = np.concatenate((test_data1, test_data2), axis=1)
	train_data = np.transpose(train_data)
	test_data = np.transpose(test_data)
	return train_data, test_data



def load_NARMA_dataset(path):

	df = pandas.read_csv(path)
	c = 0
	s = 0
	data1 = np.array([])
	data2 = np.array([])
	for col in df.columns:
		if(col=='CV1' or col=='CV2'):
			if(s==0):
				data2 = np.expand_dims(np.array(df[col][1:],dtype=np.float32), axis=1) 
				s+=1
			else:
				data2 = np.concatenate((data2,np.expand_dims(np.array(df[col][1:],dtype=np.float32),axis=1)),axis=1) 

		elif(col=='DV1' or col=='DV2'):
			if(c==0):
				data1 = np.expand_dims(np.array(df[col][1:],dtype=np.float32), axis=1) 
				c+=1
			else:
				data1 = np.concatenate((data1,np.expand_dims(np.array(df[col][1:],dtype=np.float32),axis=1)),axis=1)


	print(data1.shape)
	print(data2.shape)

	train_data1 = data1[0:int(data1.shape[0]*0.8)]
	test_data1 = data1[int(data1.shape[0]*0.8):]

	train_data2 = data2[0:int(data2.shape[0]*0.8)]
	test_data2 = data2[int(data2.shape[0]*0.8):]

	'''
	min_ = np.nanmin(train_data1, axis=0)
	max_ = np.nanmax(train_data1, axis=0)
	train_data1 = (train_data1-min_)/(max_ - min_)
	test_data1 = (test_data1-min_)/(max_ - min_)

	min_ = np.nanmin(train_data2, axis=0)
	max_ = np.nanmax(train_data2, axis=0)
	train_data2 = (train_data2-min_)/(max_ - min_)
	test_data2 = (test_data2-min_)/(max_ - min_)
	'''

	train_data = np.concatenate((train_data1, train_data2), axis=1)
	test_data = np.concatenate((test_data1, test_data2), axis=1)
	train_data = np.transpose(train_data)
	test_data = np.transpose(test_data)

	return train_data, test_data

def IsOODNarma( control ):

	assert(control.shape[0] == 2)

	flag = True

	for j in range(control.shape[1]):
		if (not ((control[1][j] > 0.4*control[0][j] + 0.6) or (control[1][j] < 0.4*control[0][j]))):
			flag = False
			break

	return flag

def createNarmaDataset(path, window_length, horizon, num_cont, num_objects, shift=1, shuffle_data=True, train=True, steps=1, setup = 0): #

	print(path)#train_data, test_data = load_swatNew_dataset(path) if setup==3 else load_swat_dataset(path=path,setup=setup)
	train_data, test_data = load_NARMA_dataset(path=path )
	data = train_data if train else test_data
	length = data.shape[1]
	num_variables = data.shape[0]

	if(not train and steps > 1):
		horizon = steps*horizon 

	if(shift==0):
		print('shift cannot be 0.')
		sys.exit(0)

	control = data[num_objects:num_objects+num_cont] # assuming that CVs are always at the end
	state = data[:num_objects] # assuming DVs are always in the beginning

	total = window_length + horizon
	num_windows = ((length-total)//shift) + 1

	state_time_series = []
	control_time_series = []
	future_control_time_series = []
	future_state_time_series = []
	c = 0

	for i in range(num_windows):
		if(not setup):
			state_time_series.append(state[:,i*shift:i*shift+window_length])
			control_time_series.append(control[:,i*shift:i*shift+window_length])
			future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
			future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])
		
		else:
			if(IsOODNarma(control[:, i*shift+window_length:i*shift+total])):
				state_time_series.append(state[:,i*shift:i*shift+window_length])
				control_time_series.append(control[:,i*shift:i*shift+window_length])
				future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
				future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])        


	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)

	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	if shuffle_data:
		perm=np.random.permutation(len(state_time_series))
		return state_time_series[perm], control_time_series[perm], future_control_time_series[perm], future_state_time_series[perm]
	
	return state_time_series, control_time_series, future_control_time_series, future_state_time_series

class NarmaDataset(data.Dataset):

	def __init__(self, state_past, cont_past, action, next_obs):
		super(NarmaDataset, self).__init__()
		self.state_past = state_past
		self.cont_past = cont_past
		self.action = action
		self.next_obs = next_obs
		self.length = self.state_past.shape[0]

	def __getitem__(self, index):
		return self.state_past[index], self.cont_past[index], self.action[index], self.next_obs[index]

	def __len__(self):
		return self.length



class PMSMDataset(data.Dataset):

	def __init__(self, state_past, cont_past, action, next_obs):
		super(PMSMDataset, self).__init__()
		self.state_past = state_past
		self.cont_past = cont_past
		self.action = action
		self.next_obs = next_obs
		self.length = self.state_past.shape[0]

	def __getitem__(self, index):
		return self.state_past[index], self.cont_past[index], self.action[index], self.next_obs[index]

	def __len__(self):
		return self.length



def load_PMSM_dataset(path):

	df = pandas.read_csv(path)
	c = 0
	s = 0
	data1 = np.array([])
	data2 = np.array([])
	for col in df.columns:
		if(col=='u_d' or col=='u_q'):
			if(s==0):
				data2 = np.expand_dims(np.array(df[col][1:],dtype=np.float32), axis=1) 
				s+=1
			else:
				data2 = np.concatenate((data2,np.expand_dims(np.array(df[col][1:],dtype=np.float32),axis=1)),axis=1) 

		elif(col=='i_d' or col=='i_q'):
			if(c==0):
				data1 = np.expand_dims(np.array(df[col][1:],dtype=np.float32), axis=1) 
				c+=1
			else:
				data1 = np.concatenate((data1,np.expand_dims(np.array(df[col][1:],dtype=np.float32),axis=1)),axis=1)


	print(data1.shape)
	print(data2.shape)

	train_data1 = data1[0:int(data1.shape[0]*0.8)]
	test_data1 = data1[int(data1.shape[0]*0.8):]

	train_data2 = data2[0:int(data2.shape[0]*0.8)]
	test_data2 = data2[int(data2.shape[0]*0.8):]

	train_data = np.concatenate((train_data1, train_data2), axis=1)
	test_data = np.concatenate((test_data1, test_data2), axis=1)
	train_data = np.transpose(train_data)
	test_data = np.transpose(test_data)

	return train_data, test_data

def createPMSMDataset(path, window_length, horizon, num_cont, num_objects, shift=1, shuffle_data=True, train=True, steps=1): #

	print(path)
	train_data, test_data = load_PMSM_dataset(path=path)
	data = train_data if train else test_data
	length = data.shape[1]
	num_variables = data.shape[0]

	if(not train and steps > 1):
		horizon = steps*horizon 

	if(shift==0):
		print('shift cannot be 0.')
		sys.exit(0)

	control = data[:num_cont] 
	state = data[num_cont:num_cont+num_objects] 

	total = window_length + horizon
	num_windows = ((length-total)//shift) + 1

	state_time_series = []
	control_time_series = []
	future_control_time_series = []
	future_state_time_series = []
	c = 0

	for i in range(num_windows):
		state_time_series.append(state[:,i*shift:i*shift+window_length])
		control_time_series.append(control[:,i*shift:i*shift+window_length])
		future_state_time_series.append(state[:,i*shift+window_length:i*shift+total])
		future_control_time_series.append(control[:,i*shift+window_length:i*shift+total])


	state_time_series = np.array(state_time_series)
	control_time_series = np.array(control_time_series)

	future_state_time_series = np.array(future_state_time_series)
	future_control_time_series = np.array(future_control_time_series)

	if shuffle_data:
		perm=np.random.permutation(len(state_time_series))
		return state_time_series[perm], control_time_series[perm], future_control_time_series[perm], future_state_time_series[perm]
	
	return state_time_series, control_time_series, future_control_time_series, future_state_time_series