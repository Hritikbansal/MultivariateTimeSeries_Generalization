import argparse
import torch
import utils
import datetime
import os
import pickle
import random

import numpy as np
from torch.utils import data
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

import sys


def save_predictions(save_folder, used_params, target_,predicted_,i):
	a = target_.cpu()
	#print(a.shape)
	b = predicted_.detach().cpu().numpy()
	#target_ = torch.reshape(a,(a.shape[1]*a.shape[0],a.shape[2])).float()
	#predicted_ = np.reshape(b,(b.shape[1]*b.shape[0],b.shape[2]))
	target_ = torch.reshape(a,(a.shape[0],a.shape[1]*a.shape[2])).float()
	predicted_ = np.reshape(b,(b.shape[0],b.shape[1]*b.shape[2]))
	with open(save_folder+used_params+"target_prediction_"+str(i)+".csv",'ab') as f:
		np.savetxt(f,np.concatenate((target_,predicted_),axis=1),delimiter=',')

def get_indices(control):
	ind = []
	for i in range(control.size(-1)-1):
		if(control[i]!=control[i+1]):
			ind.append(i)
	return ind


def getHitsandMRR(pred_states, next_states):

	topk = [1, 2, 5, 10]
	hits_at = defaultdict(int)
	num_samples = 0
	rr_sum = 0

	pred_state_cat = torch.cat(pred_states, dim=0)
	next_state_cat = torch.cat(next_states, dim=0)

	full_size = pred_state_cat.size(0)

	# Flatten object/feature dimensions
	next_state_flat = next_state_cat.view(full_size, -1)
	pred_state_flat = pred_state_cat.view(full_size, -1)

	dist_matrix = utils.pairwise_distance_matrix(
		next_state_flat, pred_state_flat)
	dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
	dist_matrix_augmented = torch.cat(
		[dist_matrix_diag, dist_matrix], dim=1)

	# Workaround to get a stable sort in numpy.
	dist_np = dist_matrix_augmented.detach().numpy()
	np.savetxt(save_folder+used_params+"dist_np.csv",dist_np,delimiter=',',fmt='%10.3f')#.shape,dist_matrix.shape,dist_matrix_diag.shape,dist_matrix_augmented.shape)
	indices = []
	for row in dist_np:
		keys = (np.arange(len(row)), row)
		indices.append(np.lexsort(keys))
	indices = np.stack(indices, axis=0)
	indices = torch.from_numpy(indices).long()

	labels = torch.zeros(
		indices.size(0), device=indices.device,
		dtype=torch.int64).unsqueeze(-1)

	num_samples += full_size
	print('Size of current topk evaluation batch: {}'.format(
		full_size))

	for k in topk:
		match = indices[:, :k] == labels
		num_matches = match.sum()
		hits_at[k] += num_matches.item()

	match = indices == labels
	_, ranks = match.max(1)

	reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
	rr_sum += reciprocal_ranks.sum()

	return num_samples, topk, hits_at, rr_sum

def per_obj_mse(x1, x2):
	#x1 (bsz, o)
	diff  = x1-x2
	return (diff**2).mean(dim=0).squeeze() #shape (o, )

def getStages_norm(model):

	c = 0
	for name, param in model.named_parameters():
		if"first_stage" in name:
			if "weight" in name:
				c+=(torch.norm(param.view(-1), 1)/param.view(-1).size()[0])
		# elif "second_stage" in name:
		# 	if "weight" in name:
		# 		c+=(torch.norm(param.view(-1), 1)/param.view(-1).size()[0])
	return c

def getTM_norm(model):

	c = 0
	for name, param in model.named_parameters():
		if "transition_model_" in name:
			if "weight" in name:
				c+=(torch.norm(param.view(-1), 1)/param.view(-1).size()[0])
	return c

def applyGroupLassoStages(model, lr, lambda_1, emb_dim):

	softShrink = nn.Softshrink(lr*lambda_1)
	n = emb_dim
	
	with torch.no_grad():
		for name, param in model.named_parameters():
			if "first_stage_0_1" in name or "first_stage_1_0" in name:
				if "weight" in name:
					normTensor = torch.norm(param[:,:n], p=2, keepdim = True)
					param[:,:n] = param[:,:n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
					normTensor = torch.norm(param[:,n:], p=2, keepdim = True)
					param[:,n:] = param[:,n:]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)

			# if "second_stage" in name:
			# 	if "weight" in name:
			# 		normTensor = torch.norm(param[:,:n], p=2, keepdim = True)
			# 		param[:,:n] = param[:,:n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
			# 		normTensor = torch.norm(param[:,n:], p=2, keepdim = True)
			# 		param[:,n:] = param[:,n:]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)

			'''
			if "extract_state_ls" in name:
				if "weight" in name:	
					normTensor = torch.norm(param[:,:n], p=2, keepdim = True)
					param[:,:n] = param[:,:n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
					normTensor = torch.norm(param[:,n:], p=2, keepdim = True)
					param[:,n:] = param[:,n:]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
			'''


def applyGroupLassoTM(model, lr, lambda_1, emb_dim):

	softShrink = nn.Softshrink(lr*lambda_1)
	n = emb_dim

	with torch.no_grad():
		for name, param in model.named_parameters():
			if "transition_model_" in name:
				if "weight" in name:
					normTensor = torch.norm(param[:,:n], p=2, keepdim = True)
					param[:,:n] = param[:,:n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
					normTensor = torch.norm(param[:,n:], p=2, keepdim = True)
					param[:,n:] = param[:,n:]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)


def applyGroupLassoBaseLine(model, lr, lambda_1, emb_dim):

	softShrink = nn.Softshrink(lr*lambda_1)
	n = emb_dim
	size = model.transition_model[0].weight.data.shape
	assert size[0]%n == 0
	assert size[1]%n == 0
	assert model.transition_model[2].weight.data.shape[0] == model.transition_model[2].weight.data.shape[1]

	O, I = size[0]//n, size[1]//n

	with torch.no_grad():
		for name, param in model.named_parameters():
			if "transition_model.0" in name:
				if "weight" in name:
					for i in range(O):
						for j in range(I):
							normTensor = torch.norm(param[i*n:(i+1)*n,j*n:(j+1)*n], p=2, keepdim = True)
							param[i*n:(i+1)*n,j*n:(j+1)*n] = param[i*n:(i+1)*n,j*n:(j+1)*n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)
			if "transition_model.2" in name:
				if "weight" in name:
					for i in range(O):
						for j in range(O):
							normTensor = torch.norm(param[i*n:(i+1)*n,j*n:(j+1)*n], p=2, keepdim = True)
							param[i*n:(i+1)*n,j*n:(j+1)*n] = param[i*n:(i+1)*n,j*n:(j+1)*n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_1*0.1)

def applyGroupLassoDecoder(model, lr, lambda_2, emb_dim, num_objects):

	softShrink = nn.Softshrink(lr*lambda_2)
	n = emb_dim

	size = model.decoder.weight.data.shape
	assert size[0] == num_objects
	assert size[1]%n == 0

	K = size[1]//n

	with torch.no_grad():
		for i in range(K):
			normTensor = torch.norm(model.decoder.weight.data[:,i*n:(i+1)*n], p=2, keepdim=True)
			model.decoder.weight.data[:,i*n:(i+1)*n] = model.decoder.weight.data[:,i*n:(i+1)*n]*softShrink(normTensor)/torch.clamp(normTensor, min=lr*lambda_2*0.1)
