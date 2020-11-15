import argparse
import torch
import utils
import datetime
import os
import pickle
import random

import numpy as np
import logging

from torch.utils import data
from torch import nn

import mts_model
import sys
import helper

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8,
					help='Batch size.')
parser.add_argument('--epochs', type=int, default=100,
					help='Number of training epochs.')
parser.add_argument('--dec_epochs', type=int, default=100,
					help='Number of training epochs of decoder in CL+Dec case.')
parser.add_argument('--learning_rate', type=float, default=5e-4,
					help='Learning rate.')

parser.add_argument('--decoder_l1', type=float, default=1.0,
					help='Learning rate of the decoder.')
parser.add_argument('--decoder_gl', type=float, default=1.0,
					help='Learning rate of the decoder.')
parser.add_argument('--soft_decoder_l1', action='store_true', default=False,
					help='soft decoder with l1')
parser.add_argument('--soft_decoder_gl', action='store_true', default=False,
					help='soft decoder with group lasso')
parser.add_argument('--hard_decoder', action='store_true', default=False,
					help='hard decoder')

parser.add_argument('--sigma', type=float, default=0.5,
					help='Energy scale.')
parser.add_argument('--per_node_MLP', action='store_true', default=False,
					help='different MLP for every node')

parser.add_argument('--layer_l1', action='store_true', default=False,
					help='L1 within hidden layers')
parser.add_argument('--layer_gl', action='store_true', default=False,
					help='group lasso within hidden layers')

parser.add_argument('--l1', type=float, default=1.,
					help='L1 regularization hyperparameter within the hidden layers.')
parser.add_argument('--message_l1', type=float, default=1.,
					help='L1 regularization hyperparameter for the messages.')
parser.add_argument('--gl', type=float, default=1.,
					help='group lasso hyperparameter.')

parser.add_argument('--ood', action='store_true', default=False,
					help='whether testing on OOD data')

parser.add_argument('--hidden-dim', type=int, default=512,
					help='Number of hidden units in transition MLP.')
parser.add_argument('--ignore-action', action='store_true', default=False,
					help='Ignore action in GNN transition model.')
parser.add_argument('--copy-action', action='store_true', default=False,
					help='Apply same action to all object slots.')
parser.add_argument('--normalize', action='store_true', default=False,
					help='Normalize the embeddings for contrastive loss.')

parser.add_argument('--shuffle', action='store_true', default=False,
					help='Shuffle data')
parser.add_argument('--recurrent', action='store_true', default=False,
					help='recurrent transition model')

parser.add_argument('--message_pass', action='store_true', default=False,
					help='allow nodes to pass messages, i.e. use GNN instead of MLP as transition model')

parser.add_argument('--hierarchical_ls', action='store_true', default=False,
					help='hierarchical latent structure')

parser.add_argument('--sepCTRL', action='store_true', default=False,
					help='separate control encoder')

parser.add_argument('--onlyReLU', action='store_true', default=False,
					help='have ReLUs instead of tanh and bypass LSTMs')
					
parser.add_argument('--save_predictions', action='store_true', default=False,
					help='Dump prediction and target csv files')
parser.add_argument('--save_embeddings', action='store_true', default=False,
					help='Dump embeddings')

parser.add_argument('--path', type=str, default='none',required=True,
					help='Path to dataset')
parser.add_argument('--num_objects', type=int, default=21,
					help='Number of object slots in model.')
parser.add_argument('--num_cont', type=int, default=7,
					help='Number of object slots in model.')
parser.add_argument('--stride', type=int, default=1,
					help='CNN strides.')
parser.add_argument('--length', type=int, default=500,
					help='input dim.')
parser.add_argument('--shift', type=int, default=1,
					help='shift length')
parser.add_argument('--window_size', type=int, default=100,
					help='chunk length.')
parser.add_argument('--embedding_dim', type=int, default=50,
					help='embedding dim.')
parser.add_argument('--action_dim', type=int, default=15,
					help='action dim.')
parser.add_argument('--nodes', type=int, default=None,
					help='Number of nodes.')
parser.add_argument('--horizon', type=int, default=100,
					help='Horizon.')
parser.add_argument('--steps', type=int, default=1,
					help='Number of steps.')

parser.add_argument('--setup', type=int, default=1,
					help='swat 1/2.')

parser.add_argument('--use_condenser', type=bool, default=False,
					help='using condensery.')
parser.add_argument('--split', type=float, default=0.75,
					help='Train-test.')
parser.add_argument('--decoder', action='store_true', default=False,
					help='Train model using decoder and mse loss.')
parser.add_argument('--full', action='store_true', default=False,
					help='Have direct transition instead of delta transition')
parser.add_argument('--isControl', action='store_true', default=False,
					help='GNN should have control or not')
parser.add_argument('--dropout', type=float, default=0.0,
					help='Train-test.')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
					help='Random seed (default: 42).')
parser.add_argument('--log_interval', type=int, default=20,
					help='How many batches to wait before logging'
						 'training status.')
parser.add_argument('--save_interval', type=int, default=20,
					help='How many batches to wait before logging'
						 'training status.')
parser.add_argument('--dataset', type=str,
					default='swat', required=True,
					help='Name of the dataset')
parser.add_argument('--name', type=str, default='MVTS',
					help='Experiment name.')
parser.add_argument('--save-folder', type=str,
					default='checkpoints',
					help='Path to checkpoints.')

parser.add_argument('--baseline', action='store_true', default=False,
					help='training for forecasting without CL')
parser.add_argument('--forecasting_cl', action='store_true', default=False,
					help='training for forecasting with CL')
parser.add_argument('--forecasting_M5', action='store_true', default=False,
					help='training for forecasting with CL in M5')
parser.add_argument('--pastStateOnly', action='store_true', default=False,
					help='training encoder for CL followed by decoder')
parser.add_argument('--pastControlOnly', action='store_true', default=False,
					help='training encoder for CL followed by decoder')
parser.add_argument('--futureControlOnly', action='store_true', default=False,
					help='training encoder for CL followed by decoder')
parser.add_argument('--pastinfo', action='store_true', default=False,
					help='training encoder for CL followed by decoder')

parser.add_argument('--noise', action='store_true', default=False,
					help='todo noise')
parser.add_argument('--pert', type=int, default=None,
					help='Which obj to pert')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

now = datetime.datetime.now()
timestamp = now.isoformat()
train_split = args.split

if args.name == 'none':
	exp_name = timestamp
else:
	exp_name = args.name

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


exp_counter = 0
#save_folder = '{}\\{}\\'.format(args.save_folder, exp_name)
save_folder = '{}/{}/'.format(args.save_folder, exp_name)

if not os.path.exists(save_folder):
	os.makedirs(save_folder)
	print('created save folder')
	print(save_folder)


def _init_fn():
	np.random.seed(args.seed)


'''used_params='-'.join([args.dataset, str(args.l1), str(args.message_l1), str(args.length),str(args.window_size),str(args.embedding_dim),str(args.nodes),str(args.decoder),str(args.shift),
				str(5),str(args.dropout),str(args.message_pass),str(args.isControl),str(args.sepCTRL),
				str(args.stride),str(args.full),str(args.baseline), str(args.forecasting_cl), str(args.forecasting_M5)])'''

used_params='-'.join([args.dataset, str(args.batch_size), str(args.embedding_dim),str(args.full),str(args.learning_rate),str(args.baseline), str(args.sepCTRL),str(args.layer_gl), str(args.gl),str(args.hierarchical_ls),str(args.soft_decoder_gl),str(args.decoder_gl),str(args.hard_decoder)]) #,str(args.onlyReLU)])


meta_file = os.path.join(save_folder, used_params+',metadata.pkl')
model_file = os.path.join(save_folder, used_params+',model.pt')
decoder_file = os.path.join(save_folder, used_params+',decoder.pt')
log_file = os.path.join(save_folder, used_params+',log.txt')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(log_file, 'a'))
print = logger.info

pickle.dump({'args': args}, open(meta_file, "wb"))

device = torch.device('cuda' if args.cuda else 'cpu')

len_change_5=0
len_change_10=0
len_change_20=0
len_remaining_5=0
len_remaining_10=0
len_remaining_20=0

print(device)
print(args)
MSEloss = nn.MSELoss()

if(args.dataset=='sarcos'):
	data_path=args.path+'sarcos_inv.csv'
	train_state, train_control, train_fcontrol, train_fstate, valid_state, valid_control, valid_fcontrol, valid_fstate, test_state, test_control, test_fcontrol, test_fstate = utils.createSarcosDataset(path=data_path, window_length=args.length, horizon=args.horizon, shift=args.shift,shuffle_data=args.shuffle, steps = args.steps)
	print(train_state.shape)
	print(train_control.shape)
	print(train_fstate.shape)
	print(train_fcontrol.shape)
elif(args.dataset=='swat'):
	data_path=args.path +'Swat3CV3DV_new.csv' if args.setup==3 else args.path+'SWaT_Dataset_Normal_v01.csv'
	state_past, cont_past, action, next_obs = utils.createSwatDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, shift=args.shift, 
		shuffle_data=args.shuffle, train=True, setup=args.setup)
	test_state_past, test_cont_past, test_action, test_next_obs = utils.createSwatDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, shift=args.shift, 
		shuffle_data=args.shuffle,train=False,steps=args.steps,setup=args.setup)
elif('narma' in args.dataset):
	data_path=args.path + args.dataset
	state_past, cont_past, action, next_obs = utils.createNarmaDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, num_cont=args.num_cont, num_objects=args.num_objects, shift=args.shift, 
		shuffle_data=args.shuffle, train=True)
	test_state_past, test_cont_past, test_action, test_next_obs = utils.createNarmaDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, num_cont=args.num_cont, num_objects=args.num_objects, shift=args.shift, 
		shuffle_data=args.shuffle,train=False)
	print(state_past.shape)
	print(test_state_past.shape)

elif('pmsm' in args.dataset):
	data_path = args.path 
	state_past, cont_past, action, next_obs = utils.createPMSMDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, num_cont=args.num_cont, num_objects=args.num_objects, shift=args.shift, 
		shuffle_data=args.shuffle, train=True)
	test_state_past, test_cont_past, test_action, test_next_obs = utils.createPMSMDataset(path=data_path, window_length=args.length, 
		horizon=args.horizon, num_cont=args.num_cont, num_objects=args.num_objects, shift=args.shift, 
		shuffle_data=args.shuffle,train=False)
	print(state_past.shape)
	print(cont_past.shape)
	print(test_state_past.shape)
	print(test_cont_past.shape)
	
list_shift = [i for i in range(6,17,1)]
swatMasterDict={}
if(args.dataset == 'swat_full'):
	
	for shift in list_shift:
		data_path=args.path+'SWaT_Dataset_Normal_v01.csv'
		
		state_past, cont_past, action, next_obs = utils.createSwatFullDataset(path=data_path, window_length=args.length, 
			horizon=args.horizon, shift=shift, 
			shuffle_data=args.shuffle, train=True)
		test_state_past, test_cont_past, test_action, test_next_obs = utils.createSwatFullDataset(path=data_path, window_length=args.length, 
			horizon=args.horizon, shift=shift, 
			shuffle_data=args.shuffle,train=False,steps=args.steps)
		print(state_past.shape)
		print(test_state_past.shape)
		train_dataset = utils.SwatFullDataset(state_past[0:int(train_split*state_past.shape[0])], cont_past[0:int(train_split*state_past.shape[0])], 
			action[0:int(train_split*state_past.shape[0])], next_obs[0:int(train_split*state_past.shape[0])])
		validation_dataset = utils.SwatFullDataset(state_past[int(train_split*int(state_past.shape[0])):], cont_past[int(train_split*int(state_past.shape[0])):], 
			action[int(train_split*int(state_past.shape[0])):], next_obs[int(train_split*int(state_past.shape[0])):])
		test_dataset = utils.SwatFullDataset(test_state_past, test_cont_past, test_action, test_next_obs)

		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)
		validation_loader = data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)
		test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)
		swatMasterDict[shift] = [train_loader, validation_loader,test_loader]


if(args.dataset=='sarcos'):
	train_dataset = utils.SarcosDataset(train_state, train_control, train_fcontrol, train_fstate)
	validation_dataset = utils.SarcosDataset(valid_state, valid_control, valid_fcontrol, valid_fstate)
	test_dataset = utils.SarcosDataset(test_state, test_control, test_fcontrol, test_fstate)


elif(args.dataset=='swat'):
	train_dataset = utils.SwatDataset(state_past[0:int(train_split*state_past.shape[0])], cont_past[0:int(train_split*state_past.shape[0])], 
		action[0:int(train_split*state_past.shape[0])], next_obs[0:int(train_split*state_past.shape[0])])
	validation_dataset = utils.SwatDataset(state_past[int(train_split*int(state_past.shape[0])):], cont_past[int(train_split*int(state_past.shape[0])):], 
		action[int(train_split*int(state_past.shape[0])):], next_obs[int(train_split*int(state_past.shape[0])):])
	test_dataset = utils.SwatDataset(test_state_past, test_cont_past, test_action, test_next_obs)


elif('narma' in args.dataset):
	train_dataset = utils.NarmaDataset(state_past[0:int(train_split*state_past.shape[0])], cont_past[0:int(train_split*state_past.shape[0])], 
		action[0:int(train_split*state_past.shape[0])], next_obs[0:int(train_split*state_past.shape[0])])
	validation_dataset = utils.NarmaDataset(state_past[int(train_split*int(state_past.shape[0])):], cont_past[int(train_split*int(state_past.shape[0])):], 
		action[int(train_split*int(state_past.shape[0])):], next_obs[int(train_split*int(state_past.shape[0])):])
	test_dataset = utils.NarmaDataset(test_state_past, test_cont_past, test_action, test_next_obs)

elif('pmsm' in args.dataset):
	train_dataset = utils.PMSMDataset(state_past[0:int(train_split*state_past.shape[0])], cont_past[0:int(train_split*state_past.shape[0])], 
		action[0:int(train_split*state_past.shape[0])], next_obs[0:int(train_split*state_past.shape[0])])
	validation_dataset = utils.PMSMDataset(state_past[int(train_split*int(state_past.shape[0])):], cont_past[int(train_split*int(state_past.shape[0])):], 
		action[int(train_split*int(state_past.shape[0])):], next_obs[int(train_split*int(state_past.shape[0])):])
	test_dataset = utils.PMSMDataset(test_state_past, test_cont_past, test_action, test_next_obs)

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)
validation_loader = data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0, worker_init_fn=_init_fn)

print('Data loaded!')

model = mts_model.MVTS(
	embedding_dim=args.embedding_dim,
	input_dim=args.length,
	num_objects=args.num_objects,
	num_cont=args.num_cont,
	tau = args.horizon,
	window=args.window_size, 
	action_dim =args.action_dim,
	final_nodes = args.num_cont,
	steps = args.steps,
	full = args.full,
	isControl=args.isControl,
	dropout = args.dropout,
	stride=args.stride,
	use_condenser=args.use_condenser,
	use_GNN=args.message_pass,
	normalize=args.normalize,
	per_node_MLP=args.per_node_MLP,
	sepCTRL=args.sepCTRL,
	baseline = args.baseline, 
	forecasting_cl = args.forecasting_cl, 
	recurrent = args.recurrent,
	forecasting_M5 =  args.forecasting_M5,
	hierarchical_ls = args.hierarchical_ls,
	pastinfo=args.pastinfo,
	only=args.pastStateOnly or args.pastControlOnly,
	soft_decoder= not args.hard_decoder,
	hard_decoder=args.hard_decoder,
	onlyReLU=args.onlyReLU).to(device)

model.apply(utils.weights_init)
print(model)

optimizer = torch.optim.Adam(
	model.parameters(),
	lr=args.learning_rate)


def eval_changes(predicted, truth, futureControls, inspection=5):
	# predicted,truth (b,c,horizon) futureControls (b,d,horizon)
	loss_ec = torch.zeros(futureControls.shape[1])
	loss_remaining = torch.zeros(futureControls.shape[1])
	
	for j in range(futureControls.shape[0]):
		indices = []
		for k in range(futureControls.shape[1]):
			indices += helper.get_indices(futureControls[j][k])
		indices.sort()
		checkpoints = []
		left = []
		prev = 0
		for point in indices:
			left += [a for a in range(prev,point)]
			prev = point+inspection if point+inspection < args.horizon else args.horizon 
			checkpoints+=[b for b in range(point,prev)]
		checkpoints=np.array(checkpoints)
		left+= [j for j in range(prev,args.horizon)]
		left = np.array(left)
		if(inspection==5):
			global len_change_5
			global len_remaining_5
			len_change_5+=len(checkpoints)
			len_remaining_5+=len(left)
		elif(inspection==10):
			global len_change_10
			global len_remaining_10
			len_change_10+=len(checkpoints)
			len_remaining_10+=len(left)
		elif(inspection==20):
			global len_change_20
			global len_remaining_20
			len_change_20+=len(checkpoints)
			len_remaining_20+=len(left)
		for k in range(futureControls.shape[1]):
			if(len(checkpoints)!=0):
				loss_ec[k] += MSEloss(predicted[j][k][checkpoints],truth[j][k][checkpoints])*len(checkpoints)
			if(len(left)!=0):
				loss_remaining[k] += MSEloss(predicted[j][k][left],truth[j][k][left])*len(left)

	return (loss_ec,loss_remaining)
	

def evaluate_forecasting(model, loader, args, isTest=False, getLoss=False):
	with torch.no_grad():
		model.eval()
		
		for name, param in model.named_parameters():			
			if "transition_model" in name and "weight" in name and "stage" in name:
				print(name)
				#print(param)
				print(torch.norm(param[:,:40], p=2, keepdim = True))
				print(torch.norm(param[:,40:], p=2, keepdim = True))
		#sys.exit()
		
		
		loss_total=0
		objwise_mse=0
		total_len=0
		pred_states = []
		next_states = []    
		input_space=False
		for batch_idx, data_batch in enumerate(loader):
			data_batch = [tensor.to(device) for tensor in data_batch]
			statePast = data_batch[0].float()
			contPast = data_batch[1].float()
			action = data_batch[2].float()
			nextState = data_batch[3].float()

			# multi step 
			loss = 0
			message_loss = 0
			per_obj_loss = 0
			l1_term = 0
			for i in range(action.shape[2]):
				if input_space:				
					state_encoding, cont_encoding, action_encoding = model.getEncodings(torch.cat([statePast[:,:,i:],nextState[:,:,:i] if i==0 else predicted[:,:,max(0,i-args.window_size):i]],dim=-1), torch.cat([contPast[:,:,i:],action[:,:,max(0,i-args.window_size):i]],dim=-1), action[:,:,i].unsqueeze(2))
					pred = model.getTransition(state_encoding, cont_encoding, action_encoding) if args.full else state_encoding + model.getTransition(state_encoding if i==0 else pred,cont_encoding,action_encoding)
					if(i==0):
						predicted = model.decode(pred)	
					else: 
						predicted = torch.cat([predicted, model.decode(pred)], dim=-1)
					#print(predicted.size())
				else:
					state_encoding, cont_encoding, action_encoding = model.getEncodings(torch.cat([statePast[:,:,i:],nextState[:,:,:i]],dim=-1), torch.cat([contPast[:,:,i:],action[:,:,max(0,i-args.window_size):i]],dim=-1), action[:,:,i].unsqueeze(2))
					pred = model.getTransition(state_encoding if i==0 else pred, cont_encoding, action_encoding) if args.full else state_encoding + model.getTransition(state_encoding if i==0 else pred,cont_encoding,action_encoding)
				if(args.message_pass):
					message_loss += model.get_l1_Message()
				mse_loss = MSEloss(model.decode(pred), nextState[:,:,i].unsqueeze(-1)) 
				objwise = helper.per_obj_mse(model.decode(pred), nextState[:,:,i].unsqueeze(-1))
				loss += mse_loss
				per_obj_loss += objwise

				if(args.save_predictions):
					#helper.save_predictions(nextState[:,:,i].unsqueeze(2), pred, i)
					helper.save_predictions(save_folder, used_params, nextState[:,:,i].unsqueeze(-1), model.decode(pred), i)

			if(args.hierarchical_ls and args.layer_l1):
				l1_term = args.l1*helper.getStages_norm(model) 
			elif(args.per_node_MLP and args.layer_l1):
				l1_term = args.l1*helper.getTM_norm(model)

			if(args.soft_decoder_l1):
				decoder_params = [x.view(-1) for x in model.decoder.parameters()][0]
				l1_term = args.decoder_l1*torch.norm(decoder_params,1)/decoder_params.size()[0]		

			if(args.message_pass and getLoss):
				message_loss = args.message_l1*message_loss
				loss += message_loss

			if(getLoss):	
				loss_total += ((loss.item())*len(pred)+l1_term)
			else:
				loss_total += (loss.item())*len(pred)
			objwise_mse += per_obj_loss*len(pred)

			total_len+=len(pred)

		if getLoss:
			model.train()
			return loss_total/float(total_len)

		objwise_mse_list =  (objwise_mse/float(total_len)).tolist()
		dump_object_wise = ['{}'.format(per_object) for per_object in objwise_mse_list]
		save_name = 'result_M3.txt' if not args.sepCTRL else 'results_M4.txt'      
		re_loss = loss_total/float(total_len)
		objwise_mse_list.append(re_loss)
		results = np.expand_dims(np.array(objwise_mse_list),axis=0)
		
		print('Reconstruction Loss {}'.format(loss_total/float(total_len)))
		print('per_obj_mse: '+str(dump_object_wise))
		return results[0]

def train_forecasting(model, args):

	print('Starting model training...')
	best_loss = 1e9
	model.train()

	for epoch in range(1, args.epochs+1):
		train_loss = 0
		mse_total = 0
		for batch_idx, data_batch in enumerate(train_loader):
			optimizer.zero_grad()
			data_batch = [tensor.to(device) for tensor in data_batch]

			statePast = data_batch[0].float()
			contPast = data_batch[1].float()
			action = data_batch[2].float()
			nextState = data_batch[3].float()

			# multi step 
			loss = 0
			l1_term = 0
			message_loss = 0
			for i in range(action.shape[2]):
				state_encoding, cont_encoding, action_encoding = model.getEncodings(torch.cat([statePast[:,:,i:],nextState[:,:,:i]],dim=-1), 
						torch.cat([contPast[:,:,i:],action[:,:,:i]],dim=-1), action[:,:,i].unsqueeze(2))
				pred = model.getTransition(state_encoding if i==0 else pred, cont_encoding, action_encoding) if args.full else state_encoding + model.getTransition(state_encoding if i==0 else pred,cont_encoding,action_encoding)
				if(args.message_pass):
					message_loss += model.get_l1_Message() 

				mse_loss = MSEloss(model.decode(pred), nextState[:,:,i].unsqueeze(-1)) #+ args.l1*torch.norm(decoder_params, 1)
				loss += mse_loss

			if(args.hierarchical_ls and args.layer_l1):
				l1_term = args.l1*helper.getStages_norm(model) 
			elif(args.per_node_MLP and args.layer_l1):
				l1_term = args.l1*helper.getTM_norm(model)
			if(args.soft_decoder_l1):
				decoder_params = [x.view(-1) for x in model.decoder.parameters()][0]
				l1_term = args.decoder_l1*torch.norm(decoder_params,1)/decoder_params.size()[0]		

			if(args.message_pass):
				message_loss = args.message_l1*message_loss
				loss += message_loss
			loss += (loss*len(pred) + l1_term)
			loss.backward()
			train_loss += loss.item()
			optimizer.step()

			if(epoch<120):
				optimizer.zero_grad()				
				if(args.soft_decoder_gl):
					helper.applyGroupLassoDecoder(model, args.learning_rate, args.decoder_gl, args.embedding_dim, args.num_objects)
				if(args.hierarchical_ls and args.layer_gl):
					helper.applyGroupLassoStages(model, args.learning_rate, args.gl, args.embedding_dim)
				elif(args.per_node_MLP and args.layer_gl):
					helper.applyGroupLassoTM(model, args.learning_rate, args.gl, args.embedding_dim)
				elif(args.baseline and args.layer_gl):
					helper.applyGroupLassoBaseLine(model, args.learning_rate, args.gl, args.embedding_dim)
		
			if batch_idx % args.log_interval == 0:
				print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data_batch[0]),len(train_loader.dataset),100. * batch_idx / len(train_loader),loss.item()))
			if batch_idx % args.save_interval == 0 and batch_idx > 0:
				Valoss = evaluate_forecasting(model, validation_loader, args, getLoss = True)
				if Valoss < best_loss:
					best_loss = Valoss
					torch.save(model.state_dict(), model_file)
					print("saving model at:"+str(epoch)+","+str(best_loss))
				model.train()
		avg_loss = train_loss / len(train_loader.dataset)
		print('====> Epoch: {} Average loss: {:.6f} '.format(epoch, avg_loss))

ratio = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.6, 0.7, 0.8,0.9]


def save_results():
	results=[]
	val_results = evaluate_forecasting(model, validation_loader, args)
	test_results = evaluate_forecasting(model, test_loader, args, isTest=True)
	for val in val_results:
		results.append(val)		
	for test in test_results:
		results.append(test)
	if args.ood:
		save_file_name = save_folder+used_params+'numbers_test_ood.csv'
	else:
		save_file_name = save_folder+used_params+'numbers_iid.csv'
	with open(save_file_name,'a') as f:
		f.write(','.join([str(res) for res in results]))
		f.write('\n')

if args.noise:
	model.load_state_dict(torch.load(model_file))
	for r in ratio:
		print(r)
		print(args.pert)
		print("################################################")
		if r==0:		
			print("no noise added!")
			# evaluate_forecasting(model, validation_loader, args)
			print("Testing noise on test set!!!")
			test_results = evaluate_forecasting(model, test_loader, args, isTest=True)
		else:
			# evaluate_forecasting(model, utils.addnoise_ratio(validation_loader, r, args.pert),args)
			print("Testing noise on test set!!!")
			test_results = evaluate_forecasting(model, utils.addnoise_ratio(test_loader, r, args.pert),args, isTest=True)
		save_file_name = save_folder+used_params+'numbers_sensitivitytest.csv'
		with open(save_file_name,'a') as f:
			f.write(','.join([str(res) for res in test_results]))
			f.write('\n')

else:
	#model.load_state_dict(torch.load(model_file))
	train_forecasting(model, args)
	model.load_state_dict(torch.load(model_file,map_location=torch.device('cpu')))
	save_results()
	
	

	
