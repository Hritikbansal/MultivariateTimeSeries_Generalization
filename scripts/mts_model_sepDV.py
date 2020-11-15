import numpy as np
import utils 
import torch
from torch import nn
import sys
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def time_pad_representations(states, window_size, num_objects):
	bsz= states.size(0)
	length = states.size(-1)
	pad_init = length%window_size
	if pad_init!=0:
		zeros = torch.zeros((bsz, num_objects, pad_init), dtype=states.dtype).to(device)
		states = torch.cat([zeros, states], dim=-1)
	time_stamps = states.size(-1)//window_size
	time_series=[] 
	for i in range(bsz):
		for j in range(1, time_stamps+1):
			time_series.append(states[i, :, (j-1)*window_size:j*window_size])
	if(time_stamps>0):
		states = torch.stack(time_series,dim=0) # (bsz*time_stamps, len//time_stamps) which can be passed to CNN
	return states, time_stamps

def get_time_representation(states, time_stamps, bsz):
	tot=[states[i*time_stamps:(i+1)*time_stamps].unsqueeze(1) for i in range(bsz)]
	states = torch.cat(tot, dim=1)
	return states


"""__author__ = Hritik Bansal """

class MVTS(nn.Module):
	"""Main module for a Contrastively-trained Structured World Model (C-SWM).
			
	Args:
		embedding_dim: Dimensionality of abstract state space.
		input_dims: Shape of input observation. 
		hidden_dim: Number of hidden units in encoder and transition model.
		num_objects: Number of object slots.
	"""
	def __init__(self, embedding_dim, input_dim,
				 num_objects, num_cont, tau, dropout, window=50, final_nodes=None, num_layers=1, action_dim=None, use_condenser=False, use_GNN=False
				, remove_factored=False, steps=1, normalize=False,per_node_MLP=False, sepCTRL=False, full=False, isControl=False, stride=1
				, baseline = False, forecasting_cl = False, forecasting_M5=False, pastinfo=False,only=False, recurrent=True, hierarchical_ls = False
				, soft_decoder=False, hard_decoder=False):
		super(MVTS, self).__init__()

		self.input_dim = input_dim
		self.embedding_dim = embedding_dim # final latent representation dimension
		self.action_dim = action_dim  # final condensed future action dim
		self.num_objects = num_objects # states in the system
		self.num_cont = num_cont # number of control inputs 
		self.num_layers = num_layers  # LSTM layers
		self.tau = tau
		self.steps = steps
		self.use_condenser=use_condenser
		self.window_size=window
		self.use_GNN=use_GNN
		self.remove_factored=remove_factored
		self.normalize=normalize
		self.dropout = dropout
		self.sepCTRL = sepCTRL
		self.per_node_MLP = per_node_MLP
		self.baseline = baseline
		self.forecasting_cl = forecasting_cl
		self.forecasting_M5 = forecasting_M5
		self.hierarchical_ls = hierarchical_ls
		self.pastinfo = pastinfo
		self.only=only
		self.recurrent = recurrent
		if final_nodes==None:
			self.final_nodes=self.num_cont
		else:
			self.final_nodes=final_nodes
		
		self.pos_loss = 0
		self.neg_loss = 0
		self.full = full
		self.isControl = isControl
		self.stride = stride
		self.soft_decoder = soft_decoder
		self.hard_decoder = hard_decoder

		if(self.sepCTRL):
			for i in range(self.num_cont):
				setattr(self,'control_CNN_{}'.format(i),CNN_Extractor_cont(self.num_cont, self.num_objects, self.num_cont, self.dropout, self.stride, self.sepCTRL))
		else:
			self.control_CNN = CNN_Extractor_cont(self.num_cont, self.num_objects, self.final_nodes, self.dropout, self.stride, self.sepCTRL)

		# self.state_CNN = CNN_Extractor(self.num_objects, self.num_cont, self.stride)  
		for i in range(self.num_objects):
			setattr(self, 'state_CNN_{}'.format(i), CNN_Extractor(self.num_objects, self.num_cont, self.stride))

		if self.use_condenser:
			self.CondenserMLP = nn.Linear(
				self.embedding_dim,
				self.action_dim)
		else:
			self.action_dim  = self.embedding_dim

		######### adaptive size evaluation #########################################
		example = torch.randn(1, 1, self.window_size)
		self.hidden_dim= self.state_CNN_0(example).size(-1)        
		self.adaptive_dim = self.hidden_dim
		####################################################################

		# self.state_LSTM  = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers)
		# Separate encoder for states
		for i in range(self.num_objects):
			setattr(self, 'state_LSTM_{}'.format(i), nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers))
		if(self.sepCTRL):
			for i in range(self.num_cont):
				setattr(self, 'control_LSTM_{}'.format(i), nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers))
		else:
			self.control_LSTM = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=self.num_layers)


		if self.remove_factored:
			if(isControl):
				self.extract_state = nn.Linear(2*self.embedding_dim*self.final_nodes, self.embedding_dim*self.final_nodes)
			else:
				self.extract_state = nn.Linear(self.embedding_dim*self.final_nodes, self.embedding_dim*self.final_nodes)

		if self.use_GNN:
			inp_dim = 2*embedding_dim if isControl else embedding_dim
			self.transition_model = TransitionGNN(
				input_dim=inp_dim,
				hidden_dim=self.embedding_dim,
				action_dim=self.embedding_dim, # for 1 step 
				num_objects=self.final_nodes,
				isControl = self.isControl)
			self.decoder = nn.Linear(self.num_cont*self.embedding_dim, self.num_objects) # 1 step prediction

		elif self.hierarchical_ls:
			self.transition_model = Hierarchical_ls(
				num_cont = self.num_cont,
				num_objects = self.num_objects,
				nodes = self.final_nodes,
				input_size = self.embedding_dim,
				hidden_size = self.embedding_dim)
			if(self.hard_decoder):
				for i in range(self.num_objects):
					setattr(self,'decoder_{}'.format(i), nn.Linear(self.embedding_dim,1))
			else:
				self.decoder = nn.Linear(self.final_nodes*self.embedding_dim, self.num_objects)

		elif self.baseline:
			self.transition_model = nn.Sequential(
				nn.Linear(self.final_nodes*(2*self.embedding_dim), self.final_nodes*self.embedding_dim), nn.Tanh(),
				nn.Linear(self.final_nodes*self.embedding_dim, self.final_nodes*self.embedding_dim), nn.Tanh())
			self.decoder = nn.Linear(self.final_nodes*self.embedding_dim, self.num_objects) # 1 step prediction

		elif self.recurrent:
			hidden_dim = 2*self.embedding_dim if isControl else self.embedding_dim
			if(1):
				for i in range(self.final_nodes):
					setattr(self,'transition_model_{}'.format(i), RNN_transition(input_size = self.embedding_dim, hidden_size = self.embedding_dim))
			#self.transition_model = RNN_transition(input_size = self.embedding_dim, hidden_size = self.embedding_dim) 
			self.decoder = nn.Linear(self.embedding_dim, 1) # 1 step prediction

		else:
			inp_dim = 2*embedding_dim if isControl else embedding_dim
			if(self.per_node_MLP):
				for i in range(self.final_nodes):
					setattr(self,'transition_model_{}'.format(i), MLP_transition(input_dim=inp_dim, hidden_dim=self.embedding_dim,action_dim=self.embedding_dim,num_objects=self.final_nodes,remove_factored=self.remove_factored,isControl=self.isControl,per_node_MLP=self.per_node_MLP))
					if(self.hard_decoder): 
						setattr(self,'decoder_{}'.format(i), nn.Linear(self.embedding_dim,1))
				if(self.soft_decoder):
					self.decoder = nn.Linear(self.final_nodes*self.embedding_dim, self.num_objects) # 1 step prediction					
			else:
				self.transition_model = MLP_transition(
					input_dim=inp_dim,
					hidden_dim=self.embedding_dim,
					action_dim=self.embedding_dim,
					num_objects=self.final_nodes,
					remove_factored=self.remove_factored,
					isControl = self.isControl,
					per_node_MLP = self.per_node_MLP)
				self.decoder = nn.Linear(self.embedding_dim, 1) # 1 step prediction

		self.MSE = nn.MSELoss()
		#self.decoder = Decoder(self.num_cont, self.num_objects, self.final_nodes, self.hidden_dim, self.embedding_dim, self.adaptive_dim, self.window_size, self.tau, self.stride, self.baseline,self.pastinfo, self.only)
	
	
	def noisePerturbed(self, state):
		return state + torch.normal(mean=0.0, std=0.01, size=(state.size(0),state.size(1),state.size(2)))

	def get_control(self,i):
		return getattr(self,'control_CNN_{}'.format(i))
	def get_transition_model(self,i):
		return getattr(self,'transition_model_{}'.format(i))
	def get_decoder(self,i):
		return getattr(self,'decoder_{}'.format(i))
	def get_l1_Message(self):
		return self.transition_model.get_l1_messages()
	def get_LSTM(self,i):
		if(self.sepCTRL):
			return getattr(self,'control_LSTM_{}'.format(i))
		else:
			return self.control_LSTM
	def get_state_LSTM(self,i):
		return getattr(self, 'state_LSTM_{}'.format(i))
	def get_state_CNN(self,i):
		return getattr(self, 'state_CNN_{}'.format(i))

	def apply_ar(self, state_encoding):

		state_h = []
		bsz = state_encoding.shape[1] if len(state_encoding.shape)==4 else state_encoding.shape[0]
		state_encoding = state_encoding if len(state_encoding.shape)==4 else state_encoding.unsqueeze(0)
		h0 = torch.zeros(1, bsz, self.embedding_dim).to(device)
		c0 = torch.zeros(1, bsz, self.embedding_dim).to(device)
		featureState = self.num_cont if self.baseline else self.final_nodes

		for objects in range(featureState):
			_, (h_sn, c_sn) = self.get_state_LSTM(objects)(state_encoding[:, :, objects, :], (h0, c0)) #last value of top_layer hidden unit 
			state_h.append(h_sn[-1].unsqueeze(1))  # its shape is (bsz, emb) 
			
		return torch.cat(state_h, dim=1)   # final shape being bsz, num_obj, emb_dim

	def getEncodings(self, obs, cont, action, predicted=None, isEval=False):

		bsz = obs.size(0)
		num_objects = obs.size(1)

		states, time_stamps_ = time_pad_representations(obs, self.window_size, self.num_objects)
		state_encoding = []
		if(time_stamps_ > 0):
			# state_encoding =  self.state_CNN(states)
			for i in range(self.num_objects):
				state_cnn = self.get_state_CNN(i)
				state_encoding.append(state_cnn(states[:,i,:].unsqueeze(1)))
			state_encoding = torch.cat(state_encoding, dim=1)

			state_encoding = get_time_representation(state_encoding, time_stamps_, bsz)

		# if(isEval):
		# 	state_encoding = predicted if time_stamps_ == 0 else torch.cat([state_encoding,predicted],dim=0)		

		state_encoding = self.apply_ar(state_encoding)

		cont_encoding = []
		action_encoding = []

		cont, time_stamps = time_pad_representations(cont, self.window_size, self.num_cont)	#bsz*time, num_obj, window_size
		assert action.shape[2] == 1 # one step ahead
		action, time_stamps = time_pad_representations(torch.cat([cont[:,:,1:], action],dim=-1), self.window_size, self.num_cont) 
		if self.sepCTRL:
			for i in range(self.num_cont):
				control_cnn = self.get_control(i)
				cont_encoding.append(control_cnn(cont[:,i,:].unsqueeze(1)))
				action_encoding.append(control_cnn(action[:,i,:].unsqueeze(1)))
			cont_encoding = torch.cat(cont_encoding,dim=1)
			action_encoding = torch.cat(action_encoding,dim=1)

		else:	
			cont_encoding = self.control_CNN(cont) # bsz*time, num_nodes, hidden_dim
			action_encoding = self.control_CNN(action)

		cont_encoding = get_time_representation(cont_encoding, time_stamps, bsz) # time_stamps, bsz, num_nodes, hidden_dim
		action_encoding = get_time_representation(action_encoding, time_stamps, bsz)
		cont_h = []
		action_h = []

		h0 = torch.zeros(1, bsz, self.embedding_dim).to(device)
		c0 = torch.zeros(1, bsz, self.embedding_dim).to(device)
		featureControl = self.num_cont if self.baseline else self.final_nodes
	
		for objects in range(featureControl):
			_, (h_cn, c_cn) = self.get_LSTM(objects)(cont_encoding[:, :, objects, :], (h0, c0)) #last value of top_layer hidden unit (bsz, embedding_dim)
			_, (h_an, c_an) = self.get_LSTM(objects)(action_encoding[:, :, objects, :], (h0, c0))#, (h_cn[-1].unsqueeze(0), c_cn[-1].unsqueeze(0)))
			cont_h.append(h_cn[-1].unsqueeze(1))  # its shape is (bsz, emb)  
			action_h.append(h_an[-1].unsqueeze(1))          
			
		cont_encoding= torch.cat(cont_h, dim=1)   # final shape being bsz, num_obj, emb_dim
		action_encoding = torch.cat(action_h, dim=1)

		if self.normalize:
			#print('using normalized embeddings!')
			state_encoding=F.normalize(state_encoding,dim=2)
			cont_encoding=F.normalize(cont_encoding,dim=2)
			action_encoding=F.normalize(action_encoding,dim=2)
			future_state_encoding=F.normalize(future_state_encoding,dim=2)


		return state_encoding, cont_encoding, action_encoding

	def getTransition(self, state, cont, action):
		
		if self.baseline:
			state = state.view(state.size(0), -1)
			action = action.view(action.size(0), -1)
			input_encoding = torch.cat([state, action], dim = -1)
			bsz = input_encoding.size(0) 
			out = self.transition_model(input_encoding)
			return out
		
		elif self.recurrent:
			if self.isControl:
				input_encoding = torch.cat([state, cont], dim= -1)
				return self.extract_state(self.transition_model(action, input_encoding))
			else:
				trans = []
				for i in range(self.final_nodes):
					tm = self.get_transition_model(i)
					trans.append(tm(action[:,i,:].unsqueeze(1), state[:,i,:].unsqueeze(1)))
				return torch.cat(trans,dim=1)

		elif self.per_node_MLP:
			trans = []
			for i in range(self.final_nodes):
				tm = self.get_transition_model(i)
				trans.append(tm(state[:,i,:].unsqueeze(1), action[:,i,:].unsqueeze(1)))
			return torch.cat(trans, dim=1)
		else:
			return self.transition_model(state, action) #bsz, num_obj, embedding_dim

	def decode(self, input):
		if((self.hierarchical_ls or self.per_node_MLP) and self.soft_decoder):
			input = input.view(input.size(0), -1)
			return self.decoder(input).unsqueeze(-1)

		elif((self.hierarchical_ls or self.per_node_MLP) and self.hard_decoder):
			l = []
			for i in range(input.shape[1]):
				l.append(getattr(self, 'decoder_{}'.format(i))(input[:,i,:]).unsqueeze(1))
				#print(l[-1].shape)
			t = torch.cat(l, dim=1)
			#print(t.shape)
			return t
		return self.decoder(input).unsqueeze(-1)


class edge_mlp(torch.nn.Module):
	"""Edge model"""

	def __init__(self, input_dim, hidden_dim):
		super(edge_mlp, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.edge = nn.Sequential(
			nn.Linear(2*input_dim, hidden_dim),
			nn.Tanh())

	def forward(self, input):
		return self.edge(input)

class node_mlp(torch.nn.Module):
	"""node model"""

	def __init__(self, input_dim, action_dim, hidden_dim):
		super(node_mlp, self).__init__()

		self.input_dim = input_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		# self.hidden_dim - comes from edge model
		# self.action_dim - dim of the action applied to the node
		node_input_dim = self.input_dim + self.hidden_dim + self.action_dim
		self.node = nn.Sequential(
			nn.Linear(node_input_dim, input_dim),
			nn.Tanh())

	def forward(self, input):
		return self.node(input)

class TransitionGNN(torch.nn.Module):
	"""GNN-based transition function.
		Here num_objects - number of nodes."""

	def __init__(self, input_dim, hidden_dim, action_dim, num_objects,isControl, act_fn='relu'):
		super(TransitionGNN, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_objects = num_objects
		self.action_dim = action_dim
		self.isControl = isControl

		for i in range(num_objects):
			setattr(self,'node_mlp_{}'.format(i), node_mlp(input_dim = self.input_dim, action_dim = self.action_dim, hidden_dim = self.hidden_dim))

		for i in range(0, self.num_objects-1):
			for j in range(i+1, self.num_objects):
				setattr(self, 'edge_mlp_{}_{}'.format(i,j), edge_mlp(input_dim = self.input_dim, hidden_dim = self.hidden_dim))
				setattr(self, 'edge_mlp_{}_{}'.format(j,i), edge_mlp(input_dim = self.input_dim, hidden_dim = self.hidden_dim))

		self.edge_list = None
		self.batch_size = 0
		self.messages = None

	def get_edge(self, i, j):
		return getattr(self,'edge_mlp_{}_{}'.format(i,j))

	def get_node(self, i):
		return getattr(self,'node_mlp_{}'.format(i))

	def _edge_model(self, node_attr, edge_attr, row, col):
		del edge_attr  # Unused.
		# source - B*N*(N-1)
		# target - B*N*(N-1)
		Dict = {}
		L = zip(row, col)
		for i,j in L:
			i, j = i.item(), j.item()
			if (i%self.num_objects,j%self.num_objects) not in Dict:
				Dict[i%self.num_objects,j%self.num_objects] = []
				Dict[i%self.num_objects,j%self.num_objects].append(torch.cat([node_attr[i].unsqueeze(0),node_attr[j].unsqueeze(0)], dim=1))
			else:
				Dict[i%self.num_objects,j%self.num_objects].append(torch.cat([node_attr[i].unsqueeze(0),node_attr[j].unsqueeze(0)], dim=1))		
		# torch.cat(Dict[a,b], dim=0) (B, 2*input_dim)
		out = [self.get_edge(i,j)(torch.cat(Dict[i,j], dim=0)) for i,j in sorted(Dict.keys())] 
		out = torch.cat(out, dim=1)
		out = out.contiguous().view(-1, self.hidden_dim)
		
		del Dict
		self.messages = out
		return out

	def _node_model(self, node_attr, edge_index, edge_attr):
		if edge_attr is not None:
			row, col = edge_index
			agg = utils.unsorted_segment_sum(
				edge_attr, row, num_segments=node_attr.size(0))
			out = torch.cat([node_attr, agg], dim=1)
		else:
			out = node_attr
		out = out.contiguous().view(-1, self.num_objects, self.input_dim + self.hidden_dim + self.action_dim)
		out = [self.get_node(i)(out[:,i,:].unsqueeze(1)) for i in range(self.num_objects)]
		out = torch.cat(out, dim=1)
		return out

	def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
		# Only re-evaluate if necessary (e.g. if batch size changed).
		if self.edge_list is None or self.batch_size != batch_size:
			self.batch_size = batch_size

			# Create fully-connected adjacency matrix for single sample.
			adj_full = torch.ones(num_objects, num_objects)

			# Remove diagonal.
			adj_full -= torch.eye(num_objects)
			self.edge_list = adj_full.nonzero()

			# Copy `batch_size` times and add offset.
			self.edge_list = self.edge_list.repeat(batch_size, 1)
			offset = torch.arange(
				0, batch_size * num_objects, num_objects).unsqueeze(-1)
			offset = offset.expand(batch_size, num_objects * (num_objects - 1))
			offset = offset.contiguous().view(-1)
			self.edge_list += offset.unsqueeze(-1)

			# Transpose to COO format -> Shape: [2, num_edges].
			self.edge_list = self.edge_list.transpose(0, 1)

			if cuda:
				self.edge_list = self.edge_list.cuda()

		return self.edge_list

	def get_l1_messages(self):

		# messages - B*N*(N-1), H 
		size = self.messages.shape[0]*self.messages.shape[1]
		L1 = torch.abs(torch.sum(self.messages))
		return (L1/size)		

	def forward(self, states, action_vec):

		cuda = states.is_cuda
		batch_size = states.size(0)
		num_nodes = states.size(1)

		# states: [batch_size (B), num_objects, embedding_dim]
		# node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
		# if(self.isControl):
		# 	states = torch.stack([states, control], dim=-1)

		node_attr = states.view(-1, self.input_dim)  #input dim is 2*embedding_dim
		
		edge_attr = None
		edge_index = None

		if num_nodes > 1:
			# edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
			edge_index = self._get_edge_list_fully_connected(
				batch_size, num_nodes, cuda)

			row, col = edge_index
			edge_attr = self._edge_model(
				node_attr, edge_attr, row, col)

		action_vec = action_vec.view(-1, self.action_dim)  
		# Attach action to each state
		node_attr = torch.cat([node_attr, action_vec], dim=-1)

		node_attr = self._node_model(
			node_attr, edge_index, edge_attr)

		# [batch_size, num_nodes, hidden_dim]
		return node_attr.view(batch_size, num_nodes, -1)


class Chomp1d(nn.Module):
	"""
	Removes the last elements of a time series.

	Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
	batch size, `C` is the number of input channels, and `L` is the length of
	the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
	is the number of elements to remove.

	@param chomp_size Number of elements to remove.
	"""
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size]

class CNN_Extractor(nn.Module):
	def __init__(self, num_objects, final_nodes, stride):
		self.num_objects=num_objects
		self.final_nodes=final_nodes
		self.stride = stride
		super(CNN_Extractor, self).__init__()
		self.CNN =  nn.Sequential(
			nn.Conv1d(1, 4*self.num_objects, kernel_size=5, stride=self.stride, padding=4),
			Chomp1d(4),
			nn.BatchNorm1d(4*self.num_objects), 
			nn.LeakyReLU(), 
			nn.Conv1d(4*self.num_objects, 4*self.num_objects, kernel_size=5, stride=self.stride, padding=4),
			Chomp1d(4),
			nn.BatchNorm1d(4*self.num_objects), 
			nn.LeakyReLU(),
			nn.Conv1d(4*self.num_objects, 1, kernel_size=5, stride=self.stride, padding=4),
			Chomp1d(4),
			nn.BatchNorm1d(1), 
			nn.Tanh())

	def forward(self, input):
		return self.CNN(input)

class CNN_Extractor_cont(nn.Module):
	def __init__(self, num_cont, num_objects, final_nodes, dropout, stride, sepCTRL=False):
		self.num_objects=num_objects
		self.final_nodes=final_nodes
		self.num_cont = num_cont 
		self.stride = stride
		self.sepCTRL = sepCTRL
		super(CNN_Extractor_cont, self).__init__()
		self.dropoutlayer = nn.Dropout(p=dropout)
		if self.sepCTRL:
			if num_cont!=final_nodes:
				print("Final nodes should be same as the num_cont in case of separate controls!!")
				sys.exit()
			self.CNN =  nn.Sequential(
				nn.Conv1d(1, 4*self.num_cont,5, stride=self.stride, padding=4),Chomp1d(4),
				nn.BatchNorm1d(4*self.num_cont), nn.LeakyReLU(), 
				nn.Conv1d(4*self.num_cont, 4*self.num_objects, 5, stride=self.stride, padding=4),Chomp1d(4),
				nn.BatchNorm1d(4*self.num_objects), nn.LeakyReLU(),
				nn.Conv1d(4*self.num_objects, 1, 5, stride=self.stride, padding=4),Chomp1d(4),
				nn.BatchNorm1d(1), 
				nn.Tanh())
		else:	
			self.CNN =  nn.Sequential(
				nn.Conv1d(self.num_cont, 4*self.num_cont,5, stride=self.stride, padding=4), Chomp1d(4),
				nn.BatchNorm1d(4*self.num_cont), nn.LeakyReLU(), 
				nn.Conv1d(4*self.num_cont, 4*self.num_objects, 5, stride=self.stride, padding=4),Chomp1d(4),
				nn.BatchNorm1d(4*self.num_objects), nn.LeakyReLU(),
				nn.Conv1d(4*self.num_objects, self.final_nodes, 5, stride=self.stride, padding=4),Chomp1d(4),
				nn.BatchNorm1d(self.final_nodes), 
				nn.Tanh())

	def forward(self, input):
		x = self.dropoutlayer(input)
		return self.CNN(x)

class Decoder(nn.Module):
	def __init__(self, num_cont, num_objects, final_nodes, hidden_dim, embedding_dim, adaptive_dim, window_size, tau, stride, baseline,pastinfo,only=False):
		self.num_objects = num_objects
		self.final_nodes = final_nodes
		self.num_cont = num_cont
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.adaptive_dim = adaptive_dim
		self.window_size = window_size
		self.tau = tau
		super(Decoder, self).__init__()
		self.stride = stride
		self.baseline = baseline
		self.pastinfo = pastinfo
		self.only = only
		self.linear1 = nn.Linear(self.embedding_dim if (self.pastinfo or self.only) else self.hidden_dim, self.hidden_dim)
		self.act1 = nn.Tanh()
		self.act2 = nn.LeakyReLU()
		self.ln = nn.LayerNorm(hidden_dim)

		self.network1 = nn.Sequential(self.linear1,
			self.ln, self.act1)

		self.deconv1 = nn.ConvTranspose1d(2*self.num_cont if self.pastinfo else self.final_nodes, 4*self.num_objects, kernel_size=4, stride=self.stride, padding=1)
		self.deconv2 = nn.ConvTranspose1d(4*self.num_objects,4*self.num_objects, kernel_size=8, stride=self.stride, padding=3)
		self.deconv3 = nn.ConvTranspose1d(4*self.num_objects, self.num_objects, kernel_size=10, stride=self.stride, padding=4) 

		self.network2 = nn.Sequential(self.deconv1, self.act2, self.deconv2, self.act2, self.deconv3)

		self.net = nn.Sequential(self.network1,self.network2)
		
		######### adjusted size evaluation #########################################
		# example = torch.randn(1, self.final_nodes, self.hidden_dim)
		# self.adjusted_dim= self.getDim(example).size(-1)        
		# ####################################################################	
		# self.linear3 = nn.Linear(self.adjusted_dim, self.tau)
		
	def getDim(self, input):
		return self.net(input)

	def forward(self, state_encoding):
		# state_encoding (bsz,final_nodes,embedding_dim)
		output1 = self.network1(state_encoding)
		output2 = self.network2(output1)
		return output2

class linearDecoder(nn.Module):
	def __init__(self, num_cont, num_objects, final_nodes, hidden_dim, embedding_dim, adaptive_dim, window_size, tau, baseline,pastinfo,only=False):
		self.num_objects = num_objects
		self.final_nodes = final_nodes
		self.num_cont = num_cont
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.adaptive_dim = adaptive_dim
		self.window_size = window_size
		self.tau = tau
		self.baseline = baseline
		self.pastinfo = pastinfo
		self.only = only
		super(linearDecoder, self).__init__()
		if(self.baseline):
			self.linear = nn.Linear(self.num_cont*self.hidden_dim, self.num_objects*self.tau)
		elif(self.pastinfo):
			self.linear = nn.Linear(2*self.num_cont*self.embedding_dim, self.num_objects*self.tau)
		elif(only):
			self.linear = nn.Linear(self.num_cont*self.embedding_dim, self.num_objects*self.tau)
		else:
			self.linear = nn.Linear(self.final_nodes*self.hidden_dim, self.num_objects*self.tau)
	
	def forward(self, input):
		# input (bsz,final_nodes,embedding_dim)
		bsz = input.size(0)
		return self.linear(input.view(bsz,input.size(1)*input.size(2))).view(bsz,self.num_objects,-1)


class MLP_transition(nn.Module):
	def __init__(self, input_dim, hidden_dim, action_dim, num_objects, act_fn='relu', remove_factored=True, isControl=False, per_node_MLP=False):
		super(MLP_transition, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.num_objects = num_objects
		self.action_dim = action_dim
		self.remove_factored=remove_factored
		self.isControl = isControl
		self.per_node_MLP = per_node_MLP

		if remove_factored:
			self.input_dim = self.input_dim*self.num_objects
			self.action_dim = self.action_dim*self.num_objects

		node_input_dim = self.input_dim + self.action_dim

		self.node_mlp = nn.Sequential(   #node_input_dim-----> input_dim == 3*emb_dim +  delta -> 2*embedding_dim --> embedding_dim
			nn.Linear(node_input_dim, self.input_dim),
			nn.Tanh())

	def forward(self,states, action_vec):
		updated=[]
		bsz = states.size(0)

		if self.remove_factored:
			states = states.view(bsz, -1)
			action_vec = action_vec.view(bsz, -1)
			return self.node_mlp(torch.cat([states, action_vec], dim=-1)) #(bsz, dim)

		for i in range(1 if self.per_node_MLP else self.num_objects):
			concat_ = torch.cat([states[:, i, :], action_vec[:,i,:]], dim=-1)
			updated.append(self.node_mlp(concat_).unsqueeze(1))

		return torch.cat(updated, dim=1)

'''__author__  Hritik Bansal'''

class RNN_transition(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		super(RNN_transition, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.cell = nn.GRU(hidden_size = self.hidden_size, input_size = self.input_size, num_layers = self.num_layers)

	def forward(self, inp, h0):

		bsz = inp.shape[0]
		assert inp.shape[2] == self.input_size
		assert h0.shape[2] == self.hidden_size

		state_h = []
		for i in range(inp.shape[1]):
			out, hn = self.cell(inp[:,i,:].view(-1, bsz, self.input_size).contiguous(), h0[:,i,:].view(-1, bsz, self.hidden_size).contiguous())
			# shape of hn is (1,b,h)
			state_h.append(hn[0].unsqueeze(1))
		return torch.cat(state_h, dim=1)



##############################################################################
'''___author__ = HB
CODE FOR HIERARCHICAL LATENT STRUCTURE DISCUSSED ON 16TH JULY 2020'''
##############################################################################

class first_stage(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(first_stage, self).__init__()

		self.input_size = 2*input_size  # One feature map from state and another from action
		self.hidden_size = hidden_size

		self.net = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.Tanh())

	def forward(self, input):
		# print("First")
		return self.net(input)

# class second_stage(nn.Module):
# 	def __init__(self, nodes, hidden_size):
# 		super(second_stage, self).__init__()

# 		self.nodes = nodes
# 		self.hidden_size = hidden_size

# 		self.net = nn.Sequential(
# 			nn.Linear(self.nodes*self.hidden_size, self.hidden_size),
# 			nn.Tanh())

# 	def forward(self, input):
# 		# print("Second")
# 		return self.net(input)

class extract_state_ls(nn.Module):
	def __init__(self, nodes, hidden_size):
		super(extract_state_ls, self).__init__()

		self.nodes = nodes
		self.hidden_size = hidden_size

		self.net = nn.Sequential(
			nn.Linear(self.nodes*self.hidden_size, self.hidden_size),
			nn.Tanh())

	def forward(self, input):
		return self.net(input)

class Hierarchical_ls(nn.Module):
	def __init__(self, num_cont, num_objects, nodes, input_size, hidden_size):
		super(Hierarchical_ls, self).__init__()
		self.num_cont = num_cont
		self.num_objects = num_objects
		self.nodes = nodes
		self.input_size = input_size
		self.hidden_size = hidden_size

		for i in range(self.nodes):
			for j in range(i, self.nodes):
				setattr(self, 'first_stage_{}_{}'.format(i,j), first_stage(input_size = self.input_size, hidden_size = self.hidden_size))
				if(i!=j):
					setattr(self, 'first_stage_{}_{}'.format(j,i), first_stage(input_size = self.input_size, hidden_size = self.hidden_size))
			setattr(self, 'extract_state_ls_{}'.format(i), extract_state_ls(nodes = self.nodes, hidden_size = self.hidden_size))

		# for i in range(self.nodes):
		# 	for j in range(i, self.nodes):
		# 		setattr(self, 'second_stage_{}_{}'.format(i,j), second_stage(nodes = self.nodes, hidden_size = self.hidden_size))
		# 		if(i!=j):
		# 			setattr(self, 'second_stage_{}_{}'.format(j,i), second_stage(nodes = self.nodes, hidden_size = self.hidden_size))
	
	def get_first_stage(self, i, j):
		return getattr(self, 'first_stage_{}_{}'.format(i,j))

	# def get_second_stage(self, i, j):
	# 	return getattr(self, 'second_stage_{}_{}'.format(i,j))

	def get_extract_state(self, i):
		return getattr(self, 'extract_state_ls_{}'.format(i))


	def forward(self, state, action):

		assert state.shape[1] == action.shape[1]

		for i in range(state.shape[1]):
			setattr(self, 'node_first_stage_{}'.format(i), [])
			# setattr(self, 'node_second_stage_{}'.format(i), [])

		for i in range(state.shape[1]):
			for j in range(i, state.shape[1]):
				getattr(self,'node_first_stage_{}'.format(j)).append(self.get_first_stage(i,j)(torch.cat((state[:,i,:], action[:,i,:]), dim=-1)))
				# print(getattr(self,'node_first_stage_{}'.format(j))[-1].shape)
				if(i!=j):
					getattr(self,'node_first_stage_{}'.format(i)).append(self.get_first_stage(j,i)(torch.cat((state[:,j,:], action[:,j,:]), dim=-1)))
					# print(getattr(self,'node_first_stage_{}'.format(i))[-1].shape)


		# nodes_first_stage = [torch.cat(getattr(self, 'node_first_stage_{}'.format(i)), dim=-1) for i in range(state.shape[1])]

		# for i in range(state.shape[1]):
		# 	for j in range(i, state.shape[1]):
		# 		getattr(self,'node_second_stage_{}'.format(j)).append(self.get_second_stage(i,j)(nodes_first_stage[i]))
		# 		# print(getattr(self,'node_second_stage_{}'.format(j))[-1].shape)
		# 		if(i!=j):
		# 			getattr(self,'node_second_stage_{}'.format(i)).append(self.get_second_stage(j,i)(nodes_first_stage[j]))
		# 			# print(getattr(self,'node_second_stage_{}'.format(i))[-1].shape)

		# nodes_second_stage = [self.get_extract_state(i)(torch.cat(getattr(self, 'node_second_stage_{}'.format(i)), dim=-1)).unsqueeze(1) for i in range(state.shape[1])]
		nodes_first_stage = [self.get_extract_state(i)(torch.cat(getattr(self, 'node_first_stage_{}'.format(i)), dim=-1)).unsqueeze(1) for i in range(state.shape[1])]

		return torch.cat(nodes_first_stage, dim=1)
