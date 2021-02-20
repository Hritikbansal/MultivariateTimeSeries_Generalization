import os, sys


dec_epochs=0

dataset = sys.argv[1]
file_name = sys.argv[2]
setup = sys.argv[3]
exp_name = sys.argv[4]
epochs = sys.argv[5]
ood= int(sys.argv[6])
seed = int(sys.argv[7])
noise = int(sys.argv[8])
pert = int(sys.argv[9])

root = "~/scratch/gantavya/chacha/MTSG/"

if dataset=='swat':
	shift=100
	length=1000 #600
	window_size=200 #600
	horizon=200 #300
	num_objects=2 if setup=='1' or setup=='3' else 3 #25
	num_control=2 if setup=='1' or setup=='3' else 3 #12
	stride=2
	bsz=16
	setup=setup

elif dataset=='sarcos':
	shift=6
	length=16
	window_size=16
	horizon=16
	num_objects=7
	num_control=7
	stride=1
	bsz=16

elif dataset=='swat_full':
	shift = 10
	length = 300
	window_size = 60#print('unknown dataset')
	horizon=60#print('unknown dataset')
	num_objects=8#print('unknown dataset')
	num_control=11#print('unknown dataset')
	stride =1#print('unknown dataset')
	bsz=16#print('unknown dataset')
	#exit()
	
elif 'narma' in dataset:
	shift=10
	length=11
	window_size=11
	num_objects=2
	num_control=2
	stride=1
	bsz=128

elif 'PMSM' in dataset:
	shift = 5
	length = 10
	window_size = 10
	num_objects = 2
	num_control = 2
	stride = 1
	bsz = 128
else:
	print('unknown dataset')
	exit()


'''
Arguments pertaining to L1 or group lasso

1. Within hidden layers (only applicable to structured latent transition model)
--layer_l1 : Sets L1
--l1 : hyperparameter that goes with --layer_l1
--layer_gl : Sets group lasso
--gl : hyperparamter that goes with --layer_gl

2. Decoder side 
--soft_decoder_l1 : Sets L1 (applicable to all) 
--decoder_l1 : hyperparameter that goes with --soft_decoder_l1
--soft_decoder_gl : Sets group lasso (only applicable to structured latent transition model)
--decoder_gl : hyperparameter that goes with --soft_decoder_gl
--hard_decoder : No L1 or group lasso (only applicable to structured latent transition model)

B: Baseline, GL: Group LASSO, SC: Separate Control, O: Ours, HD: Hard Decoder, HN: Hard Nodes

'''

gl=1.0
decoder_gl=1.0
for gnn_nodes in [num_control]:
	for emb_dims in [40]:
		for p in [0.0]:
			for h in [5]:#10, 15, 20, 25, 30, 35, 40, 45, 50]: #,20,60,100,140]:
				if ood:
					base_cmd="python train.py --path "+ root+ "data/test_ood_alpha_0.0_0.1/ --file_name " + file_name + " --dataset " + dataset + " --save_folder . --batch-size "+str(bsz)+" --epochs "+ str(epochs) +" --dec_epochs "+str(dec_epochs) + " --embedding_dim "+ str(emb_dims) + " --nodes "+ str(gnn_nodes) +" --shuffle --dropout " + str(p)+" --length "+str(length)+" --window_size "+str(window_size)+ " --horizon " + str(h) + " --shift "+str(shift) +" --num_objects "+str(num_objects)+" --num_cont "+str(num_control)+" --stride "+str(stride) + " --learning_rate 0.001 --full --ood --seed "+str(seed)+ " " #" --onlyReLU "
				else:
					if noise:
						base_cmd="python train.py --path "+ root + "data/train_test_iid_alpha_0.4_0.7/ --file_name " + file_name + " --dataset " + dataset + " --save_folder . --batch-size "+str(bsz)+" --epochs "+ str(epochs) +" --dec_epochs "+str(dec_epochs) + " --embedding_dim "+ str(emb_dims) + " --nodes "+ str(gnn_nodes) +" --shuffle --dropout " + str(p)+" --length "+str(length)+" --window_size "+str(window_size)+ " --horizon " + str(h) + " --shift "+str(shift) +" --num_objects "+str(num_objects)+" --num_cont "+str(num_control)+" --stride "+str(stride) + " --learning_rate 0.001 --full --seed "+str(seed)+ " --noise --pert " + str(pert) +" " #" --onlyReLU "
					else:
						base_cmd="python train.py --path "+ root + "data/train_test_iid_alpha_0.4_0.7/ --file_name " + file_name + " --dataset " + dataset + " --save_folder . --batch-size "+str(bsz)+" --epochs "+ str(epochs) +" --dec_epochs "+str(dec_epochs) + " --embedding_dim "+ str(emb_dims) + " --nodes "+ str(gnn_nodes) +" --shuffle --dropout " + str(p)+" --length "+str(length)+" --window_size "+str(window_size)+ " --horizon " + str(h) + " --shift "+str(shift) +" --num_objects "+str(num_objects)+" --num_cont "+str(num_control)+" --stride "+str(stride) + " --learning_rate 0.001 --full --seed "+str(seed)+" "
				
				if exp_name=='B':
					os.system(base_cmd + "--name B --baseline")
				elif exp_name=='O-HN-GL' or exp_name=='B+SC':
					os.system(base_cmd + "--name B+SC --baseline --sepCTRL")
				elif exp_name=='O-GL' or exp_name=='B+SC+HN':
					os.system(base_cmd + "--name B+SC+HN --hierarchical_ls --sepCTRL")
				elif exp_name=='O+HD-GL' or exp_name=='B+SC+HN+HD':
					os.system(base_cmd + "--name B+SC+HN+HD --hierarchical_ls --sepCTRL --hard_decoder")                    
				elif exp_name=='B+GL':
					os.system(base_cmd + "--name B+GL --baseline --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
				elif exp_name=='O-SC-GL':
					os.system(base_cmd + "--name O-SC-GL --hierarchical_ls")
				elif exp_name=='O':
					os.system(base_cmd + "--name O --hierarchical_ls --sepCTRL --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
				elif exp_name=='B+SC+HN+LGL':
					os.system(base_cmd + "--name B+SC+HN+LGL --hierarchical_ls --sepCTRL --layer_gl --gl "+str(gl))
				elif exp_name=='O+HD':
					os.system(base_cmd + "--name O+HD --hierarchical_ls --sepCTRL --layer_gl --gl "+str(gl)+" --hard_decoder")
				elif exp_name=='O-SC':
					os.system(base_cmd + "--name O-SC --hierarchical_ls --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
				elif exp_name=='O-HN':
					os.system(base_cmd + "--name O-HN --baseline --sepCTRL --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
				

# SAMPLE RUN
'''python train.py --path ____ --save-folder ____ --dataset narma --batch-size 64 --shuffle --epochs 1 
--embedding_dim 20 --nodes 2 --length 100 --window_size 100 --horizon 2 --shift 10 --num_objects 2 --num_cont 2 
--stride 1 --full --hierarchical_ls --sepCTRL --l1 1.0'''
