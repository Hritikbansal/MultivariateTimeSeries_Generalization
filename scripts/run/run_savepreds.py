import os, sys
epochs=0
dec_epochs=0

#root = '/home/hw1415904/data3/codes/CL4MTS/18april_perNodeCL_dropout/CLTS/'
root = '/data/pankaj/CLTS/'
dataset = sys.argv[1]
save_folder = root+'output/CombGen/31Jul2020_1x1train_32test_8x8regimes_wo_minmax_400epochs_wl11/'+dataset
setup = sys.argv[2]
exp_name = sys.argv[3]
ood=False

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
    horizon=2
    num_objects=2
    num_control=2
    stride=1
    bsz=128

else:
    print('unknown dataset')
    exit()



'''
Arguments pertaining to L1 or group lasso

1. Within hidden layers (only applicable to M5 and M9 only)
--layer_l1 : Sets L1
--l1 : hyperparameter that goes with --layer_l1
--layer_gl : Sets group lasso
--gl : hyperparamter that goes with --layer_gl

2. Decoder side 
--soft_decoder_l1 : Sets L1 (applicable to all) 
--decoder_l1 : hyperparameter that goes with --soft_decoder_l1
--soft_decoder_gl : Sets group lasso (applicable to M5 and M9 only)
--decoder_gl : hyperparameter that goes with --soft_decoder_gl
--hard_decoder : No L1 or group lasso (applicable to M5 and M9 only)

B: Baseline, GL: Group LASSO, SC: Separate Control, O: Ours, HD: Hard Decoder, HN: Hard Nodes

'''

gl=1.0
decoder_gl=1.0
for gnn_nodes in [num_control]:
    for emb_dims in [40]:
        for p in [0.0]:
            for h in [5]:#,25,30,35,45,55,80,105]:
                if ood:
                    base_cmd="python train.py --path "+root+"/data/narma/combgen/4/test_ood/ --dataset " + dataset + " --save-folder "+save_folder+" --batch-size "+str(bsz)+" --epochs "+ str(epochs) +" --dec_epochs "+str(dec_epochs) + " --embedding_dim "+ str(emb_dims) + " --nodes "+ str(gnn_nodes) +" --shuffle --dropout " + str(p)+" --length "+str(length)+" --window_size "+str(window_size)+ " --horizon " + str(h) + " --shift "+str(shift) +" --num_objects "+str(num_objects)+" --num_cont "+str(num_control)+" --stride "+str(stride) + " --learning_rate 0.001 --full --ood --save_predictions "
                else:
                    base_cmd="python train.py --path "+root+"/data/narma/combgen/4/train_test_iid/ --dataset " + dataset + " --save-folder "+save_folder+" --batch-size "+str(bsz)+" --epochs "+ str(epochs) +" --dec_epochs "+str(dec_epochs) + " --embedding_dim "+ str(emb_dims) + " --nodes "+ str(gnn_nodes) +" --shuffle --dropout " + str(p)+" --length "+str(length)+" --window_size "+str(window_size)+ " --horizon " + str(h) + " --shift "+str(shift) +" --num_objects "+str(num_objects)+" --num_cont "+str(num_control)+" --stride "+str(stride) + " --learning_rate 0.001 --full --save_predictions "
                if exp_name=='B':
                    os.system(base_cmd + "--name B --baseline")                    
                elif exp_name=='B+GL':
                    os.system(base_cmd + "--name B+GL --baseline --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
                elif exp_name=='O-SC-GL':
                    os.system(base_cmd + "--name O-SC-GL --hierarchical_ls")
                elif exp_name=='O':
                    os.system(base_cmd + "--name O --hierarchical_ls --sepCTRL --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
                elif exp_name=='O+HD':
                    os.system(base_cmd + "--name O+HD --hierarchical_ls --sepCTRL --layer_gl --gl "+str(gl)+" --hard_decoder")
                elif exp_name=='O-SC':
                    os.system(base_cmd + "--name O-SC --hierarchical_ls --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
                elif exp_name=='O-HN':
                    os.system(base_cmd + "--name O-HN --baseline --sepCTRL --layer_gl --gl "+str(gl)+" --soft_decoder_gl --decoder_gl "+str(decoder_gl))
                elif exp_name=='O-GL':
                    os.system(base_cmd + "--name O-GL --hierarchical_ls --sepCTRL")
                elif exp_name=='O-HN-GL':
                    os.system(base_cmd + "--name O-HN-GL --baseline --sepCTRL")
                elif exp_name=='O+HD-GL':
                    os.system(base_cmd + "--name O+HD-GL --hierarchical_ls --sepCTRL --hard_decoder")

# SAMPLE RUN
'''python train.py --path ____ --save-folder ____ --dataset narma --batch-size 64 --shuffle --epochs 1 
--embedding_dim 20 --nodes 2 --length 100 --window_size 100 --horizon 2 --shift 10 --num_objects 2 --num_cont 2 
--stride 1 --full --hierarchical_ls --sepCTRL --l1 1.0'''
