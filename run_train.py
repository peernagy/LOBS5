# CAVE: only for debugging purposes
#import os
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=48'

import argparse
from s5.utils.util import str2bool
from lob.train import train
from lob.dataloading import Datasets
#import tensorflow as tf
import os
import jax
import torch
import cProfile


if __name__ == "__main__":

	#physical_devices = tf.config.list_physical_devices('GPU')
	#tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#tf.config.experimental.set_visible_devices([], "GPU")

	# no GPU use at all
	#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
	#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

	torch.multiprocessing.set_start_method('spawn')

	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=True,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default="LOBS5",
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default="<your-username>",
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='<your data folder>',
						help="name of directory where data is cached")
	parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
						default='lobster-prediction',
						help="dataset name")
	parser.add_argument("--masking", type=str, choices={'causal', 'random'},
						default='causal',  # random
						help="causal or random masking of sequences")
	parser.add_argument("--use_book_data", type=str2bool, default=True,
		     			help="use book data in addition to message data")
	parser.add_argument("--use_simple_book", type=str2bool, default=False,
		     			help="use raw price (-p0) and volume series instead of 'volume image representation'")
	parser.add_argument("--book_transform", type=str2bool, default=True,
		     			help="transform loaded book data to volume image repr. in dataloader")
	parser.add_argument("--book_depth", type=int, default=500,
		     			help="number of tick levels to use in book data [if book_transform=True]")
	parser.add_argument("--restore", type=str,
		     			help="if given restore from given checkpoint dir")
	parser.add_argument("--restore_step", type=int)
	parser.add_argument("--msg_seq_len", type=int, default=500,  # 500
						help="How many past messages to include in each sample")
	parser.add_argument("--n_data_workers", type=int, default=0,
		     			help="number of workers used in DataLoader")

	# Model Parameters
	parser.add_argument("--n_message_layers", type=int, default=2,  # 2
						help="Number of layers after fusing message and book data")
	parser.add_argument("--n_book_pre_layers", type=int, default=1,  # 1
						help="Number of layers taking in raw book data (before projecting dimensions)")
	parser.add_argument("--n_book_post_layers", type=int, default=1,  # 1
						help="Number of book seq layers after projecting book data dimensions")
	parser.add_argument("--n_layers", type=int, default=6,  #6
						help="Number of layers after fusing message and book data")
	parser.add_argument("--d_model", type=int, default=32,  #128, 32, 16
						help="Number of features, i.e. H, "
							 "dimension of layer inputs/outputs")
	parser.add_argument("--ssm_size_base", type=int, default=32,  # 256
						help="SSM Latent size, i.e. P")
	parser.add_argument("--blocks", type=int, default=8,  # 8, 4
						help="How many blocks, J, to initialize with")
	parser.add_argument("--C_init", type=str, default="trunc_standard_normal",
						choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
						help="Options for initialization of C: \\"
							 "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ " \
							 "lecun_normal sample from lecun normal, then multiply by V\\ " \
							 "complex_normal: sample directly from complex standard normal")
	parser.add_argument("--discretization", type=str, default="zoh", choices=["zoh", "bilinear"])
	parser.add_argument("--mode", type=str, default="pool", choices=["pool", "last"],
						help="options: (for classification tasks) \\" \
							 " pool: mean pooling \\" \
							 "last: take last element")
	parser.add_argument("--activation_fn", default="half_glu1", type=str,
						choices=["full_glu", "half_glu1", "half_glu2", "gelu"])
	parser.add_argument("--conj_sym", type=str2bool, default=True,
						help="whether to enforce conjugate symmetry")
	parser.add_argument("--clip_eigs", type=str2bool, default=False,
						help="whether to enforce the left-half plane condition")
	parser.add_argument("--bidirectional", type=str2bool, default=False,  #False,
						help="whether to use bidirectional model")
	parser.add_argument("--dt_min", type=float, default=0.001,
						help="min value to sample initial timescale params from")
	parser.add_argument("--dt_max", type=float, default=0.1,
						help="max value to sample initial timescale params from")

	# Optimization Parameters
	parser.add_argument("--prenorm", type=str2bool, default=True,
						help="True: use prenorm, False: use postnorm")
	parser.add_argument("--batchnorm", type=str2bool, default=True,
						help="True: use batchnorm, False: use layernorm")
	parser.add_argument("--bn_momentum", type=float, default=0.95,
						help="batchnorm momentum")
	parser.add_argument("--bsz", type=int, default=1, #64, (max 16 with full size)
						help="batch size")
	parser.add_argument("--num_devices", type=int, default=jax.device_count(),
		     			help="number of devices (GPUs) to use")
	parser.add_argument("--epochs", type=int, default=2,  #100, 20
						help="max number of epochs")
	parser.add_argument("--early_stop_patience", type=int, default=1000,
						help="number of epochs to continue training when val loss plateaus")
	parser.add_argument("--ssm_lr_base", type=float, default=1e-3,
						help="initial ssm learning rate")
	parser.add_argument("--lr_factor", type=float, default=1,
						help="global learning rate = lr_factor*ssm_lr_base")
	parser.add_argument("--dt_global", type=str2bool, default=False,
						help="Treat timescale parameter as global parameter or SSM parameter")
	parser.add_argument("--lr_min", type=float, default=0,
						help="minimum learning rate")
	parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
	parser.add_argument("--warmup_end", type=int, default=1,
						help="epoch to end linear warmup")
	parser.add_argument("--lr_patience", type=int, default=1000000,
						help="patience before decaying learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--reduce_factor", type=float, default=1.0,
						help="factor to decay learning rate for lr_decay_on_val_plateau")
	parser.add_argument("--p_dropout", type=float, default=0.0,
						help="probability of dropout")
	parser.add_argument("--weight_decay", type=float, default=0.05,
						help="weight decay value")
	parser.add_argument("--opt_config", type=str, default="standard", choices=['standard',
																			   'BandCdecay',
																			   'BfastandCdecay',
																			   'noBCdecay'],
						help="Opt configurations: \\ " \
			   "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\" \
	  	       "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\" \
	  	       "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\")
	parser.add_argument("--jax_seed", type=int, default=1919,
						help="seed randomness")

	#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
	train(parser.parse_args())
	#cProfile.run('train(parser.parse_args())')
