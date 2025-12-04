# CAVE: only for debugging purposes
import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=48'
# no GPU use at all
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# allocate and de-allocate memory as needed (SLOW)
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# TODO: change this if num_devices changes (is less than all of the available ones11)
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["NCCL_DEBUG"]="INFO"

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"
if __name__ == "__main__":
	pass
else:
	# Forces all generated worker processes to not run on GPU.
	#  Required at this high level, because the init func in the 
	# worker spawn interface happens after init. of the CUDA process. 
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	os.environ["JAX_PLATFORMS"] = "cpu"

from lob.dataloading import Datasets

if __name__ == "__main__":
	import argparse
	from s5.utils.util import str2bool
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
	import time
	time.sleep(1)
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
	time.sleep(1)

	#physical_devices = tf.config.list_physical_devices('GPU')
	#tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#tf.config.experimental.set_visible_devices([], "GPU")

	parser = argparse.ArgumentParser()

	parser.add_argument("--USE_WANDB", type=str2bool, default=True,
						help="log with wandb?")
	parser.add_argument("--wandb_project", type=str, default="LOBS5v2",
						help="wandb project name")
	parser.add_argument("--wandb_entity", type=str, default="sasrey",
						help="wandb entity name, e.g. username")
	parser.add_argument("--dir_name", type=str, default='./data/LOBS5v2Cached/',
						help="name of directory where data is cached")
	parser.add_argument("--test_dir_name", type=str, default=None,
						help="separate test data directory (if None, split from train data)")
	parser.add_argument("--data_mode", type=str, choices=['preproc', 'encoded'],
						default='preproc',
						help="data loading mode: 'preproc' (encode on-the-fly) or 'encoded' (load pre-encoded data)")
	parser.add_argument("--dataset", type=str, choices=Datasets.keys(),
						default='lobster-prediction',
						help="dataset name")
	parser.add_argument("--masking", type=str, choices={'causal', 'random','last_pos','none'},
						default='causal',  # random
						help="causal, random or last position masking of sequences")
	parser.add_argument("--use_book_data", type=str2bool, default=False,
		     			help="use book data in addition to message data")
	parser.add_argument("--merging", type=str, choices={'projected', 'padded'},
						default='projected', 
						help="Method for merging the book model with the message model. Cannot use RNN mode with projected mode.")
	parser.add_argument("--use_simple_book", type=str2bool, default=False,
		     			help="use raw price (-p0) and volume series instead of 'volume image representation'")
	parser.add_argument("--book_transform", type=str2bool, default=False,
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
	parser.add_argument("--mode", type=str, default="none", choices=["none","pool", "last","ema"],
						help="options: (for classification tasks) \\" \
							 " none: no aggregation, raw output at decoder stage \\" \
							 " pool: mean pooling \\" \
							 "last: take last element \\" \
							 "ema : take exponential moving avg across all")
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
	parser.add_argument("--per_gpu_bsz", type=int, default=16, #64, (max 16 with full size)
						help="Per-GPU batch size (each device processes this many samples). "
						     "Effective batch size = per_gpu_bsz × num_devices")
	# Legacy support: also accept --bsz for backwards compatibility
	parser.add_argument("--bsz", type=int, default=None,
						help="(DEPRECATED: use --per_gpu_bsz) batch size")
	parser.add_argument("--num_devices", type=int, default=1,
		     			help="number of devices (GPUs) to use")
	parser.add_argument("--epochs", type=int, default=100,  #100, 20
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
	parser.add_argument("--reduce_factor", type=float, default=0.8,
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
	parser.add_argument("--debug_loading", type=str2bool, default=False,
						help="Set flag to True to skip any training and just run the loading process.")
	parser.add_argument("--enable_profiler", type=str2bool, default=False,
					help="Set flag to True to use the TB profiler.")
	parser.add_argument("--curtail_epochs", type=int, default=None,
				help="End epoch after n steps. Default is None, never. ")
	parser.add_argument("--random_offsets_train", type=str2bool, default=True,
				help="Whether or not the training data is offset randomly at each epoch.")
	parser.add_argument("--shuffle_train", type=str2bool, default=True,
				help="Whether or not the training data shuffled.")
	parser.add_argument("--ignore_times", type=str2bool, default=False,
                    help="Ignore the loss due to predicting the time.")
	parser.add_argument("--debug_overfit", type=str2bool, default=False,
				help="Runs the training loop in overfit mode on a single batch of data. Validation and testing are from the same set. ")
	parser.add_argument("--log_ce_tables", type=str2bool, default=False,
				help="Logs the CE values on a per token level to wandb. Memory intensive.")
	parser.add_argument("--use_remat", type=str2bool, default=False,
				help="Use gradient checkpointing (rematerialization) to reduce runtime memory. "
				     "Trades compute for memory by recomputing activations during backward pass.")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
				help="Number of gradient accumulation steps. Default is 1 (no accumulation). "
				     "Effective batch size = per_gpu_bsz * num_devices * gradient_accumulation_steps.")

	args = parser.parse_args()

	# ============================================
	# Step 1: Detect execution environment (Slurm multi-node vs single machine)
	# ============================================
	is_slurm_multi_node = int(os.environ.get('SLURM_NNODES', '1')) > 1

	if is_slurm_multi_node:
		# Multi-node environment: Don't set CUDA_VISIBLE_DEVICES, let Slurm manage it
		print(f"[*] Detected Slurm multi-node environment ({os.environ.get('SLURM_NNODES')} nodes)")
		print(f"[*] Using Slurm GPU allocation: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
	else:
		# Single machine environment: Set CUDA_VISIBLE_DEVICES based on num_devices
		if hasattr(args, 'num_devices') and args.num_devices > 0:
			visible_devices = ",".join(str(i) for i in range(args.num_devices))
			os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
			print(f"[*] Single machine mode: Setting CUDA_VISIBLE_DEVICES={visible_devices}")
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = "0"
			print("[*] Single machine mode: Setting CUDA_VISIBLE_DEVICES=0 (default)")

	# ============================================
	# Step 2: Import torch (used for DataLoader)
	# ============================================
	import torch
	torch.multiprocessing.set_start_method('spawn')

	# ============================================
	# Step 3: JAX Distributed Initialization
	# ============================================
	import jax
	import jax.numpy as jnp
	from jax import config
	from jax.experimental import multihost_utils

	# DISABLED: Forcing FP32 causes 2x slowdown by disabling TF32 on Hopper GPUs
	# Let JAX use default "fastest" precision which allows TF32
	# print("[*] Using FP32 full precision training")
	# config.update("jax_default_matmul_precision", "float32")

	if is_slurm_multi_node or os.environ.get('JAX_COORDINATOR_ADDRESS'):
		# Multi-node mode: Explicitly initialize JAX distributed
		coord = os.environ.get('JAX_COORDINATOR_ADDRESS')
		pid = int(os.environ.get('JAX_PROCESS_INDEX', os.environ.get('SLURM_PROCID', '0')))
		pcnt = int(os.environ.get('JAX_PROCESS_COUNT', os.environ.get('SLURM_NNODES', '1')))
		if not coord:
			raise RuntimeError('JAX_COORDINATOR_ADDRESS is not set for multi-node run')

		# CRITICAL: Infer local visible GPUs and explicitly specify local_device_ids
		# This workaround prevents JAX from only assigning 1 GPU per process
		cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
		if cvd and cvd != '-1':
			try:
				n_local = len([d for d in cvd.split(',') if d.strip() != ''])
			except Exception:
				n_local = 1
		else:
			n_local = 1
		local_device_ids = list(range(n_local))

		print(f"\n[*] Initializing JAX distributed: coord={coord}, pid={pid}, pcnt={pcnt}")
		print(f"[*] CRITICAL: Explicitly specifying local_device_ids={local_device_ids} (from CUDA_VISIBLE_DEVICES='{cvd}')")

		jax.distributed.initialize(
			coordinator_address=coord,
			num_processes=pcnt,
			process_id=pid,
			local_device_ids=local_device_ids  # CRITICAL WORKAROUND!
		)
		is_distributed = True
		process_index = jax.process_index()
		process_count = jax.process_count()
		local_device_count = jax.local_device_count()

		print(f"[*] JAX distributed mode enabled:")
		print(f"    Process ID: {process_index}/{process_count}")
		print(f"    GPUs per process: {local_device_count}")
		print(f"    Total global GPUs: {len(jax.devices())}")

		# ========== Synchronization Barrier 1: After JAX Distributed Init ==========
		print(f"[*] Synchronization barrier 1/4: JAX distributed init complete, waiting for all nodes...")
		try:
			import time
			sync_start = time.time()
			multihost_utils.sync_global_devices("jax_distributed_init")
			sync_time = time.time() - sync_start
			print(f"[*] ✓ All {process_count} nodes completed JAX distributed init (took: {sync_time:.2f}s)")
		except Exception as e:
			print(f"[ERROR] Synchronization barrier 1/4 failed: {e}")
			print(f"[ERROR] Process {process_index} cannot sync after JAX distributed init")
			raise

		# ========== Device Topology Validation ==========
		print(f"\n[*] Validating device topology consistency...")

		expected_total_devices = process_count * local_device_count
		actual_total_devices = len(jax.devices())

		if actual_total_devices != expected_total_devices:
			print(f"\n{'='*80}")
			print(f"❌ Device topology mismatch detected!")
			print(f"{'='*80}")
			print(f"Expected: {process_count} processes × {local_device_count} GPUs/process = {expected_total_devices} total GPUs")
			print(f"Actual: {actual_total_devices} total GPUs")
			print(f"Difference: {expected_total_devices - actual_total_devices} GPUs missing")
			print(f"\nPossible causes:")
			print(f"  • Some nodes have GPU hardware failures (e.g., ECC errors)")
			print(f"  • Some nodes have GPUs occupied by other processes")
			print(f"  • CUDA device initialization failed")
			print(f"{'='*80}\n")

			# Force exit to avoid wasting compute time
			import sys
			sys.exit(1)
		else:
			print(f"    ✓ Device topology validation passed")
			print(f"    ✓ All {process_count} processes have {local_device_count} GPUs")
			print(f"    ✓ Total {actual_total_devices} GPUs available")
		print(f"")

		# CRITICAL: In multi-node mode, num_devices should equal local device count
		args.num_devices = local_device_count
		print(f"    Adjusted num_devices: {args.num_devices} (local device count)")
	else:
		# Single machine mode
		is_distributed = False
		process_index = 0
		process_count = 1
		print(f"\n[*] Single machine mode")

	# Add distributed info to args
	args.is_distributed = is_distributed
	args.process_index = process_index
	args.process_count = process_count

	# IMPORTANT: Handle legacy bsz parameter
	if hasattr(args, 'bsz') and args.bsz is not None and not hasattr(args, 'per_gpu_bsz'):
		args.per_gpu_bsz = args.bsz
		print(f"\n[*] Using legacy 'bsz' parameter as 'per_gpu_bsz'")
	elif hasattr(args, 'bsz') and args.bsz is not None:
		# If both exist, bsz overrides per_gpu_bsz
		args.per_gpu_bsz = args.bsz
		print(f"\n[*] Legacy 'bsz' parameter overrides 'per_gpu_bsz'")

	# Calculate effective batch size based on per-GPU batch size and number of devices
	# effective_bsz is the total number of samples processed per process
	# per_gpu_bsz is the number of samples each individual GPU processes
	args.effective_bsz = args.per_gpu_bsz * args.num_devices

	print(f"\n[*] Batch size configuration:")
	print(f"    Per-GPU batch size: {args.per_gpu_bsz}")
	print(f"    Devices per process: {args.num_devices}")
	print(f"    Effective batch size (per process): {args.effective_bsz}")
	print(f"    Gradient accumulation steps: {args.gradient_accumulation_steps}")
	if is_distributed:
		global_effective_bsz = args.effective_bsz * args.process_count * args.gradient_accumulation_steps
		print(f"    Global effective batch size: {global_effective_bsz} (across {args.process_count} processes, with {args.gradient_accumulation_steps} accumulation steps)")
	else:
		local_effective_bsz = args.effective_bsz * args.gradient_accumulation_steps
		print(f"    Effective batch size (with accumulation): {local_effective_bsz}")
	print()

	from lob.train import train

	# Import time and socket for debug timestamps
	import time
	import socket
	host = socket.gethostname()

	# ========== Synchronization Barrier 2: Before Training Starts ==========
	if is_distributed:
		print(f"[*] Synchronization barrier 2/4: Pre-training final sync, waiting for all nodes...")
		try:
			sync_start = time.time()
			multihost_utils.sync_global_devices("pre_training_sync")
			sync_time = time.time() - sync_start
			print(f"[*] ✓ All {process_count} nodes ready to start training (took: {sync_time:.2f}s)")
		except Exception as e:
			print(f"[ERROR] Synchronization barrier 2/4 failed: {e}")
			print(f"[ERROR] Process {process_index} failed pre-training sync")
			raise

	# Wrap training in try/finally to ensure proper cleanup even on errors
	t0 = time.time()
	print(f"[{host}] [proc {process_index}/{process_count}] start train() at {time.strftime('%H:%M:%S')}", flush=True)

	try:
		train(args)
		print(f"[{host}] [proc {process_index}/{process_count}] end train(), dt={time.time()-t0:.2f}s", flush=True)
	except Exception as e:
		print(f"[{host}] [proc {process_index}] EXCEPTION in train(): {e}", flush=True)
		import traceback
		traceback.print_exc()
		raise
	finally:
		# Provide an explicit multihost rendezvous before shutdown to reduce skew
		if is_distributed:
			try:
				from jax.experimental import multihost_utils as _mh
				t1 = time.time()
				print(f"[{host}] [proc {process_index}] entering rendezvous(end-of-train)", flush=True)
				_ = _mh.sync_global_devices("end-of-train")
				print(f"[{host}] [proc {process_index}] rendezvous done, dt={time.time()-t1:.2f}s", flush=True)
			except Exception as e:
				print(f"[{host}] [proc {process_index}] rendezvous error: {e}", flush=True)

		# Ensure clean multi-host shutdown to avoid coordination barrier timeouts
		if is_distributed:
			import jax as _jax
			t2 = time.time()
			print(f"[{host}] [proc {process_index}] entering shutdown", flush=True)
			_jax.distributed.shutdown()
			print(f"[{host}] [proc {process_index}] shutdown done, dt={time.time()-t2:.2f}s", flush=True)

		# Finalize W&B only on process 0 after distributed shutdown to avoid skew
		try:
			if args.USE_WANDB and process_index == 0:
				import wandb as _wandb
				t3 = time.time()
				print(f"[{host}] [proc {process_index}] wandb.finish() ...", flush=True)
				_wandb.finish()
				print(f"[{host}] [proc {process_index}] wandb.finish() done, dt={time.time()-t3:.2f}s", flush=True)
		except Exception as e:
			print(f"[{host}] [proc {process_index}] wandb.finish() error: {e}", flush=True)
