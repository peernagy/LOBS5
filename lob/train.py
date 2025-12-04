import os
import jax
from jax import random
import jax.numpy as jnp
import flax
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_manager import MultiprocessingOptions



# WandB configuration (must be set before wandb import)
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_INSECURE_DISABLE_SSL"] = "True"
import wandb


import gc

from lob.init_train import init_train_state, load_checkpoint, save_checkpoint, deduplicate_trainstate
from lob.dataloading import create_lobster_prediction_dataset, create_lobster_train_loader#, Datasets
from lob.lobster_dataloader import LOBSTER_Dataset
from lob.train_helpers import reduce_lr_on_plateau, linear_warmup, \
    cosine_annealing, constant_lr, train_epoch, validate




def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    # Initialize wandb (only on process 0 to avoid network load and sync issues)
    wandb_run_id = None
    run = None  # Initialize run variable for checkpoint naming

    if args.USE_WANDB and args.process_index == 0:
        # Get model size and job ID
        job_id = os.environ.get("SLURM_JOB_ID", "local")

        # Enhanced run name with configuration details
        run_name = f"lobs5_d{args.d_model}_l{args.n_layers}_b{args.blocks}_bsz{args.per_gpu_bsz}x{args.num_devices}_seed{args.jax_seed}_jid{job_id}"

        # Initialize WandB
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            settings=wandb.Settings(init_timeout=300),
        )
        wandb_run_id = wandb.run.id
        print(f"[*] WandB: Created new run (ID: {wandb_run_id})")
    else:
        # Disable wandb on non-zero processes
        os.environ["WANDB_MODE"] = "disabled"
        # Create a dummy run object for checkpoint naming on non-zero processes
        # Use SLURM_JOB_ID to ensure unique checkpoint directory per job
        class DummyRun:
            def __init__(self):
                job_id = os.environ.get("SLURM_JOB_ID", "local")
                self.name = f"offline_{job_id}"
                self.id = job_id
        run = DummyRun()

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    if args.USE_WANDB and args.process_index == 0:
        wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    ds = 'lobster-prediction'
    #create_dataset_fn =  Datasets[ds]

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    mask_fn=None
    if args.masking == 'causal':
        mask_fn = LOBSTER_Dataset.causal_mask
    elif args.masking == 'random':
        mask_fn = LOBSTER_Dataset.random_mask
    elif args.masking == 'last_pos':
         mask_fn = LOBSTER_Dataset.last_pos_mask
    elif args.masking == 'none':
         mask_fn = LOBSTER_Dataset.no_mask
    else:
        ValueError('Issue with mask function: logic for '+args.masking+' not implemented.')

    # Track dataloader seed for checkpoint resume support
    current_dataloader_seed = args.jax_seed

    (lobster_dataset, trainloader, valloader, testloader, aux_dataloaders,
        n_classes, seq_len, in_dim, book_seq_len, book_dim, train_size) = \
        create_lobster_prediction_dataset(
            args.dir_name,
            seed=current_dataloader_seed,
            mask_fn=mask_fn,
            msg_seq_len=args.msg_seq_len,
            per_gpu_bsz=args.per_gpu_bsz,
            num_devices=args.num_devices,
            use_book_data=args.use_book_data,
            use_simple_book=args.use_simple_book,
            book_transform=args.book_transform,
            n_data_workers=args.n_data_workers,
            shuffle_train=args.shuffle_train,
            rand_offset=args.random_offsets_train,
            debug_overfit=args.debug_overfit,
            test_dir=args.test_dir_name if hasattr(args, 'test_dir_name') else None,
            data_mode=args.data_mode if hasattr(args, 'data_mode') else 'preproc',
            use_distributed_sampler=args.is_distributed,
            process_rank=args.process_index,
            process_count=args.process_count,
        )



    print(f"[*] Starting S5 Training on {ds} =>> Initializing...")

    # ========== Synchronization Barrier 4: After Data Loading ==========
    if hasattr(args, 'process_count') and args.process_count > 1:
        import time as time_module
        from jax.experimental import multihost_utils
        if jax.process_index() == 0:
            print(f"[*] Synchronization barrier 4/4: Data loading complete, waiting for all nodes...")
        try:
            sync_start = time_module.time()
            multihost_utils.sync_global_devices("data_loading_complete")
            sync_time = time_module.time() - sync_start
            if jax.process_index() == 0:
                print(f"[*] ✓ All {args.process_count} nodes completed data loading (took: {sync_time:.2f}s)")
                print(f"[*] 🚀 All initialization phases complete, starting distributed training!")
        except Exception as e:
            print(f"[ERROR] Synchronization barrier 4/4 failed (process {jax.process_index()}): {e}")
            raise
    if args.debug_loading:
        state=None
        val_model=None
        init_hidden=None
    else:
        state, model_cls = init_train_state(
            args,
            n_classes=n_classes,
            seq_len=seq_len,
            book_dim=book_dim,
            book_seq_len=book_seq_len,
            print_shapes=True
        )

        if args.restore is not None and args.restore != '':
            print(f"[*] Restoring weights from {args.restore}")
            ckpt = load_checkpoint(
                state,
                args.restore,
                # args.__dict__,
                step=args.restore_step,
            )
            state = ckpt['model']
        
        val_model = model_cls(training=False, step_rescale=1)
        init_hidden=model_cls().initialize_carry(batch_size=args.per_gpu_bsz,
                                                hidden_size=(ssm_size // pow(2,int(args.conj_sym))),
                                                n_message_layers=args.n_message_layers,
                                                n_book_pre_layers=args.n_book_pre_layers ,
                                                n_book_post_layers=args.n_book_post_layers,
                                                n_fused_layers=args.n_layers,
                                                h_size_ema=ssm_size)
    
    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay

    # Calculate steps per epoch (per-process)
    # - train_size is global sample count
    # - args.effective_bsz is per-process batch size (per_gpu_bsz × local num_devices)
    # - In multi-node, each process sees ~1/process_count of the data via DistributedSampler
    if getattr(args, 'is_distributed', False) and getattr(args, 'process_count', 1) > 1:
        steps_per_epoch = int(train_size / (args.effective_bsz * args.process_count)) if args.curtail_epochs is None else args.curtail_epochs+1
        print(f"[DEBUG] Steps calculation (distributed): {train_size:,} / ({args.effective_bsz} × {args.process_count}) = {steps_per_epoch:,}")
    else:
        steps_per_epoch = int(train_size / args.effective_bsz) if args.curtail_epochs is None else args.curtail_epochs+1
        print(f"[DEBUG] Steps calculation (single-node): {train_size:,} / {args.effective_bsz} = {steps_per_epoch:,}")

    # Create checkpoint manager (only on process 0)
    # IMPORTANT: Use active_processes={0} to tell Orbax only process 0 is involved
    # This prevents Orbax from waiting for other processes in its internal barriers
    if args.process_index == 0:
        print(f"[DEBUG] Process 0: Creating checkpoint manager options...")
        mgr_options = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            create=False,
            max_to_keep=10,
            keep_period=5,
            # step_prefix=f'{run.name}_{run.id}',
            # Disable async checkpointing to avoid multi-node issues
            enable_async_checkpointing=False,
            # CRITICAL: Tell Orbax only process 0 participates in checkpointing
            # This prevents Orbax's internal barriers from waiting for other processes
            multiprocessing_options=MultiprocessingOptions(primary_host=0, active_processes={0})
        )
        print(f"[DEBUG] Process 0: Options created, initializing CheckpointManager...")
        os.makedirs(f'checkpoints/{run.name}_{run.id}/', exist_ok=True)
        ckpt_mgr = ocp.CheckpointManager(
            os.path.abspath(f'checkpoints/{run.name}_{run.id}/'),
            # ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
            # ocp.Checkpointer(ocp.StandardCheckpointHandler()),
            item_names=('state', 'metadata'),
            options=mgr_options,
            metadata=vars(args)
        )
        print(f"[*] Checkpoint manager created: checkpoints/{run.name}_{run.id}/")
    else:
        ckpt_mgr = None
        print(f"[*] Process {args.process_index}: Skipping checkpoint manager (only process 0 saves)")


    # Initialize wandb table (only on process 0)
    if args.process_index == 0:
        if args.ignore_times:
            # Removing the 5 abs time tokens from the length of the sequence.
            dt = [[x] for (x,) in zip([*range(seq_len-5*args.msg_seq_len)])]
        else:
            dt = [[x] for (x,) in zip([*range(seq_len)])]
        ce_table=wandb.Table(columns=["tok"] ,data=dt)
    else:
        ce_table = None

    ignore_times=args.ignore_times
    batchnorm=args.batchnorm

    # ========== Synchronization Barrier 5: Before Training Loop ==========
    # CRITICAL: Ensure all processes are synchronized before entering pmap calls
    # Without this, process 1 races ahead while process 0 creates checkpoint manager
    if hasattr(args, 'process_count') and args.process_count > 1:
        import time as time_module
        from jax.experimental import multihost_utils
        print(f"[*] Process {args.process_index}: Barrier 5/5 - Before training loop...")
        try:
            sync_start = time_module.time()
            multihost_utils.sync_global_devices("before_training_loop")
            sync_time = time_module.time() - sync_start
            if jax.process_index() == 0:
                print(f"[*] ✓ All nodes ready to start training (barrier took: {sync_time:.2f}s)")
        except Exception as e:
            print(f"[ERROR] Barrier 5/5 failed (process {jax.process_index()}): {e}")
            raise

    # ===== Check for Resume from Intra-Epoch Checkpoint =====
    start_epoch = 0
    start_segment = 0
    resume_from_checkpoint = False
    resume_dataloader_seed = None  # Will be set if resuming with random_offsets

    if ckpt_mgr is not None and ckpt_mgr.latest_step() is not None:
        latest_step = ckpt_mgr.latest_step()
        print(f"[*] Found checkpoint at step {latest_step}, attempting to load...")

        try:
            # Load checkpoint metadata to get resume info
            restored = ckpt_mgr.restore(latest_step)
            if restored is not None and 'resume_info' in restored:
                resume_info = restored['resume_info']
                start_epoch = resume_info['epoch']
                start_segment = resume_info['segment_idx'] + 1  # Start from next segment
                restored_step = resume_info['global_step']

                # If we finished all segments in this epoch, move to next epoch
                need_new_epoch_seed = False
                if start_segment >= resume_info['num_evals_per_epoch']:
                    start_epoch += 1
                    start_segment = 0
                    need_new_epoch_seed = True  # Don't use saved seed - new epoch needs new seed

                # Get dataloader_seed if saved (for random_offsets support)
                # But only use it if we're resuming mid-epoch (not starting a new epoch)
                if 'dataloader_seed' in resume_info and not need_new_epoch_seed:
                    resume_dataloader_seed = resume_info['dataloader_seed']
                    print(f"[*] Resuming from: epoch {start_epoch}, segment {start_segment}, global_step {restored_step}, dataloader_seed {resume_dataloader_seed}")
                else:
                    if need_new_epoch_seed:
                        print(f"[*] Resuming from: epoch {start_epoch}, segment {start_segment}, global_step {restored_step} (new epoch, will use fresh seed)")
                    else:
                        print(f"[*] Resuming from: epoch {start_epoch}, segment {start_segment}, global_step {restored_step}")

                # Restore model state
                state = restored['model']
                step = restored_step
                resume_from_checkpoint = True

                # Handle dataloader recreation for random_offsets_train
                if args.random_offsets_train:
                    if resume_dataloader_seed is not None:
                        # Resuming mid-epoch: use saved seed to ensure correct data order for skip-batches
                        print(f"[*] Recreating trainloader with saved seed {resume_dataloader_seed}...")
                        current_dataloader_seed = resume_dataloader_seed
                    else:
                        # Starting new epoch: generate fresh random seed
                        # (need_new_epoch_seed was True, so we didn't load the old seed)
                        current_dataloader_seed = int(random.randint(train_rng, (1,), 0, 100000)[0])
                        print(f"[*] New epoch: generating fresh dataloader seed {current_dataloader_seed}...")

                    del trainloader
                    trainloader = create_lobster_train_loader(
                        lobster_dataset,
                        current_dataloader_seed,
                        args.effective_bsz,
                        num_workers=args.n_data_workers,
                        reset_train_offsets=args.random_offsets_train,
                        shuffle=args.shuffle_train,
                        use_distributed_sampler=args.is_distributed,
                        process_rank=args.process_index,
                        process_count=args.process_count)
                    print(f"[*] Trainloader recreated with seed {current_dataloader_seed}")

                print(f"[*] Model state restored successfully")
            else:
                print(f"[*] Checkpoint found but no resume_info, starting from scratch")
        except Exception as e:
            print(f"[WARNING] Failed to restore checkpoint: {e}")
            print(f"[*] Starting training from scratch")

    for epoch in range(start_epoch, args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")
        # jax.profiler.start_trace("./jax-traces")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        print('Training on', args.num_devices, 'devices.')

        # ===== Intra-Epoch Evaluation Setup =====
        num_evals_per_epoch = 5
        eval_interval = steps_per_epoch // num_evals_per_epoch
        print(f"[*] Intra-epoch evaluation: {num_evals_per_epoch} evals, interval={eval_interval} steps")

        # Create iterator for trainloader (important for resuming training across segments)
        trainloader_iter = iter(trainloader)

        # ===== Skip Already Trained Batches (for Resume) =====
        # Determine which segment to start from
        current_start_segment = start_segment if epoch == start_epoch else 0

        if current_start_segment > 0:
            batches_to_skip = current_start_segment * eval_interval
            print(f"[*] Resuming: skipping {batches_to_skip} batches ({current_start_segment} segments)...")

            from tqdm import tqdm
            skipped = 0
            for _ in tqdm(range(batches_to_skip), desc="Fast-forwarding", disable=(args.process_index != 0)):
                try:
                    _ = next(trainloader_iter)
                    skipped += 1
                except StopIteration:
                    print(f"[WARNING] DataLoader exhausted after skipping {skipped} batches")
                    break

            print(f"[*] Skipped {skipped} batches, resuming from segment {current_start_segment}")

        # ===== Intra-Epoch Loop: Train segments + Validate =====
        for eval_idx in range(current_start_segment, num_evals_per_epoch):
            print(f"\n[*] Epoch {epoch + 1} - Segment {eval_idx + 1}/{num_evals_per_epoch}")

            train_rng, skey = random.split(train_rng)

            # Determine how many batches to train in this segment
            is_last_segment = (eval_idx == num_evals_per_epoch - 1)
            max_batches = None if is_last_segment else eval_interval

            # Train one segment
            state, train_loss, ce_by_tok, step = train_epoch(
                state,
                skey,
                trainloader_iter,  # Use iterator instead of dataloader
                seq_len,
                batchnorm,
                lr_params,
                args.num_devices,
                args.debug_loading,
                args.enable_profiler,
                None,  # Disable curtail_epochs in intra-epoch mode (use max_batches instead)
                init_hidden,
                epoch,
                ignore_times,
                args.log_ce_tables,
                use_wandb=args.USE_WANDB,
                process_index=args.process_index,
                max_batches=max_batches,  # Limit batches for this segment
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

            # Update lr_params with new step count
            lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

            # Skip validation for last segment (will run full validation at end of epoch)
            if not is_last_segment:
                # Mid-epoch validation (complete validation set)
                print(f"[*] Mid-Epoch Validation at step {step}")
                (intra_val_loss, intra_val_acc, _, _) = validate(
                    state,
                    val_model.apply,
                    valloader,
                    seq_len,
                    in_dim,
                    batchnorm,
                    args.num_devices,
                    epoch,
                    curtail_epoch=None,
                    apply_method='__call_ar__',
                    ignore_times=ignore_times,
                    log_ce_tables=False  # Don't log tables for mid-epoch evals
                )

                print(f"    Train Loss: {train_loss:.5f}, Val Loss: {intra_val_loss:.5f}, Val Acc: {intra_val_acc:.4f}")

                # Log to WandB (only process 0)
                if args.USE_WANDB and args.process_index == 0:
                    wandb.log({
                        "intra_epoch/val_loss": intra_val_loss,
                        "intra_epoch/val_acc": intra_val_acc,
                        "intra_epoch/train_loss": train_loss,
                        "intra_epoch/eval_idx": eval_idx,
                        "intra_epoch/epoch": epoch,
                    }, step=step)
            else:
                print(f"    Train Loss: {train_loss:.5f} (skipping validation, will run at end of epoch)")

            # ===== Intra-Epoch Checkpoint Save =====
            # Save checkpoint after each segment (only process 0)
            # Multi-node: use barrier BEFORE checkpoint to sync, then only process 0 saves
            is_multi_node = hasattr(args, 'process_count') and args.process_count > 1

            if is_multi_node:
                from jax.experimental import multihost_utils as mhu_ckpt
                print(f"[*] Process {args.process_index}: entering checkpoint_save barrier")
                mhu_ckpt.sync_global_devices("checkpoint_save")
                print(f"[*] Process {args.process_index}: passed checkpoint_save barrier")

            if args.process_index == 0 and ckpt_mgr is not None:
                # Deduplicate state: take first replica and place on CPU for serialization
                # This works for both single-node and multi-node (each process only sees local devices)
                dedup_state = jax.tree.map(lambda x: jax.device_get(x[0]), state)
                intra_ckpt = {
                    'model': dedup_state,
                    'config': vars(args),
                    'resume_info': {
                        'epoch': epoch,
                        'segment_idx': eval_idx,
                        'global_step': step,
                        'num_evals_per_epoch': num_evals_per_epoch,
                        'eval_interval': eval_interval,
                        'dataloader_seed': current_dataloader_seed,  # For random_offsets resume
                    },
                    'metrics': {
                        'train_loss': float(train_loss),
                    }
                }
                # Use a unique step that encodes both epoch and segment
                ckpt_step = epoch * num_evals_per_epoch + eval_idx
                save_checkpoint(ckpt_mgr, intra_ckpt, ckpt_step)
                print(f"[*] Intra-epoch checkpoint saved: epoch {epoch}, segment {eval_idx}, step {step}, dataloader_seed {current_dataloader_seed}")

            # Second barrier after checkpoint save to ensure all processes sync before next segment
            if is_multi_node:
                print(f"[*] Process {args.process_index}: entering post_checkpoint barrier")
                mhu_ckpt.sync_global_devices("post_checkpoint")
                print(f"[*] Process {args.process_index}: passed post_checkpoint barrier")

        # End of intra-epoch loop
        print(f"\n[*] Epoch {epoch + 1} Training Complete - All {num_evals_per_epoch} segments done")

        if args.random_offsets_train:
            # reinit training loader, so that sequences are initialised with different offsets
            # Generate new random seed for next epoch and update current_dataloader_seed for checkpoint
            current_dataloader_seed = int(random.randint(skey, (1,), 0, 100000)[0])
            print(f"[*] Next epoch dataloader_seed: {current_dataloader_seed}")
            del trainloader
            trainloader = create_lobster_train_loader(
                lobster_dataset,
                current_dataloader_seed,
                args.effective_bsz,
                num_workers=args.n_data_workers,
                reset_train_offsets=args.random_offsets_train,
                shuffle=args.shuffle_train,
                use_distributed_sampler=args.is_distributed,
                process_rank=args.process_index,
                process_count=args.process_count)
        # ===== End-of-Epoch: Full Validation and Test Evaluation =====
        print(f"\n{'='*80}")
        print(f"[*] Epoch {epoch + 1} - Final Evaluation (complete val + test sets)")
        print(f"{'='*80}")

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Final Validation")
            (val_loss,
              val_acc,
                val_ce_means,
                val_acc_means) = validate(state,
                                        #model_cls,
                                        val_model.apply,
                                        valloader,
                                        seq_len,
                                        in_dim,
                                        batchnorm,
                                        args.num_devices,
                                        epoch,
                                        curtail_epoch=None,
                                        apply_method='__call_ar__',
                                        ignore_times=ignore_times,
                                        log_ce_tables=args.log_ce_tables)

            print(f"[*] Running Epoch {epoch + 1} Test ") #on train set (With Call RNN)...
            (test_loss, test_acc,
              test_ce_means,test_acc_means) = validate(state,
                                           #model_cls,
                                           val_model.apply,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           batchnorm,
                                           args.num_devices,
                                           epoch,
                                           curtail_epoch=None,
                                           apply_method='__call_ar__',
                                           ignore_times=ignore_times,
                                           log_ce_tables=args.log_ce_tables)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss (AR): {val_loss:.5f} --Test Loss (RNN): {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            # print("Testing on train data (diff offset) for debugging purposes")
            (test_loss, test_acc,
              test_ce_means,test_acc_means) = validate(state,
                                         #model_cls,
                                         val_model.apply,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         batchnorm,
                                         args.num_devices,
                                         epoch,
                                         curtail_epoch=None,
                                         ignore_times=ignore_times,
                                         log_ce_tables=args.log_ce_tables)
            val_loss=test_loss
            val_acc=test_acc

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # Save checkpoint (only on process 0)
        if args.process_index == 0:
            ckpt = {
                'model': deduplicate_trainstate(state),
                'config': vars(args),
                'metrics': {
                    'loss_train': float(train_loss),
                    'loss_val_ar': float(val_loss),
                    'loss_test_rnn': float(test_loss),
                    'acc_val_ar': float(val_acc),
                    'acc_test_rnn': float(test_acc),
                }
            }
            save_checkpoint(ckpt_mgr, ckpt, epoch)
            print(f"[*] Checkpoint saved: epoch {epoch}")
        else:
            if args.process_index <= 2 or args.process_index == args.process_count - 1:
                # Only log for first few and last process to avoid spam
                print(f"[Checkpoint] Process {args.process_index}: Skipped (only process 0 saves)")

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1



        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if args.log_ce_tables and args.process_index == 0:
            ce_table.add_column(name="val_ce_"+str(epoch),data=val_ce_means.tolist())
            ce_table.add_column(name="test_ce_"+str(epoch),data=test_ce_means.tolist())
            ce_table.add_column(name="val_acc_"+str(epoch),data=val_acc_means.tolist())
            ce_table.add_column(name="test_acc_"+str(epoch),data=test_acc_means.tolist())
            ce_table.add_column(name="train_ce_"+str(epoch),data=ce_by_tok.tolist())
            ce_table=wandb.Table(columns=ce_table.columns,data=ce_table.data)
        

        # Log metrics to wandb (only on process 0)
        if args.USE_WANDB and args.process_index == 0:
            if valloader is not None:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": float(state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'][0]),
                        "ssm_lr": float(state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'][0]),
                        # "Training CE by token":ce_table
                    },
                    step=step  # Use global step for consistency
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": float(state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'][0]),
                        "ssm_lr": float(state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'][0]),
                        # "Training CE by token":ce_table
                    },
                    step=step  # Use global step for consistency
                )

            if args.log_ce_tables:
                wandb.log({"CE by token": ce_table}, step=step)

        # Update best metrics in wandb (only on process 0)
        if args.USE_WANDB and args.process_index == 0:
            wandb.run.summary["Best Val Loss"] = best_loss
            wandb.run.summary["Best Val Accuracy"] = best_acc
            wandb.run.summary["Best Epoch"] = best_epoch
            wandb.run.summary["Best Test Loss"] = best_test_loss
            wandb.run.summary["Best Test Accuracy"] = best_test_acc
        # print("IGNORING EARLY STOPPING FOR TINY EPOCH SIZE ")
        # After each epoch
        gc.collect()
        # jax.clear_backends()
        jax.clear_caches()
        # jax.profiler.stop_trace()
        if count > args.early_stop_patience:
            break
