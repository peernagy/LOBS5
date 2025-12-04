from functools import partial
import numpy as onp
import jax
import jax.numpy as np
# from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
from flax import jax_utils
import optax
from typing import Any, Dict, Optional, Tuple, Union
from lob.encoding import Message_Tokenizer
import sys
import json
import os

# from lob.lob_seq_model import LobPredModel


TIME_START_I=9
TIME_END_I =13

# num_devices_global = 2
# global_devices = jax.local_devices()[0: num_devices_global]


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = \
        jax_utils.replicate(np.array(lr_val, dtype=np.float32))
        
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = \
        jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = \
            jax_utils.replicate(np.array(ssm_lr_val, dtype=np.float32))

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       use_book_data,
                       book_dim,
                       book_seq_len,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False,
                       num_devices=1,
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    # batch size is given for data across all devices
    # i.e. batch is split between GPUs but dummy data is per GPU
    assert bsz % num_devices == 0
    bsz = bsz // num_devices

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        if use_book_data:
            dummy_input = (
                # np.ones((bsz, seq_len, in_dim), dtype=np.int32),  # messages
                np.ones((bsz, seq_len, ), dtype=np.int32),  # messages
                np.ones((bsz, seq_len, book_dim)),  # books
            )
            integration_timesteps = (
                np.ones((bsz, seq_len, )),
                np.ones((bsz, seq_len, )),
            )
        else:
            # dummy_input = (np.ones((bsz, seq_len, in_dim), dtype=np.int32) , )
            dummy_input = (np.ones((bsz, seq_len, ), dtype=np.int32) , )
            integration_timesteps = (np.ones((bsz, seq_len, )), )

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)

    # jax.debug.print("Dummy input shapes (msg,book) ({}, \n {})",dummy_input[0].shape,dummy_input[1].shape)
    print(f"[DEBUG] Dummy input shapes (msg,book): {dummy_input[0].shape}, {dummy_input[1].shape}")
    #RNN mode and initialisation needs to go in here if we need it. 

    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           *dummy_input, *integration_timesteps,
                           method='__call_ar__' 
                           )
    
    if batchnorm:
        params = variables["params"]#.unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"]#.unfreeze()
        # Note: `unfreeze()` is for using Optax.

    print(params['message_encoder']['encoder']['embedding'].shape)

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "norm"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in ["B", "C", "C1", "C2", "D",
                         "Lambda_re", "Lambda_im", "log_step", "norm"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    #print(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # BF16 Mixed Precision: 正确实现
    # Master weights (params) 保持 FP32，用于 optimizer 更新
    # BF16 计算通过模型层的 dtype 参数控制 (在 forward pass 中)
    # 这样 optimizer states (m, v) 也保持 FP32，避免 NaN
    use_bf16 = os.environ.get('USE_BF16', '1') == '1'
    if use_bf16:
        print(f"[*] BF16 Mixed Precision enabled: params=FP32 (master), compute=BF16 (via model dtype)")
    else:
        print(f"[*] Full FP32 training (BF16 disabled via USE_BF16=0)")

    # keep copy of state on each device
    print(state.params['message_encoder']['encoder']['embedding'].shape)
    state = jax_utils.replicate(state)#, devices=global_devices)
    print(state.params['message_encoder']['encoder']['embedding'].shape)

    return state

def get_slices(dims):
    slices = []
    last_i = 0
    for d in dims:
        slices.append(slice(last_i, last_i+d))
        last_i += d
    return slices

# Train and eval steps
# @partial(np.vectorize, signature="(c),()->()")
# def cross_entropy_loss(logits, label):
#     one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[-1])
#     return -np.sum(one_hot_label * logits)

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    return -np.sum(logits[label])


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss_test(logits, label):
    return -np.sum(logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label

def prep_batch(
        batch: Union[
            Tuple[onp.ndarray, onp.ndarray, Dict[str, onp.ndarray]],
            Tuple[onp.ndarray, onp.ndarray]],
        seq_len: int,
        # in_dim: int,
        num_devices: int,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:

    if len(batch) == 2:
        inputs, targets = batch
        book_data, timestep_msg, timestep_book = None, None, None
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
        book_data = aux_data.get("book_data", None)
        timestep_msg = aux_data.get("timesteps_msg", None)
        timestep_book = aux_data.get("timesteps_book", None)            
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # reshape from large batch to multiple device batches
    inputs, targets, book_data, timestep_msg, timestep_book = device_reshape(
        num_devices,
        inputs,
        targets,
        book_data,
        timestep_msg,
        timestep_book,
    )
    # print('inputs shape (device_reshape):', inputs.shape)

    # split large batch into smaller device batches on the GPUs
    inputs, labels, integration_times = _prep_batch_par(
        inputs,
        targets,
        seq_len,
        # in_dim,
        book_data,
        timestep_msg,
        timestep_book,
    )
    # print('inputs (targets) shape (_prep_batch_par):', inputs[1].shape)

    return inputs, labels, integration_times

@partial(
#    jax.vmap,
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(2,),
    # in_axes=(0, 0, None, None, 0, 0, 0),
    in_axes=(0, 0, None, 0, 0, 0),
    # out_axes=(0, 0, 0),
    # devices=global_devices
)
def _prep_batch_par(
        inputs: jax.Array,
        targets: jax.Array,
        seq_len: int,
        # in_dim: int,
        book_data: Optional[jax.Array] = None,
        timestep_msg: Optional[jax.Array] = None,
        timestep_book: Optional[jax.Array] = None,
    ) -> Tuple[Tuple, np.ndarray, Tuple]:
    """
    Take a batch and convert it to a standard x/y format per device
    TODO: document this better for pmapped version
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """

    assert inputs.shape[1] == seq_len, f'inputs: {inputs.shape} seq_len {seq_len}'
    # inputs = one_hot(inputs, in_dim)

    # If there is an aux channel containing the integration times, then add that.
    if timestep_msg is not None:
        #timestep_msg = jax.device_put(timestep_msg, jax.devices()[0])
        integration_timesteps = (np.diff(np.asarray(timestep_msg)), )
    else:
        integration_timesteps = (np.ones((len(inputs), seq_len)), )

    if book_data is not None:
        #book_data = jax.device_put(book_data, jax.devices()[0])
        full_inputs = (inputs.astype(np.int32), book_data)
        if timestep_book is not None:
            #timestep_book = jax.device_put(timestep_book, jax.devices()[0])
            integration_timesteps += (np.diff(timestep_book), )
        else:
            integration_timesteps += (np.ones((len(inputs), seq_len)), )
    else:
        full_inputs = (inputs.astype(np.int32), )

    # CAVE: squeeze very important for training!
    return full_inputs, np.squeeze(targets.astype(np.int32)), integration_timesteps

@partial(jax.jit, static_argnums=(0,), backend='gpu')# backend='cpu')
def device_reshape(
        num_devices: int,
        inputs: jax.Array,
        targets: jax.Array,
        book_data: Optional[jax.Array] = None,
        timestep_msg: Optional[jax.Array] = None,
        timestep_book: Optional[jax.Array] = None,
    ) -> Tuple:
    """ 
    """
    inputs = np.reshape(inputs, (num_devices, -1, *inputs.shape[1:]))
    targets = np.reshape(targets, (num_devices, -1, *targets.shape[1:]))
    if book_data is not None:
        book_data = np.reshape(book_data, (num_devices, -1, *book_data.shape[1:]))
    if timestep_msg is not None:
        timestep_msg = np.reshape(timestep_msg, (num_devices, -1, *timestep_msg.shape[1:]))
    if timestep_book is not None:
        timestep_book = np.reshape(timestep_book, (num_devices, -1, *timestep_book.shape[1:]))
    return inputs, targets, book_data, timestep_msg, timestep_book


def train_epoch(
        state,
        rng,
        #model,
        trainloader,
        seq_len,
        # in_dim,
        batchnorm,
        lr_params,
        num_devices,
        debug_loading,
        debug_profiler,
        curtail_epochs,
        init_hiddens,
        epoch,
        ignore_times,
        log_ce_tables,
        use_wandb=False,
        process_index=0,
        max_batches=None,  # New: limit number of batches to train (for intra-epoch evaluation)
        gradient_accumulation_steps=1,  # Number of steps to accumulate gradients
    ):

    """
    Training function for an epoch that loops over batches.

    Args:
        max_batches: If provided, stop training after processing this many batches.
                     Used for intra-epoch evaluation to train only a segment of the epoch.
        gradient_accumulation_steps: Number of micro-batches to accumulate gradients over
                                     before performing an optimizer step. Default is 1 (no accumulation).
    """
    # Store Metrics
    batch_losses = []
    cross_entropies= [] #list of 1xNTok losses

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params
    batches_processed = 0  # Track how many batches we've processed in this call

    # Gradient accumulation state
    accumulated_grads = None
    accumulated_loss = 0.0
    accumulated_ce = None
    accumulated_batch_stats = None
    accum_count = 0

    #with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        # print(f"train_epoch: Epoch {epoch} - Batch {batch_idx} / {len(trainloader)}")
        # print(f"train_epoch: Batch input shape: {batch[0].shape}, batch target shape: {batch[1].shape}")
        if not debug_loading:
            if (step>1) & (step<3) & debug_profiler:
                jax.profiler.start_trace("/tmp/tensorboard")
            inputs, labels, integration_times = prep_batch(batch, seq_len, num_devices)
            # print("train_epoch: Prepared batch inputs shape:", inputs[0].shape)
            # print("train_epoch: Prepared batch labels shape:", labels.shape)
            # print("train_epoch: Inputs 0:5:", inputs[0][0,0:5,:])
            rng, drop_rng = jax.random.split(rng)


            # state,loss=train_step_rnn(
            #     state,
            #     drop_rng,
            #     inputs,
            #     labels,
            #     integration_times,
            #     batchnorm,
            #     init_hiddens)

            if gradient_accumulation_steps == 1:
                # Original path: no accumulation, directly update
                # print("Gets to train")
                state, loss, ce, logits = train_step(
                    state,
                    drop_rng,
                    inputs,
                    labels,
                    integration_times,
                    batchnorm,
                    ignore_times,
                )
                if debug_profiler:
                    loss.block_until_ready()
                # print("completes train step")
                # if (batch_idx==0) & (epoch%100==0):
                #     np.set_printoptions(threshold=sys.maxsize)
                #     with open(f'/data1/sascha/data/losses/losses_batch_{batch_idx}_training.txt', 'w') as f:
                #         print( ce, file=f)
                #     print("Printing logits of shape ", logits.shape, " to file")
                #     with open(f'/data1/sascha/data/losses/logits_batch_{batch_idx}_training.txt', 'w') as f:
                #         print( logits[0,0,0:44,:], file=f)
                #     np.set_printoptions()
                #     print('Done Printing')

                # losses are already averaged across devices (--> should be all the same here)
                batch_losses.append(loss[0])
                if log_ce_tables:
                    cross_entropies.append(ce)

                # DISABLED: Per-step wandb logging causes ~10% slowdown even at 1000-step intervals
                # Only using per-epoch logging in train.py instead
                # if use_wandb and process_index == 0 and step % 1000 == 0:
                #     import wandb
                #     current_lr = decay_function(step, lr, end_step, lr_min)
                #     wandb.log({
                #         "train/loss_step": float(loss[0]),
                #         "train/step": step,
                #         "train/epoch": epoch,
                #         "train/lr": float(current_lr),
                #     })

                lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
                state, step = update_learning_rate_per_step(lr_params, state)
            else:
                # Gradient accumulation path
                grads, loss, ce, mod_vars = compute_gradients_step(
                    state,
                    drop_rng,
                    inputs,
                    labels,
                    integration_times,
                    batchnorm,
                    ignore_times,
                )

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                    accumulated_ce = ce if log_ce_tables else None
                else:
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda a, g: a + g, accumulated_grads, grads
                    )
                    if log_ce_tables and accumulated_ce is not None:
                        accumulated_ce = accumulated_ce + ce

                accumulated_loss += float(loss[0])
                accum_count += 1

                # Track batch_stats for batchnorm
                if batchnorm and mod_vars is not None:
                    if accumulated_batch_stats is None:
                        accumulated_batch_stats = mod_vars["batch_stats"]
                    else:
                        accumulated_batch_stats = jax.tree_util.tree_map(
                            lambda a, b: a + b, accumulated_batch_stats, mod_vars["batch_stats"]
                        )

                # Apply accumulated gradients when we've accumulated enough
                if accum_count >= gradient_accumulation_steps:
                    # Average the accumulated gradients
                    averaged_grads = jax.tree_util.tree_map(
                        lambda g: g / gradient_accumulation_steps, accumulated_grads
                    )

                    # Apply gradients
                    if batchnorm and accumulated_batch_stats is not None:
                        avg_batch_stats = jax.tree_util.tree_map(
                            lambda s: s / gradient_accumulation_steps, accumulated_batch_stats
                        )
                        state = apply_grads_with_batchnorm(state, averaged_grads, avg_batch_stats)
                    else:
                        state = apply_grads_no_batchnorm(state, averaged_grads)

                    # Record loss
                    avg_loss = accumulated_loss / gradient_accumulation_steps
                    batch_losses.append(avg_loss)
                    if log_ce_tables and accumulated_ce is not None:
                        cross_entropies.append(accumulated_ce / gradient_accumulation_steps)

                    # Update learning rate (once per optimizer step)
                    lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
                    state, step = update_learning_rate_per_step(lr_params, state)

                    # Reset accumulation state
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accumulated_ce = None
                    accumulated_batch_stats = None
                    accum_count = 0

            # Increment batch counter
            batches_processed += 1

            if (step>20) & (step<=21) & debug_profiler:
                jax.profiler.stop_trace()
                break

            # Check max_batches limit (for intra-epoch evaluation)
            if (max_batches is not None) and (batches_processed >= max_batches):
                print(f"[train_epoch] Reached max_batches={max_batches}, stopping segment")
                break

            # Original curtail_epochs check
            if (curtail_epochs is not None) and (batch_idx>=curtail_epochs):
                print("Ending epoch early due to curtail_epochs being ",curtail_epochs)
                break
        else:
            continue



    # Return average loss over batches
    if log_ce_tables:
        ce_means=np.mean(np.concatenate(cross_entropies,axis=0),axis=0)
    else:
        ce_means=None
    # jax.debug.print("CE of epoch by token: {}",ce_means.shape)
    loss_mean=np.mean(np.array(batch_losses))
    return state,loss_mean , ce_means,step


@partial(jax.vmap,in_axes=(0,0,None),out_axes=(0,0))
@partial(jax.jit,static_argnums=(2,))
def repeat_book(msg,book,shift_start):
    #DEFINITION OF START BOOK:
    # print("checking for compile in repeat_book")
    if msg.shape[0]>book.shape[0]:
        book = np.repeat(book, (msg.shape[0]) // book.shape[0], axis=0)
    # if shift_start:
    #     pad=book[:1]
    #     #FIXME: Wrong logic, needs to be the init book state.
    #     # book=np.concatenate([book[:1],book[1:]])
    #     book=np.concatenate([pad,book[:-1]])
    return (msg,book)

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,6),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 1
        batch_inputs: Tuple[jax.Array, jax.Array], # 2
        batch_labels: jax.Array, # 3
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 4
        batchnorm: bool, # 5
        ignore_times:bool, #6
    ):

    # Print hash values of static arguments
    # print(f"batchnorm hash: {batchnorm.__hash__()}")
    # print(f"ignore_times hash: {ignore_times.__hash__()}")
    # print('checking for compile in train_step')

    batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)

    def loss_fn(params):
        # print('checking for compile in loss_fn')
        # BF16 Mixed Precision: params are already BF16 from initialization
        # No need for tree_map here - direct use saves 30-40% overhead

        # ===== NaN检测点1: 输入参数 =====
        params_has_nan = jax.tree_util.tree_reduce(
            lambda a, b: a | b,
            jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), params),
            False
        )
        jax.debug.print("[NaN Check 1] Params has NaN: {}", params_has_nan)

        if batchnorm:
            logits, mod_vars = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
                method='__call_ar__'
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
                method='__call_ar__'
            )

        # ===== NaN检测点2: Forward输出 =====
        # logits_has_nan = np.any(np.isnan(logits))
        # jax.debug.print("[NaN Check 2] Logits has NaN: {}, dtype: {}", logits_has_nan, logits.dtype)

        # BF16 Mixed Precision: Cast logits back to FP32 for loss computation
        logits = logits.astype(np.float32)

        # jax.debug.print("Shape of Logits: {}",logits.shape)
        # jax.debug.print("Shape of Labels: {}", batch_labels.shape)

        # Ensure labels are int32
        batch_labels_int = batch_labels.astype(np.int32)
        ce=cross_entropy_loss(logits, batch_labels_int)
        if ignore_times:
            ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
            ce_1=ce[:,:,:TIME_START_I]
            ce_2=ce[:,:,(TIME_END_I+1):]
            ce=np.concatenate([ce_1,ce_2],axis=2)
            ce=ce.reshape(ce.shape[0],-1)

        ce=np.mean(ce,axis=0)
        # jax.debug.print("Shape of CE: {}", ce.shape)
        # average cross-ent loss
        loss = np.mean(ce)

        # ===== NaN检测点3: Loss =====
        loss_has_nan = np.isnan(loss)
        jax.debug.print("[NaN Check 3] Loss has NaN: {}, value: {:.6f}", loss_has_nan, loss)

        return loss, (mod_vars, logits,ce)

    (loss, (mod_vars, logits,ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # ===== NaN检测点4: 梯度 =====
    grads_has_nan = jax.tree_util.tree_reduce(
        lambda a, b: a | b,
        jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), grads),
        False
    )
    jax.debug.print("[NaN Check 4] Grads has NaN: {}", grads_has_nan)

    # ===== NaN检测点5: 梯度范数 =====
    grad_norm = np.sqrt(jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(lambda x: np.sum(x.astype(np.float32) ** 2), grads),
        0.0
    ))
    jax.debug.print("[NaN Check 5] Grad norm: {:.6f}", grad_norm)

    # ===== 分层梯度统计 (写入JSON文件) =====
    # 计算每个叶子节点的梯度范数
    # def compute_leaf_norm(grad):
    #     return np.sqrt(np.sum(grad.astype(np.float32) ** 2))

    # leaf_norms = jax.tree_util.tree_map(compute_leaf_norm, grads)
    # leaf_norms_with_path = jax.tree_util.tree_leaves_with_path(leaf_norms)

    # # 构建记录并写入文件
    # def write_grad_stats(step_val, global_norm_val, leaf_norms_with_path_val):
    #     """在host端写入梯度统计到JSON文件"""
    #     # 从环境变量获取精度模式
    #     precision = os.environ.get('GRAD_STATS_PRECISION', 'bf16')

    #     # 转换为dict: path -> norm
    #     layer_norms_dict = {}
    #     for path, norm in leaf_norms_with_path_val:
    #         path_str = '/'.join(str(k.key) for k in path)
    #         layer_norms_dict[path_str] = float(norm)

    #     record = {
    #         'step': int(step_val),
    #         'precision': precision,
    #         'global_norm': float(global_norm_val),
    #         'layer_norms': layer_norms_dict,
    #     }

    #     filepath = f"grad_stats_{precision}.jsonl"
    #     with open(filepath, 'a') as f:
    #         f.write(json.dumps(record) + '\n')

    # 使用callback在host端执行文件写入
    # jax.debug.callback(write_grad_stats, state.step, grad_norm, leaf_norms_with_path)

    # # ===== 梯度裁剪 (Gradient Clipping) =====
    # # 使用全局范数裁剪
    # # 注: GPT初始化后梯度范数约4M，使用10000保持合理有效学习率
    # MAX_GRAD_NORM = 10000.0
    # clip_factor = np.minimum(1.0, MAX_GRAD_NORM / (grad_norm + 1e-6))
    # grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    # jax.debug.print("[Grad Clip] clip_factor: {:.6f}, clipped_norm: {:.6f}",
    #                 clip_factor, grad_norm * clip_factor)

    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")
    ce=jax.lax.pmean(ce,axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    # ===== NaN检测点6: 更新后参数 =====
    new_params_has_nan = jax.tree_util.tree_reduce(
        lambda a, b: a | b,
        jax.tree_util.tree_map(lambda x: np.any(np.isnan(x)), state.params),
        False
    )
    jax.debug.print("[NaN Check 6] Updated params has NaN: {}", new_params_has_nan)

    #return loss, mod_vars, grads, state
    return state, loss, ce, logits


@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,6),
    in_axes=(0, None, 0, 0, 0, None, None),
)
def compute_gradients_step(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,
        batch_inputs: Tuple[jax.Array, jax.Array],
        batch_labels: jax.Array,
        batch_integration_timesteps: Tuple[jax.Array, jax.Array],
        batchnorm: bool,
        ignore_times: bool,
    ):
    """Compute gradients without updating state. Used for gradient accumulation."""

    batch_inputs = repeat_book(*batch_inputs, True)

    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
                method='__call_ar__'
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
                method='__call_ar__'
            )

        logits = logits.astype(np.float32)
        batch_labels_int = batch_labels.astype(np.int32)
        ce = cross_entropy_loss(logits, batch_labels_int)

        if ignore_times:
            ce = ce.reshape(ce.shape[0], -1, Message_Tokenizer.MSG_LEN)
            ce_1 = ce[:, :, :TIME_START_I]
            ce_2 = ce[:, :, (TIME_END_I+1):]
            ce = np.concatenate([ce_1, ce_2], axis=2)
            ce = ce.reshape(ce.shape[0], -1)

        ce = np.mean(ce, axis=0)
        loss = np.mean(ce)
        return loss, (mod_vars, ce)

    (loss, (mod_vars, ce)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Average gradients across devices
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")
    ce = jax.lax.pmean(ce, axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        return grads, loss, ce, mod_vars
    else:
        return grads, loss, ce, None


@partial(jax.pmap, axis_name="batch_devices", in_axes=(0, 0))
def apply_grads_no_batchnorm(state, accumulated_grads):
    """Apply accumulated gradients without batchnorm."""
    return state.apply_gradients(grads=accumulated_grads)


@partial(jax.pmap, axis_name="batch_devices", in_axes=(0, 0, 0))
def apply_grads_with_batchnorm(state, accumulated_grads, batch_stats):
    """Apply accumulated gradients with batchnorm."""
    return state.apply_gradients(grads=accumulated_grads, batch_stats=batch_stats)


@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step_rnn(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
        init_hiddens: Tuple, 
    ):
    #print('tracing par_loss_and_grad')

    #Never reset the hidden states:
    
    batch_inputs=repeat_book(*batch_inputs,True)
    # batch_integration_timesteps=repeat_book(*batch_integration_timesteps)
    
    
    def loss_fn(params):
        def single_elem_loss(carry,xs):
            # shapes=jax.tree_util.tree_map(lambda x: x.shape,xs)
            # print("Shapes before using:",shapes)  # DEBUG: commented out to reduce XLA compile memory
            batch_inputs,batch_integration_timesteps,batch_labels=xs
            dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*len(hiddens)
            hiddens=carry
            if batchnorm:
                (hiddens,logits), mod_vars = state.apply_fn( 
                    {"params": params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                    method='__call_rnn__'
                )
            else:
                (hiddens,logits), mod_vars = state.apply_fn(
                    {"params": params},
                    hiddens,
                    *batch_inputs,
                    *dones,
                    *batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                    method='__call_rnn__'
                )
            
            
            ce=cross_entropy_loss(logits, batch_labels)
            # jax.debug.print("Shape of CE: {}", ce.shape)
            # average cross-ent loss
            ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
            ce=ce.at[:,:,TIME_START_I:TIME_END_I].set(0)
            ce=ce.reshape(ce.shape[0],-1)
            loss = np.mean(ce)
            return (hiddens),(loss,mod_vars)
        # jax.debug.print("Shape of loss: {}", loss.shape)
        xs=(batch_inputs,batch_integration_timesteps,batch_labels)
        xs=jax.tree_util.tree_map(lambda x: np.array(np.split(x,2,axis=1)),xs)
        hiddens,y=jax.lax.scan(single_elem_loss,init_hiddens,xs)
        losses,mod_vars=y
        loss=np.mean(losses)
        mod_vars=jax.tree_util.tree_map(np.mean,mod_vars)
        return loss, mod_vars

    (loss, mod_vars), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    #return loss, mod_vars, grads, state
    return state, loss

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(5,),  # TODO: revert to 5 for batchnorm in pmap
    in_axes=(0, None, 0, 0, 0, None),
    # out_axes=(0, 0),
    # devices=global_devices
)
def train_step_old(
        state: train_state.TrainState,
        rng: jax.dtypes.prng_key,  # 3
        batch_inputs: Tuple[jax.Array, jax.Array], # 4
        batch_labels: jax.Array, # 5
        batch_integration_timesteps: Tuple[jax.Array, jax.Array], # 6
        batchnorm: bool, # 7
    ):
    #print('tracing par_loss_and_grad')
    def loss_fn(params):
        if batchnorm:
            logits, mod_vars = state.apply_fn( 
                {"params": params, "batch_stats": state.batch_stats},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = state.apply_fn(
                {"params": params},
                *batch_inputs, *batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        # average cross-ent loss
        loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)



    # UPDATE
    # calculate means over device dimension (first)
    loss = jax.lax.pmean(loss, axis_name="batch_devices")
    grads = jax.lax.pmean(grads, axis_name="batch_devices")

    if batchnorm:
        mod_vars = jax.lax.pmean(mod_vars, axis_name="batch_devices")
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    #return loss, mod_vars, grads, state
    return state, loss


def validate(state,
             apply_fn,
             testloader,
             seq_len,
             in_dim,
             batchnorm,
             num_devices,
             epoch,
             curtail_epoch=None,
             ignore_times: bool =False,
             step_rescale=1.0,
             apply_method: str ='__call_ar__',
             init_hiddens=(np.array([0])),
             log_ce_tables : bool =False):
    """Validation function that loops over batches"""
    # losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    losses, accuracies, preds = [], [], []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, num_devices)
        # print("eval step with method: ", apply_method)
        # print("Validataion: Inputs 0:5:", inputs[0][0,0:5,:])
        loss, acc, pred = eval_step(
            inputs, labels, integration_timesteps, state, apply_fn, batchnorm,apply_method,init_hiddens,ignore_times)
        # losses = np.append(losses, loss)
        # accuracies = np.append(accuracies, acc)

        # if (batch_idx==0) & (epoch%100==0): 
        #     np.set_printoptions(threshold=sys.maxsize)
        #     with open(f'/data1/sascha/data/losses/losses_batch_{batch_idx}_testing_applying_{apply_method}.txt', 'w') as f:
        #         print(loss, file=f)
        #     print("Printing logits of shape ", pred.shape, " to file")
        #     with open(f'/data1/sascha/data/losses/logits_batch_{batch_idx}_testing_applying_{apply_method}.txt', 'w') as f:
        #         print(pred[0,0,0:44,:], file=f)
        #     np.set_printoptions()
        #     print("Done Printing")


        losses.append(loss)
        accuracies.append(acc)
        if curtail_epoch is not None and batch_idx>=curtail_epoch:
            print(f"Ending epoch early at step {batch_idx} due to curtail_epoch arg.")
            break

    concat_loss=np.concatenate(losses,axis=0)
    concat_acc=np.concatenate(accuracies,axis=0)
    print(f"Concat Loss is {concat_loss.shape}")
    print(f"Concat Acc is {concat_acc.shape}")
    if log_ce_tables:
        acc_means=np.mean(concat_acc,axis=(0,1))
        ce_means=np.mean(concat_loss,axis=(0,1))
    else:
        ce_means=None
        acc_means=None
    aveloss, aveaccu = np.mean(concat_loss), np.mean(np.asarray(accuracies))
    del losses, accuracies
    return aveloss, aveaccu, ce_means,acc_means

@partial(
    jax.pmap,
    axis_name="batch_devices",
    static_broadcasted_argnums=(4,5,6,8),
    in_axes=(0, 0, 0, 0, None, None, None,None,None),
    # devices=global_devices
)
def eval_step(
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        state,
        #model,
        apply_fn,
        batchnorm,
        apply_method,
        init_hiddens,
        ignore_times,
    ):
    # print("checking for compile in eval_step function")

    batch_inputs=repeat_book(*batch_inputs,True)

    # BF16 Mixed Precision: params are already BF16 from initialization
    # No need for tree_map here - direct use saves overhead

    if apply_method == '__call_ar__':
        if batchnorm:
            logits = apply_fn({"params": state.params, "batch_stats": state.batch_stats},
                                *batch_inputs, *batch_integration_timesteps,
                                method=apply_method,
                                )
        else:
            logits = apply_fn({"params": state.params},
                                *batch_inputs, *batch_integration_timesteps,
                                method=apply_method,
                                )

        # BF16 Mixed Precision: Cast logits to FP32 for evaluation
        logits = logits.astype(np.float32)
    elif apply_method == '__call_rnn__':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3

        if batchnorm:
            hiddens,logits=apply_fn(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
        else:
            hiddens,logits=apply_fn(
                        {"params": state.params},
                        init_hiddens,
                        *batch_inputs,
                        *dones,
                        *batch_integration_timesteps,
                        method='__call_rnn__'
                    )
    elif apply_method == 'scan_rnn':
        dones=(np.zeros_like(batch_inputs[0],dtype=bool),)*3
        hiddens,logits=eval_rnn_scan(apply_fn,
                                     init_hiddens,
                                     state,
                                     batch_inputs,
                                     dones,
                                     batch_integration_timesteps,
                                     batchnorm)



    losses = cross_entropy_loss(logits, batch_labels)  
    if ignore_times:
        ce=losses
        ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
        ce_1=ce[:,:,:TIME_START_I]
        ce_2=ce[:,:,(TIME_END_I+1):]
        ce=np.concatenate([ce_1,ce_2],axis=2)
        ce=ce.reshape(ce.shape[0],-1)
        losses=ce
    accs = compute_accuracy(logits, batch_labels)
    if ignore_times:
        ce=accs
        ce=ce.reshape(ce.shape[0],-1,Message_Tokenizer.MSG_LEN)
        ce_1=ce[:,:,:TIME_START_I]
        ce_2=ce[:,:,(TIME_END_I+1):]
        ce=np.concatenate([ce_1,ce_2],axis=2)
        ce=ce.reshape(ce.shape[0],-1)
        accs=ce

    return losses, accs, logits


def eval_rnn_scan(apply_fn,hiddens,state,batch_inputs,batch_dones,batch_inttimes,batchnorm):
    def apply_fn_scan(carry,x):
        (hiddens,state)=carry
        (batch_inputs,batch_dones,batch_inttimes)=x
        if batchnorm:
            hiddens,logits=apply_fn(
                    {"params": state.params, "batch_stats": state.batch_stats},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        else:
            hiddens,logits=apply_fn(
                    {"params": state.params},
                    hiddens,
                    *batch_inputs,
                    *batch_dones,
                    *batch_inttimes,
                    method='__call_rnn__'
                )
        return (hiddens,state),logits
    #FIXME : Poor practice, but just for debugging purposes. 
    Ntoks=11000

    init=(hiddens,state)
    xs=(batch_inputs,batch_dones,batch_inttimes)
    
    xs=jax.tree_util.tree_map(partial(swap_leading,Ntoks),xs)
    
    carry_out,logits=jax.lax.scan(apply_fn_scan,init,xs)
    (hiddens,state)
    logits=np.concatenate(logits,axis=-2)
    return hiddens,logits
    
def swap_leading(targetsize,x):
    x=np.expand_dims(x,0)
    x=np.swapaxes(x,0,x.shape.index(targetsize))
    return x




