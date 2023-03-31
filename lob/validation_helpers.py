from typing import Tuple
from lob.encoding import Message_Tokenizer, Vocab
import jax
from jax import nn
import jax.numpy as np
from functools import partial

from lob.lobster_dataloader import LOBSTER_Dataset

v = Vocab()


def is_tok_valid(tok, field, vocab):
    tok = tok.tolist()
    if isinstance(field, str):
        return tok in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]]
    else:
        return [t in vocab.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[f]] 
                for t, f in zip(tok, field)]

def get_masked_idx(seq):
    """ Get the indices of the masked tokens in a given input (batched or not)
    """
    if seq.ndim == 1:
        seq = seq.reshape(-1, Message_Tokenizer.MSG_LEN)
    elif seq.ndim == 2:
        seq = seq.reshape(seq.shape[0], -1, Message_Tokenizer.MSG_LEN)
    return np.argwhere(seq == v.MASK_TOK)

def get_field_from_idx(idx):
    """ Get the field of a given index (or indices) in a message
    """
    return Message_Tokenizer.get_field_from_idx(idx)

def get_masked_fields(inp_maybe_batched):
    """ Get the fields of the masked tokens in a given input (batched or not)
    """
    mask_pos = get_masked_idx(inp_maybe_batched)
    return get_field_from_idx(mask_pos[..., -1])

def get_valid_toks_for_field(fields):
    """ Get the valid labels for given fields
    """
    return tuple(tuple(
        v.DECODING[Message_Tokenizer.FIELD_ENC_TYPES[field]].keys())
          for field in fields)

def get_valid_toks_for_input(inp_maybe_batched):
    """ Get the valid labels for a given input (batched or not)
    """
    fields = get_masked_fields(inp_maybe_batched)
    return get_valid_toks_for_field(fields)

def valid_prediction_mass(pred, fields, top_n=None):
    """ for a predicted distribution over tokens get the total mass of the
        syntactically valid labels
        top_n: 
    """
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    assert (len(fields) == pred.shape[0])
    valid_toks = get_valid_toks_for_field(fields)
    dim_0_i = [i for i, tok_list in enumerate(valid_toks) for tok in tok_list]
    dim_1_i = [tok for tok_list in valid_toks for tok in tok_list]
    mask_valid = np.zeros_like(pred)
    mask_valid = mask_valid.at[dim_0_i, dim_1_i].set(1)

    if top_n is not None:
        mask_top_n = mask_n_highest(pred, top_n)
        mask_valid = mask_valid * mask_top_n
        top_n_mass = np.sum(np.exp(pred) * mask_top_n, axis=1)
    else:
        top_n_mass = 1.

    return (np.sum(np.exp(pred) * mask_valid, axis=1)) / top_n_mass

def mask_n_highest(a, n):
    """ Return a mask for the n highest values in the last axis
        for a given array
    """
    n_th_largest = np.sort(a, axis=-1)[..., -n]
    # add leading dimensions to match pred
    n_th_largest = n_th_largest.reshape((-1,) + (1,)*(a.ndim-1))
    mask_top_n = np.zeros_like(a, dtype=bool)
    #mask_top_n = mask_top_n.at[a >= n_th_largest].set(True)
    mask_top_n = np.where(a >= n_th_largest, True, False)
    return mask_top_n

def pred_rank(pred, labels):
    """ Get the rank of the correct label in the predicted distribution.
        Lower is better (0 is correct prediction).
    """
    correct_mask = nn.one_hot(labels.astype(int), pred.shape[-1]).astype(bool)
    # ::-1 sorts in descending order (0 is highest rank)
    a = pred.argsort(axis=-1)
    ranks = a[..., ::-1].argsort(axis=-1)
    return ranks[correct_mask]

def fill_predicted_toks(seq, pred, top_n=1, rng=None):
    """ Set the predicted token in the given sequence
        when top_n=1, the argmax is used, otherwise a random sample
        from the top_n highest scores is used (propotional to the score)
        rng cannot be None when top_n > 1
    """
    if top_n == 1:
        vals = pred.argmax(axis=-1)
    else:
        vals = sample_pred(pred, top_n, rng)
    return seq.at[seq == v.MASK_TOK].set(vals)

#@partial(np.vectorize, signature="(n),(),(n)->()")
@partial(jax.vmap, in_axes=(0, None, 0))
def sample_pred(pred, top_n, rng):
    """ Sample from the top_n predicted labels
    """
    mask_top_n = mask_n_highest(pred, top_n)
    idx = np.arange(pred.shape[0]).reshape(pred.shape)
    p = pred * mask_top_n
    p = p / p.sum(axis=-1, keepdims=True)
    return jax.random.choice(rng, idx, p=p)

def append_hid_msg(seq):
    """ Append a new empty (HID token) message to a sequence
        removing first message to keep seq_len constant
    """
    l = Message_Tokenizer.MSG_LEN
    return np.concatenate([seq[l:], np.full((Message_Tokenizer.MSG_LEN,), Vocab.HIDDEN_TOK)])

def mask_last_msg_in_seq(
        seq: np.ndarray,
        i: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    l = Message_Tokenizer.MSG_LEN
    assert (i >= -l) and (i < l), "i must be in [-MSG_LEN, MSG_LEN)"
    if i >= 0:
        i += len(seq) - l
    y = seq[i]
    return seq.at[i].set(v.MASK_TOK), y

@partial(jax.jit, static_argnums=(3, 4))
def predict(
        batch_inputs,
        batch_integration_timesteps,
        state,
        model,
        batchnorm,
    ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             )

    return logits

def pred_next_tok(
        seq,
        state,
        model,
        batchnorm,
        sample_top_n,
        mask_i,
        rng,
        vocab_len,
        new_msg=False,
    ):
    """ Predict the next token with index i of the last message in the sequence
        if new_msg=True, a new empty message is appended to the sequence
        Returns the updated sequence
        TODO: add flag to only sample from syntactically valid tokens
    """

    # create masked message for prediction
    if new_msg:
        seq = append_hid_msg(seq)
    seq, _ = mask_last_msg_in_seq(seq, mask_i)
    # inference
    integration_timesteps = np.ones((1, len(seq)))
    seq_onehot = nn.one_hot(
        np.expand_dims(seq, axis=0), vocab_len).astype(float)
    logits = predict(
        seq_onehot,
        integration_timesteps, state, model, batchnorm)
    # update sequence
    # note: rng arg expects one element per batch element
    seq = fill_predicted_toks(seq, logits, sample_top_n, np.array([rng]))
    return seq

def pred_msg(seq, n_messages, state, model, batchnorm, rng):
    l = Message_Tokenizer.MSG_LEN
    for m_i in range(n_messages):
        new_msg = True
        for i in range(l):
            seq = pred_next_tok(
                seq,
                state,
                model,
                batchnorm,
                sample_top_n=5,
                mask_i=i,
                new_msg=new_msg,
                vocab_len=len(v),
                rng=rng,
            )
            new_msg = False
    return seq
