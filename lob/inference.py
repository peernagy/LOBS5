import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as onp
import os
import sys
import pandas as pd
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
logger = logging.getLogger(__name__)
#debug = lambda *args: logger.debug(' '.join((str(arg) for arg in args)))
#info = lambda *args: logger.info(' '.join((str(arg) for arg in args)))
from utils import debug, info

import preproc
import validation_helpers as valh
import encoding
from encoding import Message_Tokenizer, Vocab


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderbook as job
from gym_exchange.environment.base_env.assets.orderflow import OrderIdGenerator


# time tokens aren't generated but calculated using delta_t
# hence, skip generation from TIME_START_I (inclusive) to TIME_END_I (exclusive)
TIME_START_I, _ = valh.get_idx_from_field('time_s')
_, TIME_END_I = valh.get_idx_from_field('time_ns')


def init_msgs_from_l2(book: Union[pd.Series, onp.ndarray]) -> jnp.ndarray:
    """"""
    orderbookLevels = len(book) // 4  # price/quantity for bid/ask
    data = jnp.array(book).reshape(int(orderbookLevels*2),2)
    newarr = jnp.zeros((int(orderbookLevels*2),8))
    initOB = newarr \
        .at[:,3].set(data[:,0]) \
        .at[:,2].set(data[:,1]) \
        .at[:,0].set(1) \
        .at[0:orderbookLevels*4:2,1].set(-1) \
        .at[1:orderbookLevels*4:2,1].set(1) \
        .at[:,4].set(0) \
        .at[:,5].set(job.INITID) \
        .at[:,6].set(34200) \
        .at[:,7].set(0).astype('int32')
    return initOB


def msgs_to_jnp(m_df: pd.DataFrame) -> jnp.ndarray:
    """"""
    m_df = m_df.copy()
    cols = ['Time', 'Type', 'OrderID', 'Quantity', 'Price', 'Side']
    if m_df.shape[1] == 7:
        cols += ["TradeID"]
    m_df.columns = cols
    m_df['TradeID'] = 0  #  TODO: should be TraderID for multi-agent support
    col_order=['Type','Side','Quantity','Price','TradeID','OrderID','Time']
    m_df = m_df[col_order]
    m_df = m_df[(m_df['Type'] != 6) & (m_df['Type'] != 7) & (m_df['Type'] != 5)]
    time = m_df["Time"].astype('string').str.split('.',expand=True)
    m_df[["TimeWhole","TimeDec"]] = time.astype('int32')
    m_df = m_df.drop("Time", axis=1)
    mJNP = jnp.array(m_df)
    return mJNP

def reset_orderbook(
        b: OrderBook,
        l2_book: Optional[Union[pd.Series, onp.ndarray]] = None,
    ) -> None:
    """"""
    b.orderbook_array = b.orderbook_array.at[:].set(-1)
    if l2_book is not None:
        msgs = init_msgs_from_l2(l2_book)
        b.process_orders_array(msgs)

def copy_orderbook(
        b: OrderBook
    ) -> OrderBook:
    b_copy = OrderBook(price_levels=b.price_levels, orderQueueLen=b.orderQueueLen)
    b_copy.orderbook_array = b.orderbook_array.copy()
    return b_copy

def get_sim(
        init_l2_book: Union[pd.Series, onp.ndarray],
        replay_msgs_raw: pd.DataFrame,
        sim_book_levels: int,
        sim_queue_len: int,
    ) -> Tuple[OrderBook, jax.Array]:
    """"""
    # reset simulator
    sim = OrderBook(price_levels=sim_book_levels, orderQueueLen=sim_queue_len)
    # init simulator at the start of the sequence
    reset_orderbook(sim, init_l2_book)
    # replay sequence in simulator (actual)
    # so that sim is at the same state as the model
    replay = msgs_to_jnp(replay_msgs_raw)
    trades = sim.process_orders_array(replay)
    return sim, trades


def get_sim_msg(
        pred_msg_enc: jnp.ndarray,
        m_seq: jnp.ndarray,
        m_seq_raw: pd.DataFrame,
        sim: OrderBook,
        #tok: Message_Tokenizer,
        #v: Vocab,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[jnp.ndarray], Optional[Dict[str, Any]]]:
    """"""
    # decoded predicted message
    # pred_msg = tok.decode(pred_msg_enc, v).squeeze()
    pred_msg = encoding.decode_msg(pred_msg_enc, encoder)
    #debug('decoded predicted message:', pred_msg)

    new_part = pred_msg[: Message_Tokenizer.N_NEW_FIELDS]
    ref_part = pred_msg[Message_Tokenizer.N_NEW_FIELDS: ]

    if onp.isnan(new_part).any():
        debug('new_part contains NaNs', new_part)
        return None, None, None

    event_type = int(pred_msg[Message_Tokenizer.FIELD_I['event_type']])
    quantity = int(pred_msg[Message_Tokenizer.FIELD_I['size']])
    side = int(pred_msg[Message_Tokenizer.FIELD_I['direction']])
    rel_price = int(pred_msg[Message_Tokenizer.FIELD_I['price']])
    # delta_t = int(pred_msg[Message_Tokenizer.FIELD_I['delta_t']])
    delta_t_s = int(pred_msg[Message_Tokenizer.FIELD_I['delta_t_s']])
    delta_t_ns = int(pred_msg[Message_Tokenizer.FIELD_I['delta_t_ns']])
    # time = int(pred_msg[Message_Tokenizer.FIELD_I['time']])
    time_s = int(pred_msg[Message_Tokenizer.FIELD_I['time_s']])
    time_ns = int(pred_msg[Message_Tokenizer.FIELD_I['time_ns']])

    # new order: no modification values present (all NA)
    # should be new LIMIT ORDER (1)
    #if onp.isnan(ref_part).all():

    # NEW LIMIT ORDER
    if event_type == 1:
        order_dict, msg_corr, raw_dict = get_sim_msg_new(
            sim,
            event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns, #delta_t, time,
            new_order_id,
            tick_size,
            encoder,
        )

    # modification / deletion / execution of existing order
    else:
        # error in msg: some modifier field is nan
        if onp.isnan(ref_part).any():
            debug('ref_part has nan', ref_part)
            return None, None, None
        
        # resolve ref fields
        rel_price_ref = int(pred_msg[Message_Tokenizer.FIELD_I['price_ref']])
        quantity_ref = int(pred_msg[Message_Tokenizer.FIELD_I['size_ref']])
        # time_ref = int(pred_msg[Message_Tokenizer.FIELD_I['time_ref']])
        time_s_ref = int(pred_msg[Message_Tokenizer.FIELD_I['time_s_ref']])
        time_ns_ref = int(pred_msg[Message_Tokenizer.FIELD_I['time_ns_ref']])

        # cancel / delete
        if event_type in {2, 3}:
            order_dict, msg_corr, raw_dict = get_sim_msg_mod(
                pred_msg_enc,
                event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns,
                rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
                m_seq,
                m_seq_raw,
                sim,
                #tok,
                #v,
                tick_size,
                encoder,)

        # modify
        elif event_type == 4:
            order_dict, msg_corr, raw_dict = get_sim_msg_exec(
                pred_msg_enc,
                event_type, quantity, side, rel_price, delta_t_s, delta_t_ns, time_s, time_ns,
                rel_price_ref, quantity_ref, time_s_ref, time_ns_ref,
                m_seq,
                m_seq_raw,
                new_order_id,
                sim,
                #tok,
                #v,
                tick_size,
                encoder,
            )

        # Invalid type of modification
        else:
            return None, None, None

    return order_dict, msg_corr, raw_dict


def get_sim_msg_new(
        sim: OrderBook,
        event_type: int,
        quantity: int,
        side: int,
        rel_price: int,
        # delta_t: int,
        delta_t_s: int,
        delta_t_ns: int,
        # time: int,
        time_s: int,
        time_ns: int,
        new_order_id: int,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:
        
        # if onp.isnan(new_part).any():
        #     return None, None, None

        # event_type = int(new_part[Message_Tokenizer.FIELDS.index('event_type')])
        # quantity = int(new_part[Message_Tokenizer.FIELDS.index('size')])
        # side = int(new_part[Message_Tokenizer.FIELDS.index('direction')])
        # rel_price = int(new_part[Message_Tokenizer.FIELDS.index('price')])
        # time = int(new_part[Message_Tokenizer.FIELDS.index('time')])

        assert event_type == 1, 'Invalid event type for new order: ' + str(event_type)
        # new limit order
        debug('NEW LIMIT ORDER')
        # convert relative to absolute price
        price = sim.get_best_bid() + rel_price * tick_size

        order_dict = {
            'timestamp': format(time_s + time_ns * 1e-9 + 9.5 * 3600, '.9f'),  # TODO: check why str()??
            'type': 'limit',
            'order_id': new_order_id, 
            'quantity': quantity,
            'price': price,
            'side': 'ask' if side == 0 else 'bid',
            'trade_id': 0  # should be trader_id in future
        }

        # msg format to update raw data
        raw_dict = sim_order_to_raw({**order_dict, 'side': side}, event_type)

        # msg_corr = onp.array([
        #     str(event_type),
        #     str(side),
        #     ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
        #     str(quantity).zfill(4), 
        #     str(int(delta_t)).zfill(12),
        #     str(int(time)).zfill(15),
        # ])

        msg_corr = jnp.array([
            event_type,
            side,
            rel_price,
            quantity,
            delta_t_s,
            delta_t_ns,
            time_s,
            time_ns,
        ])

        # print('msg_corr \n', msg_corr)

        # encode corrected message
        # TODO: make this an arg?
        #tok = Message_Tokenizer()
        #v = Vocab()
        #msg_corr = tok.encode_msg(msg_corr, v)
        msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]

        nan_part = jnp.array((Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN) * [Vocab.NA_TOK])
        msg_corr = jnp.concatenate([msg_corr, nan_part])

        return order_dict, msg_corr, raw_dict


def rel_to_abs_price(
        p_rel: int,
        sim: OrderBook,
        tick_size: int = 100,
    ) -> int:

    assert tick_size % 10 == 0
    #round_to = int(-onp.log10(tick_size))
    #p_ref = onp.round((sim.get_best_bid() + sim.get_best_ask()) / 2, round_to)
    p_ref = (sim.get_best_bid() + sim.get_best_ask()) / 2
    p_ref = (p_ref // tick_size) * tick_size
    return p_ref + p_rel * tick_size

@jax.jit
def construct_orig_msg_enc(
        pred_msg_enc: jax.Array,
        #v: Vocab,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
    """ Reconstructs encoded original message WITHOUT Delta t
        from encoded message string --> delta_t field is filled with NA_TOK
    """
    return jnp.concatenate([
        #jnp.array([v.ENCODING['event_type']['1']]),
        encoding.encode(jnp.array([1]), *encoder['event_type']),
        pred_msg_enc[slice(*valh.get_idx_from_field('direction'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('price_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size_ref'))],
        # NOTE: no delta_t here
        jnp.full(
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_s']] + \
            Message_Tokenizer.TOK_LENS[Message_Tokenizer.FIELD_I['delta_t_ns']],
            Vocab.NA_TOK
        ),
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s_ref'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns_ref'))],
    ])

def convert_msg_to_ref(
        pred_msg_enc: jnp.ndarray,
    ):
    """ Converts encoded message to reference message part,
        i.e. (price, size, time) tokens
    """
    return jnp.concatenate([
        pred_msg_enc[slice(*valh.get_idx_from_field('price'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('size'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_s'))],
        pred_msg_enc[slice(*valh.get_idx_from_field('time_ns'))],
    ])

def get_sim_msg_mod(
        pred_msg_enc: jnp.ndarray,
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        # delta_t: int,
        delta_t_s: int,
        delta_t_ns: int,
        # time: int,
        time_s: int,
        time_ns: int,

        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,

        m_seq: jnp.ndarray,
        m_seq_raw: pd.DataFrame,
        sim: OrderBook,
        #tok: Message_Tokenizer,
        #v: Vocab,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:

    debug('ORDER CANCEL / DELETE')
    REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

    # the actual price of the order to be modified
    # p_mod_raw = sim.get_best_bid() + int(modif_part[3]) * tick_size
    p_mod_raw = rel_to_abs_price(rel_price, sim, tick_size)

    debug('rel price', rel_price)
    debug('side', side)
    debug('removed_quantity (raw)', removed_quantity)
    debug('total liquidity at price', sim.get_volume_at_price(side, p_mod_raw))
    debug('event_type:', event_type)

    assert event_type != 4, 'event_type 4 should be handled separately'
    
    # orig order before sequence start (no ref given or part missing)
    if onp.isnan(rel_price_ref) or onp.isnan(quantity_ref) or onp.isnan(time_s_ref) or onp.isnan(time_ns_ref):
        debug('NaN ref value found')
        debug('rel_price_ref', rel_price_ref, 'quantity_ref', quantity_ref, 'time_s_ref', time_s_ref, 'time_ns_ref', time_ns_ref)
        # if no init volume remains at price, discard current message
        if sim.get_init_volume_at_price(side, p_mod_raw) == 0:
            return None, None, None
        order_id = job.INITID
        #orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])
    
    # search for original order to get correct ID
    else:
        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        # ref part is only needed to match to an order ID
        # find original msg index location in the sequence (if it exists)
        #orig_enc = pred_msg_enc[: len(pred_msg_enc) // 2]
        orig_enc = construct_orig_msg_enc(pred_msg_enc, encoder)
        debug('orig_enc', orig_enc)

        mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim)
        orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, seq_mask=mask)
        #orig_i = valh.find_orig_msg(orig_enc, m_seq)
        
        # didn't find matching original message
        if orig_i is None:
            if sim.get_init_volume_at_price(side, p_mod_raw) == 0:
                return None, None, None
            order_id = job.INITID
            orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])
        
        # found matching original message
        else:
            # get order ID from raw data for simulator
            order_id = int(m_seq_raw.iloc[orig_i].order_id)
            # found original message: convert to ref part
            if m_seq_raw.iloc[orig_i].event_type == 1:
                # orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])
                orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
            # found reference to original message
            else:
                # take ref fields from matching message
                orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])

    # get remaining quantity in book for given order ID
    debug('looking for order', order_id, 'at price', p_mod_raw)
    remaining_quantity = sim.get_order_by_id_and_price(order_id, p_mod_raw)[0]
    #if order_id != job.INITID:
    #    assert sim.get_order_by_id(order_id)[0] == remaining_quantity, \
    #        f'order_id: {order_id}, remaining_quantity: {remaining_quantity}'
    # --> could be that p_mod_raw is wrong, hence it finds -1 volume
    #    --> find different order to modify (price beats other fields)
    debug('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        remaining_quantity = sim.get_init_volume_at_price(side, p_mod_raw)
        debug('remaining init qu.', remaining_quantity)
        # if no init volume remains at price, discard current message
        if remaining_quantity == 0:
            return None, None, None
        order_id = job.INITID
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity
        # change partial cancel to full delete
        if event_type == 2:
            event_type = 3
    # change full delete to partial cancel
    elif event_type == 3:
        event_type = 2

    debug(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')

    if event_type == 2:
        sim_type = 'cancel'
        sim_side = side
    elif event_type == 3:
        sim_type = 'delete'
        sim_side = side
    else:
        raise ValueError('Invalid event type')
    
    order_dict = {
        # format decimals to nanosecond precision
        'timestamp': format(time_s + time_ns * 1e-9 + 9.5 * 3600, '.9f'),
        'type': sim_type,
        'order_id': order_id, 
        'quantity': removed_quantity,
        'price': p_mod_raw,
        'side': 'ask' if sim_side == 0 else 'bid',  # TODO: should be 'buy' or 'sell'
        'trade_id': 0  # should be trader_id in future
    }
    # msg format to update raw data sequence
    raw_dict = sim_order_to_raw({**order_dict, 'side': side}, event_type)
    
    # msg_corr = onp.array([
    #     str(event_type),
    #     str(side),
    #     ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
    #     str(removed_quantity).zfill(4), 
    #     str(int(delta_t)).zfill(12),
    #     str(int(time)).zfill(15),
    # ])

    # TODO: encoding encodes only full message, not partial --> overwrite after encoding

    msg_corr = jnp.array([
        event_type,
        side,
        rel_price,
        removed_quantity,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
    ])
    # print('msg_corr\n', msg_corr)

    # encode corrected message
    #msg_corr = tok.encode_msg(msg_corr, v)
    msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]
    # print('msg_corr enc (part)\n', msg_corr)
    msg_corr = onp.concatenate([msg_corr, orig_msg_found])
    # print('msg_corr enc (full)\n', msg_corr)

    return order_dict, msg_corr, raw_dict


def get_sim_msg_exec(
        pred_msg_enc: jnp.ndarray,
        #new_part: onp.ndarray,
        #ref_part: onp.ndarray,
        event_type: int,
        removed_quantity: int,
        side: int,
        rel_price: int,
        #delta_t: int,
        delta_t_s: int,
        delta_t_ns: int,
        # time: int,
        time_s: int,
        time_ns: int,

        rel_price_ref: int,
        quantity_ref: int,
        time_s_ref: int,
        time_ns_ref: int,
        
        m_seq: onp.ndarray,
        m_seq_raw: pd.DataFrame,
        new_order_id: int,
        sim: OrderBook,
        #tok: Message_Tokenizer,
        #v: Vocab,
        tick_size: int,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:

    debug('ORDER EXECUTION')
    REF_LEN = Message_Tokenizer.MSG_LEN - Message_Tokenizer.NEW_MSG_LEN

    # the actual price of the order to be modified
    p_mod_raw = rel_to_abs_price(rel_price, sim, tick_size)

    debug('event_type:', event_type)
    assert event_type == 4
    debug('side:', side)
    debug('removed_quantity:', removed_quantity)

    # get order against which execution is happening
    passive_order = sim.orderbook_array[side, 0, 0]
    if side == 0:
        debug('   execution on ask side (buyer initiated)')
        debug('   best ask:', passive_order[1])
    else:
        debug('   execution on bid side (seller initiated)')
        debug('   best bid:', passive_order[1])
    if p_mod_raw != passive_order[1]:
        debug('EXECUTION AT WRONG PRICE', 'gen:', p_mod_raw, 'p_passive', passive_order[1], 'correcting...')
        p_mod_raw = passive_order[1]
        #return None, None, None

    remaining_quantity = passive_order[0]
    debug('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        debug('NOTHING TO EXECUTE AGAINST (empty side of book)')
        return None, None, None

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity

    debug(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')
    
    order_dict = {
        # format decimals to nanosecond precision
        'timestamp': format(time_s + time_ns * 1e-9 + 9.5 * 3600, '.9f'),
        'type': 'limit',
        'order_id': new_order_id, 
        'quantity': removed_quantity,
        'price': p_mod_raw,
        'side': 'ask' if side == 1 else 'bid',  # CAVE: other side as cancels
        'trade_id': 0  # should be trader_id in future
    }
    
    # msg format to update raw data, CAVE: execution is opposite side from limit order
    raw_dict = sim_order_to_raw({**order_dict, 'side': side}, event_type)

    # msg_corr = onp.array([
    #     str(event_type),
    #     str(side),
    #     ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
    #     str(removed_quantity).zfill(4), 
    #     str(int(delta_t)).zfill(12),
    #     str(int(time)).zfill(15),
    # ])

    msg_corr = jnp.array([
        event_type,
        side,
        rel_price,
        removed_quantity,
        delta_t_s,
        delta_t_ns,
        time_s,
        time_ns,
    ])

    # correct the order which is executed in the sequence
    order_id = passive_order[3]
    orig_i = onp.argwhere(m_seq_raw.order_id.values == order_id)
    # found correct order
    if len(orig_i) > 0:
        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        #debug('orig_i', orig_i)
        #debug('m_seq[orig_i]', m_seq[orig_i])
        orig_i = orig_i.flatten()[0]

        # found original message: convert to ref part
        if m_seq_raw.iloc[orig_i].event_type == 1:
            # orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])
            orig_msg_found = convert_msg_to_ref(m_seq[orig_i])
        # found reference to original message
        else:
            # take ref fields from matching message
            orig_msg_found = onp.array(m_seq[orig_i, -REF_LEN: ])

    # didn't find correct order (e.g. INITID)
    else:
        orig_msg_found = onp.array(REF_LEN * [Vocab.NA_TOK])

    # encode corrected message
    #msg_corr = tok.encode_msg(msg_corr, v)
    msg_corr = encoding.encode_msg(msg_corr, encoder)[: Message_Tokenizer.NEW_MSG_LEN]
    msg_corr = onp.concatenate([msg_corr, orig_msg_found])

    return order_dict, msg_corr, raw_dict


def get_invalid_ref_mask(
        m_seq_raw: pd.DataFrame,
        p_mod_raw: int,
        sim: OrderBook
    ):
    """
    """
    # filter sequence to prices matching the correct price level
    wrong_price_mask = (m_seq_raw.price != p_mod_raw).astype(bool).values
    # to filter to orders still in the book: get order IDs from sim
    ids = sim.get_order_ids()
    # cave: convert from jax to numpy for isin() to work
    ids = onp.array(ids[ids != -1])
    not_in_book_mask = ~(m_seq_raw.order_id.isin(ids)).astype(bool).values
    mask = wrong_price_mask | not_in_book_mask
    return mask


def sim_order_to_raw(order_dict: Dict[str, Any], event_type: int):
    return {
        'time': float(order_dict['timestamp']),
        'event_type': int(event_type),
        'order_id': int(order_dict['order_id']),
        'size': int(order_dict['quantity']),
        'price': int(order_dict['price']),
        # convert from 0/1 to -1/1
        'direction': int(order_dict['side']) * 2 - 1,
    }

@jax.jit
def add_times(
        a_s: jax.Array,
        a_ns: jax.Array,
        b_s: jax.Array,
        b_ns: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
    """ Adds two timestamps given as seconds and nanoseconds each (both fit in int32)
        and returns new timestamp, split into time_s and time_ns
    """
    #delta_t_s = delta_t // 1000000000
    #delta_t_ns = delta_t % 1000000000
    
    a_ns = b_ns + a_ns
    extra_s = a_ns // 1000000000
    a_ns = a_ns % 1000000000
    # print(a_ns, extra_s)

    a_s = a_s + b_s + extra_s
    return a_s, a_ns

def generate(
        m_seq: jax.Array,
        b_seq: jax.Array,
        m_seq_raw: jax.Array,
        n_msg_todo: int,
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        encoder: Dict[str, Tuple[jax.Array, jax.Array]],
        rng: jax.random.PRNGKeyArray,
        sample_top_n: int = 50,
        tick_size: int = 100,
    ) -> Tuple[jax.Array, jax.Array, pd.DataFrame, jax.Array]:

    id_gen = OrderIdGenerator()
    l = Message_Tokenizer.MSG_LEN
    last_start_i = m_seq.shape[0] - l
    v = Vocab()
    #tok = Message_Tokenizer()
    vocab_len = len(v)
    valid_mask_array = valh.syntax_validation_matrix()
    l2_book_states = []
    m_seq = m_seq.copy()
    b_seq = b_seq.copy()
    m_seq_raw = m_seq_raw.copy()
    num_errors = 0

    time_s_start_i, time_s_end_i = valh.get_idx_from_field('time_s')
    time_ns_start_i, time_ns_end_i = valh.get_idx_from_field('time_ns')
    delta_t_s_start_i, delta_t_s_end_i = valh.get_idx_from_field('delta_t_s')
    delta_t_ns_start_i, delta_t_ns_end_i = valh.get_idx_from_field('delta_t_ns')

    while n_msg_todo > 0:
        rng, rng_ = jax.random.split(rng)
        
        #time_init = tok.decode_toks(onp.array(m_seq[last_start_i + time_start_i: last_start_i + time_end_i]), v)
        time_init_s = encoding.decode(
            m_seq[last_start_i + time_s_start_i: last_start_i + time_s_end_i],
            *encoder['time'],
        )
        time_init_s = encoding.combine_field(time_init_s, 3)
        
        time_init_ns = encoding.decode(
            m_seq[last_start_i + time_ns_start_i: last_start_i + time_ns_end_i],
            *encoder['time'],
        )
        time_init_ns = encoding.combine_field(time_init_ns, 3)

        # jax.block_until_ready(m_seq)
        # roll sequence one step forward
        m_seq = valh.append_hid_msg(m_seq)
        # jax.block_until_ready(m_seq)

        # TODO: calculating time in case where generation is not sequentially left to right
        #       --> check if delta_t complete --> calc time once

        # get next message: generate l tokens
        for mask_i in range(l):
            # calculate time once from previous time and delta_t
            if mask_i == TIME_START_I:
                delta_t_s_toks = m_seq[last_start_i + delta_t_s_start_i: last_start_i + delta_t_s_end_i]
                delta_t_ns_toks = m_seq[last_start_i + delta_t_ns_start_i: last_start_i + delta_t_ns_end_i]
                debug('delta_t_toks', delta_t_s_toks, delta_t_ns_toks)
                # dec = v.DECODING['time']
                # delta_t = int(''.join([dec[t] for t in onp.array(delta_t_toks)]))
                delta_t_s = encoding.decode(delta_t_s_toks, *encoder['time'])
                delta_t_s = encoding.combine_field(delta_t_s, 3)
                delta_t_ns = encoding.decode(delta_t_ns_toks, *encoder['time'])
                delta_t_ns = encoding.combine_field(delta_t_ns, 3)

                debug('delta_t', delta_t_s, delta_t_ns)
                # time = time_init + delta_t
                time_s, time_ns = add_times(time_init_s, time_init_ns, delta_t_s, delta_t_ns)
                debug('time', time_s, time_ns)
                
                # encode time and add to sequence
                #time_toks = tok.encode_field(str(time).zfill(15), 'time', v)
                time_s = encoding.split_field(time_s, 2, 3)
                time_s_toks = encoding.encode(time_s, *encoder['time'])
                time_ns = encoding.split_field(time_ns, 3, 3)
                time_ns_toks = encoding.encode(time_ns, *encoder['time'])

                debug('time_toks', time_s_toks, time_ns_toks)
                m_seq = m_seq.at[last_start_i + time_s_start_i: last_start_i + time_ns_end_i].set(
                    jnp.hstack([time_s_toks, time_ns_toks]))
                # jax.block_until_ready(m_seq)
            # skip generation of time tokens
            if (mask_i >= TIME_START_I) and (mask_i < TIME_END_I):
                continue

            # syntactically valid tokens for current message position
            #valid_mask = valid_mask_array[mask_i]
            valid_mask = valh.get_valid_mask(valid_mask_array, mask_i)
            # jax.block_until_ready(m_seq)
            m_seq, _ = valh.mask_last_msg_in_seq(m_seq, mask_i)
            # jax.block_until_ready(m_seq)
            input = (
                one_hot(
                    jnp.expand_dims(m_seq, axis=0), vocab_len
                ).astype(float),
                jnp.expand_dims(b_seq, axis=0)
            )
            integration_timesteps = (
                jnp.ones((1, len(m_seq))), 
                jnp.ones((1, len(b_seq)))
            )
            # jax.block_until_ready(input)
            logits = valh.predict(
                input,
                integration_timesteps, train_state, model, batchnorm)
            # jax.block_until_ready(logits)
            
            # filter out (syntactically) invalid tokens for current position
            if valid_mask is not None:
               logits = valh.filter_valid_pred(logits, valid_mask)
               # jax.block_until_ready(logits)

            # update sequence
            # note: rng arg expects one element per batch element
            rng, rng_ = jax.random.split(rng)
            # jax.block_until_ready(m_seq)
            m_seq = valh.fill_predicted_toks(m_seq, logits, sample_top_n, jnp.array([rng_]))
            # jax.block_until_ready(m_seq)

        # jax.block_until_ready(m_seq)
        debug(m_seq[-l:])
        debug('decoded:')
        #debug(tok.decode(m_seq[-l:], v))
        ### process generated message

        #m_seq_raw = m.iloc[end_i - n_messages : end_i]
        #m_seq_raw = m_raw.iloc[end_i - n_messages + 1 : end_i]
        order_id = id_gen.step()

        # jax.block_until_ready(m_seq)

        # parse generated message for simulator, also getting corrected raw message
        # (needs to be encoded and overwrite originally generated message)
        sim_msg, msg_corr, msg_raw = get_sim_msg(
            m_seq[-l:],  # the generated message
            m_seq[:-l],  # sequence without generated message
            m_seq_raw.iloc[1:],   # raw data (same length as sequence without generated message)
            sim,
            #tok,
            #v,
            new_order_id=order_id,
            tick_size=tick_size,
            encoder=encoder,
        )

        if sim_msg is None:
            info('invalid message - discarding...\n')
            num_errors += 1

            # cut away generated message and pad begginning of sequence
            m_seq = onp.concatenate([
                onp.full((l,), Vocab.NA_TOK),
                m_seq[: -l]])
            continue

        info(sim_msg)
        #debug('quantitity', sim_msg['quantity'], 'price', sim_msg['price'])

        # replace faulty message in sequence with corrected message
        #debug('before', m_seq[-l:])
        #debug('after', msg_corr)
        m_seq = m_seq.at[-l:].set(msg_corr)
        # append new message to raw data
        #m_seq_raw = m_seq_raw.iloc[1:].append(msg_raw, ignore_index=True)
        m_seq_raw = pd.concat([
                m_seq_raw.iloc[1:],
                pd.DataFrame([msg_raw])
            ],
            ignore_index=True)
        debug('len(m_seq_raw)', len(m_seq_raw))
        debug('new raw msg', m_seq_raw.iloc[-1])

        #p_mid_old = onp.round((sim.get_best_ask() + sim.get_best_bid()) / 2, -2).astype(int)
        p_mid_old = (sim.get_best_ask() + sim.get_best_bid()) / 2
        p_mid_old = (p_mid_old // tick_size) * tick_size
        # feed message to simulator, updating book state
        _trades = sim.process_order(sim_msg)
        #p_mid_new = onp.round((sim.get_best_ask() + sim.get_best_bid()) / 2, -2).astype(int)
        p_mid_new = (sim.get_best_ask() + sim.get_best_bid()) / 2
        p_mid_new = (p_mid_new // tick_size) * tick_size
        p_change = int((p_mid_new - p_mid_old) // tick_size)

        # get new book state
        #book = sim.get_L2_state()
        book = job.get_l2_state(sim.orderbook_array)
        l2_book_states.append(book)

        new_book_raw = jnp.concatenate([jnp.array([p_change]), book]).reshape(1,-1)
        new_book = preproc.transform_L2_state(new_book_raw, 500, 100)
        #debug('new_book', new_book.shape, new_book)
        
        # update book sequence
        b_seq = jnp.concatenate([b_seq[1:], new_book])

        debug('p_change', p_change, '\n')

        n_msg_todo -= 1

    return m_seq, b_seq, m_seq_raw, jnp.array(l2_book_states), num_errors
