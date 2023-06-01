import jax
import jax.numpy as jnp
from jax.nn import one_hot
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as onp
import os
import sys
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

import preproc
import validation_helpers as valh
from encoding import Message_Tokenizer, Vocab


# add git submodule to path to allow imports to work
submodule_name = 'AlphaTrade'
(parent_folder_path, current_dir) = os.path.split(os.path.abspath(''))
sys.path.append(os.path.join(parent_folder_path, submodule_name))
from gymnax_exchange.jaxob.jorderbook import OrderBook
import gymnax_exchange.jaxob.JaxOrderbook as job
from gym_exchange.environment.base_env.assets.orderflow import OrderIdGenerator


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
        pred_msg_enc: onp.ndarray,
        m_seq: onp.ndarray,
        m_seq_raw: pd.DataFrame,
        sim: OrderBook,
        tok: Message_Tokenizer,
        v: Vocab,
        new_order_id: int,
        tick_size: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:
    """"""
    # decoded predicted message
    pred_msg = tok.decode(pred_msg_enc, v).flatten()
    #print('decoded predicted message:', pred_msg)

    if onp.isnan(pred_msg).all():
        return None, None, None
    
    orig_part = pred_msg[: len(pred_msg) // 2]
    modif_part = pred_msg[len(pred_msg) // 2:]

    # new order: no modification values present (all NA)
    # should be new LIMIT ORDER (1)
    if onp.isnan(modif_part).all():
        order_dict, msg_corr, raw_dict = get_sim_msg_new(sim, orig_part, new_order_id, tick_size)

    # modification / deletion / execution of existing order
    else:
        # error in msg: some modifier field is nan
        if onp.isnan(modif_part).any():
            return None, None, None

        mod_type = int(modif_part[1])
        # cancel / delete
        if mod_type in {2, 3}:
            order_dict, msg_corr, raw_dict = get_sim_msg_mod(
                pred_msg_enc,
                orig_part,
                modif_part,
                m_seq,
                m_seq_raw,
                sim,
                tok,
                v,
                tick_size)

        # modify
        elif mod_type == 4:
            order_dict, msg_corr, raw_dict = get_sim_msg_exec(
                pred_msg_enc,
                orig_part,
                modif_part,
                m_seq,
                m_seq_raw,
                new_order_id,
                sim,
                tok,
                v,
                tick_size)

        # Invalid type of modification
        else:
            return None, None, None

    return order_dict, msg_corr, raw_dict


def get_sim_msg_new(
        sim: OrderBook,
        orig_part: onp.ndarray,
        new_order_id: int,
        tick_size: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:

        event_type = int(orig_part[1])
        quantity = int(orig_part[2])
        side = int(orig_part[4])

        if onp.isnan(orig_part).any():
            return None, None, None
        
        assert event_type == 1
        # new limit order
        # if event_type == 1:
        print('NEW LIMIT ORDER')
        rel_price = int(orig_part[3])
        # convert relative to absolute price
        price = sim.get_best_bid() + rel_price * tick_size
            
        # type 4: order execution
        # else:
        #     print('ORDER EXECUTION')
        #     # make sure execution happens at best bid/ask
        #     # adjust price and quantity (down) accordingly to guarantee execution at only one price level
        #     #price = sim.get_best_ask() if side == 0 else sim.get_best_bid()
        #     if side == 0:
        #         price = sim.get_best_ask()
        #         rel_price = (price - sim.get_best_bid()) // tick_size
        #     else:
        #         price = sim.get_best_bid()
        #         rel_price = 0
        #     available_qty = sim.get_volume_at_price(side, price)
        #     quantity = min(quantity, available_qty)

        # TODO: validate timestamp (should be in the future but not too far)
        order_dict = {
            'timestamp': str(orig_part[0] * 1e-9 + 9.5 * 3600),
            'type': 'limit',
            'order_id': new_order_id, 
            'quantity': quantity,
            'price': price,
            'side': 'ask' if side == 0 else 'bid',  # TODO: should be 'buy' or 'sell'?
            'trade_id': 0  # should be trader_id in future
        }

        # msg format to update raw data
        raw_dict = sim_order_to_raw(order_dict, event_type)

        msg_corr = onp.array([
            str(int(orig_part[0])).zfill(15),
            str(event_type),
            str(quantity).zfill(4), 
            ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
            str(side),
        ])
        # encode corrected message
        # TODO: make this an arg?
        tok = Message_Tokenizer()
        v = Vocab()
        msg_corr = tok.encode_msg(msg_corr, v)
        nan_part = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])
        msg_corr = onp.concatenate([msg_corr, nan_part])

        return order_dict, msg_corr, raw_dict


def rel_to_abs_price(
        p_rel: int,
        sim: OrderBook,
        tick_size: int = 100,
    ) -> int:

    assert tick_size % 10 == 0
    round_to = int(-onp.log10(tick_size))
    p_ref = onp.round((sim.get_best_bid() + sim.get_best_ask()) / 2, round_to)
    return p_ref + p_rel * tick_size


def get_sim_msg_mod(
        pred_msg_enc: onp.ndarray,
        orig_part: onp.ndarray,
        modif_part: onp.ndarray,
        m_seq: onp.ndarray,
        m_seq_raw: pd.DataFrame,
        sim: OrderBook,
        tok: Message_Tokenizer,
        v: Vocab,
        tick_size: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:

    print('ORDER CANCEL / DELETE')

    # the actual price of the order to be modified
    # p_mod_raw = sim.get_best_bid() + int(modif_part[3]) * tick_size
    p_mod_raw = rel_to_abs_price(int(modif_part[3]), sim, tick_size)
    side = int(modif_part[4])
    removed_quantity = int(modif_part[2])
    event_type = int(modif_part[1])

    print('rel price', int(modif_part[3]))
    print('side', side)
    print('removed_quantity (raw)', removed_quantity)
    print('total liquidity at price', sim.get_volume_at_price(side, p_mod_raw))
    print('event_type:', event_type)

    assert event_type != 4, 'event_type 4 should be handled separately'

    # # make sure execution happens only if price generated at best bid/ask
    # if event_type == 4:
    #     best_price = sim.get_best_ask() if side == 0 else sim.get_best_bid()
    #     if p_mod_raw != best_price:
    #         print('EXECUTION AT WRONG PRICE')
    #         return None, None
    
    # orig order before sequence start (no ref given or part missing)
    if onp.isnan(orig_part).any():
        # if no init volume remains at price, discard current message
        if sim.get_init_volume_at_price(side, p_mod_raw) == 0:
            return None, None, None
        order_id = job.INITID
        orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])
    
    # search for original order to get correct ID
    else:
        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        # original part is only needed to match to an order ID
        # find original msg index location in the sequence (if it exists)
        orig_enc = pred_msg_enc[: len(pred_msg_enc) // 2]

        mask = get_invalid_ref_mask(m_seq_raw, p_mod_raw, sim)
        orig_i, n_fields_removed = valh.try_find_msg(orig_enc, m_seq, seq_mask=mask)
        #orig_i = valh.find_orig_msg(orig_enc, m_seq)
        
        # didn't find matching original message
        if orig_i is None:
            if sim.get_init_volume_at_price(side, p_mod_raw) == 0:
                return None, None, None
            order_id = job.INITID
            orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])
        
        # found matching original message
        else:
            # get order ID from raw data for simulator
            order_id = int(m_seq_raw.iloc[orig_i].order_id)
            orig_msg_found = onp.array(m_seq[orig_i, : Message_Tokenizer.MSG_LEN // 2])

    # get remaining quantity in book for given order ID
    print('looking for order', order_id, 'at price', p_mod_raw)
    remaining_quantity = sim.get_order_by_id_and_price(order_id, p_mod_raw)[0]
    #if order_id != job.INITID:
    #    assert sim.get_order_by_id(order_id)[0] == remaining_quantity, \
    #        f'order_id: {order_id}, remaining_quantity: {remaining_quantity}'
    # --> could be that p_mod_raw is wrong, hence it finds -1 volume
    #    --> find different order to modify (price beats other fields)
    print('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        remaining_quantity = sim.get_init_volume_at_price(side, p_mod_raw)
        print('remaining init qu.', remaining_quantity)
        # if no init volume remains at price, discard current message
        if remaining_quantity == 0:
            return None, None, None
        order_id = job.INITID
        orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity
        # change partial cancel to full delete
        if event_type == 2:
            event_type = 3
    # change full delete to partial cancel
    elif event_type == 3:
        event_type = 2

    print(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')

    if event_type == 2:
        sim_type = 'cancel'
        sim_side = side
    elif event_type == 3:
        sim_type = 'delete'
        sim_side = side
    # elif event_type == 4:
    #     sim_type = 'limit'
    #     # convert execution side to limit order side for simulator
    #     sim_side = 0 if side == 1 else 1
    else:
        raise ValueError('Invalid event type')
    
    order_dict = {
        # format decimals to nanosecond precision
        'timestamp': format(modif_part[0] * 1e-9 + 9.5 * 3600, '.9f'),
        'type': sim_type,
        'order_id': order_id, 
        'quantity': removed_quantity,
        'price': p_mod_raw,
        'side': 'ask' if sim_side == 0 else 'bid',  # TODO: should be 'buy' or 'sell'
        'trade_id': 0  # should be trader_id in future
    }
    # msg format to update raw data
    raw_dict = sim_order_to_raw(order_dict, event_type)
    rel_price = int(modif_part[3])
    #corr_msg = onp.empty((len(pred_msg),))
    #corr_msg[: len(pred_msg) // 2] = orig_msg_found
    msg_corr = onp.array([
        str(int(modif_part[0])).zfill(15),
        str(event_type),
        str(removed_quantity).zfill(4), 
        ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
        str(side),
    ])
    # encode corrected message
    msg_corr = tok.encode_msg(msg_corr, v)
    msg_corr = onp.concatenate([orig_msg_found, msg_corr])

    return order_dict, msg_corr, raw_dict


def get_sim_msg_exec(
        pred_msg_enc: onp.ndarray,
        orig_part: onp.ndarray,
        modif_part: onp.ndarray,
        m_seq: onp.ndarray,
        m_seq_raw: pd.DataFrame,
        new_order_id: int,
        sim: OrderBook,
        tok: Message_Tokenizer,
        v: Vocab,
        tick_size: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[onp.ndarray], Optional[Dict[str, Any]]]:

    print('ORDER EXECUTION')

    # the actual price of the order to be modified
    p_mod_raw = rel_to_abs_price(int(modif_part[3]), sim, tick_size)
    side = int(modif_part[4])
    removed_quantity = int(modif_part[2])
    event_type = int(modif_part[1])

    print('event_type:', event_type)
    assert event_type == 4
    print('side:', side)
    print('removed_quantity:', removed_quantity)

    # get order against which execution is happening
    passive_order = sim.orderbook_array[side, 0, 0]
    if side == 0:
        print('   execution on ask side (buyer initiated)')
        print('   best ask:', passive_order[1])
    else:
        print('   execution on bid side (seller initiated)')
        print('   best bid:', passive_order[1])
    if p_mod_raw != passive_order[1]:
        print('EXECUTION AT WRONG PRICE', 'gen:', p_mod_raw, 'p_passive', passive_order[1])
        return None, None, None

    remaining_quantity = passive_order[0]
    print('remaining quantity', remaining_quantity)
    if remaining_quantity == -1:
        print('NOTHING TO EXECUTE AGAINST (empty side of book)')
        return None, None, None

    # removing more than remaining quantity --> scale down to remaining
    if removed_quantity >= remaining_quantity:
        removed_quantity = remaining_quantity

    print(f'(event_type={event_type}) -{removed_quantity} from {remaining_quantity} '
          + f'@{p_mod_raw} --> {remaining_quantity-removed_quantity}')
    
    order_dict = {
        # format decimals to nanosecond precision
        'timestamp': format(modif_part[0] * 1e-9 + 9.5 * 3600, '.9f'),
        'type': 'limit',
        'order_id': new_order_id, 
        'quantity': removed_quantity,
        'price': p_mod_raw,
        'side': 'ask' if side == 1 else 'bid',  # CAVE: other side as cancels
        'trade_id': 0  # should be trader_id in future
    }
    
    # msg format to update raw data, CAVE: execution is opposite side from limit order
    raw_dict = sim_order_to_raw({**order_dict, 'side': 'ask' if side == 0 else 'bid'}, event_type)
    rel_price = int(modif_part[3])
    msg_corr = onp.array([
        str(int(modif_part[0])).zfill(15),
        str(event_type),
        str(removed_quantity).zfill(4), 
        ('+' if rel_price > 0 else '-') + str(onp.abs(rel_price)).zfill(2),
        str(side),
    ])

    # correct the order which is executed in the sequence
    order_id = passive_order[3]
    orig_i = onp.argwhere(m_seq_raw.order_id.values == order_id)
    # found correct order
    if len(orig_i) > 0:
        m_seq = m_seq.copy().reshape((-1, Message_Tokenizer.MSG_LEN))
        #print('orig_i', orig_i)
        #print('m_seq[orig_i]', m_seq[orig_i])
        orig_i = orig_i.flatten()[0]
        orig_msg_found = onp.array(m_seq[orig_i, : Message_Tokenizer.MSG_LEN // 2])
    # didn't find correct order (e.g. INITID)
    else:
        orig_msg_found = onp.array((Message_Tokenizer.MSG_LEN // 2) * [Vocab.NA_TOK])

    # encode corrected message
    msg_corr = tok.encode_msg(msg_corr, v)
    msg_corr = onp.concatenate([orig_msg_found, msg_corr])

    return order_dict, msg_corr, raw_dict


def get_invalid_ref_mask(
        m_seq_raw: pd.DataFrame,
        p_mod_raw: int,
        sim: OrderBook
    ):

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
        'time': order_dict['timestamp'],
        'event_type': event_type,
        'order_id': order_dict['order_id'],
        'size': order_dict['quantity'],
        'price': order_dict['price'],
        'direction': order_dict['side'],
    }


def generate(
        m_seq: jax.Array,
        b_seq: jax.Array,
        m_seq_raw: pd.DataFrame,
        n_msg_todo: int,
        sim: OrderBook,
        train_state: TrainState,
        model: nn.Module,
        batchnorm: bool,
        rng: jax.random.PRNGKeyArray,
        sample_top_n: int = 50,
        tick_size: int = 100,
    ) -> Tuple[jax.Array, jax.Array, pd.DataFrame, jax.Array]:

    id_gen = OrderIdGenerator()
    l = Message_Tokenizer.MSG_LEN
    v = Vocab()
    tok = Message_Tokenizer()
    vocab_len = len(v)
    valid_mask_array = valh.syntax_validation_matrix(v)
    l2_book_states = []
    m_seq_raw = m_seq_raw.copy()

    while n_msg_todo > 0:
        rng, rng_ = jax.random.split(rng)
        m_seq = valh.append_hid_msg(m_seq)

        # get next message: generate l tokens
        for mask_i in range(l):
            # syntactically valid tokens for current message position
            valid_mask = valid_mask_array[mask_i]

            m_seq, _ = valh.mask_last_msg_in_seq(m_seq, mask_i)
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
            logits = valh.predict(
                input,
                integration_timesteps, train_state, model, batchnorm)
            
            # filter out (syntactically) invalid tokens for current position
            if valid_mask is not None:
               logits = valh.filter_valid_pred(logits, valid_mask)

            # update sequence
            # note: rng arg expects one element per batch element
            rng, rng_ = jax.random.split(rng)
            m_seq = valh.fill_predicted_toks(m_seq, logits, sample_top_n, jnp.array([rng_]))

        ### process generated message

        #m_seq_raw = m.iloc[end_i - n_messages : end_i]
        #m_seq_raw = m_raw.iloc[end_i - n_messages + 1 : end_i]
        order_id = id_gen.step()

        # parse generated message for simulator, also getting corrected raw message
        # (needs to be encoded and overwrite originally generated message)
        sim_msg, msg_corr, msg_raw = get_sim_msg(
            m_seq[-l:],  # the generated message
            m_seq[:-l],  # sequence without generated message
            m_seq_raw.iloc[1:],   # raw data (same length as sequence without generated message)
            sim,
            tok,
            v,
            new_order_id=order_id,
            tick_size=tick_size
        )

        if sim_msg is None:
            print('invalid message - discarding...')
            print()
            # cut away generated message and pad begginning of sequence
            m_seq = onp.concatenate([
                onp.full((l,), Vocab.NA_TOK),
                m_seq[: -l]])
            continue

        print(sim_msg)
        #print('quantitity', sim_msg['quantity'], 'price', sim_msg['price'])

        # replace faulty message in sequence with corrected message
        #print('before', m_seq[-l:])
        #print('after', msg_corr)
        m_seq = m_seq.at[-l:].set(msg_corr)
        # append new message to raw data
        m_seq_raw = m_seq_raw.iloc[1:].append(msg_raw, ignore_index=True)
        print('len(m_seq_raw)', len(m_seq_raw))
        print('new raw msg', m_seq_raw.iloc[-1])

        p_mid_old = onp.round((sim.get_best_ask() + sim.get_best_bid()) / 2, -2).astype(int)
        # feed message to simulator, updating book state
        _trades = sim.process_order(sim_msg)
        p_mid_new = onp.round((sim.get_best_ask() + sim.get_best_bid()) / 2, -2).astype(int)
        # p_change = (p_bid_new - p_bid_old) // tick_size
        p_change = (p_mid_new - p_mid_old) // tick_size

        # get new book state
        book = sim.get_L2_state()
        l2_book_states.append(book)

        new_book_raw = jnp.concatenate([jnp.array([p_change]), book]).reshape(1,-1)
        new_book = preproc.transform_L2_state(new_book_raw, 500, 100)
        #print('new_book', new_book.shape, new_book)
        
        # update book sequence
        b_seq = jnp.concatenate([b_seq[1:], new_book])

        print('p_change', p_change)
        print()

        n_msg_todo -= 1

    return m_seq, b_seq, m_seq_raw, jnp.array(l2_book_states)