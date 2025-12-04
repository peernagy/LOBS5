from __future__ import annotations
import jax
import jax.numpy as jnp
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from decimal import Decimal
from functools import partial
# import lob.encoding as encoding

from lob.encoding import Vocab, Message_Tokenizer


@partial(jax.jit, static_argnums=(1, 2),backend='cpu')
@partial(
    jax.vmap,
    in_axes=(0, None, None),
    out_axes=0,
)
def transform_L2_state(
        book: jax.Array, 
        price_levels: int,
        tick_size: int = 100,
        #divide_by: int = 1,
    ) -> jax.Array:
    """ Transformation for data loading:
        Converts L2 book state from data to price_levels many volume
        series used as input to the model. The first element (column) of the
        input and output is the change in mid price.
        Converts sizes to negative sizes for ask side (sell orders).
    """
    delta_p_mid_and_time, book = book[:3], book[3:]
    book = book.reshape((-1,2))
    mid_price = jnp.ceil((book[0, 0] + book[1, 0]) / (2*tick_size)).__mul__(tick_size).astype(int)
    book = book.at[:, 0].set((book[:, 0] - mid_price) // tick_size)
    # change relative prices to indices
    book = book.at[:, 0].set(book[:, 0] + price_levels // 2)
    # set to out of bounds index, so that we can use -1 to indicate nan
    # out of bounds will be ignored in setting value in jax
    book = jnp.where(book < 0, -price_levels-1, book)

    mybook = jnp.zeros(price_levels, dtype=jnp.int32)
    mybook = mybook.at[book[:, 0]].set(book[:, 1])

    # Norm seconds to be in [0,1] representing percent of day. 
    delta_p_mid_and_time.at[1].set((delta_p_mid_and_time[1]-34200)/23400)
    # Norm nanoseconds to be fraction of a second
    delta_p_mid_and_time.at[2].set(delta_p_mid_and_time[2]/1e9)

    
    # set ask volume to negative (sell orders)
    mybook = mybook.at[price_levels // 2:].set(mybook[price_levels // 2:] * -1)
    mybook = jnp.concatenate((
        delta_p_mid_and_time.astype(jnp.float32),
        mybook.astype(jnp.float32) / 1000
    ))

    # return mybook.astype(jnp.float32) #/ divide_by
    return mybook 

@partial(jax.jit, static_argnums=(1, 2),backend='gpu')
@partial(
    jax.vmap,
    in_axes=(0, None, None),
    out_axes=0,
)
def transform_L2_state_gpu(
        book: jax.Array, 
        price_levels: int,
        tick_size: int = 100,
        #divide_by: int = 1,
    ) -> jax.Array:
    """ Transformation for data loading:
        Converts L2 book state from data to price_levels many volume
        series used as input to the model. The first element (column) of the
        input and output is the change in mid price.
        Converts sizes to negative sizes for ask side (sell orders).
    """
    delta_p_mid_and_time, book = book[:3], book[3:]
    book = book.reshape((-1,2))
    mid_price = jnp.ceil((book[0, 0] + book[1, 0]) / (2*tick_size)).__mul__(tick_size).astype(int)
    book = book.at[:, 0].set((book[:, 0] - mid_price) // tick_size)
    # change relative prices to indices
    book = book.at[:, 0].set(book[:, 0] + price_levels // 2)
    # set to out of bounds index, so that we can use -1 to indicate nan
    # out of bounds will be ignored in setting value in jax
    book = jnp.where(book < 0, -price_levels-1, book)

    mybook = jnp.zeros(price_levels, dtype=jnp.int32)
    mybook = mybook.at[book[:, 0]].set(book[:, 1])
    
    # Norm seconds to be in [0,1] representing percent of day. 
    delta_p_mid_and_time=delta_p_mid_and_time.astype(jnp.float32)
    delta_p_mid_and_time=delta_p_mid_and_time.at[1].set((delta_p_mid_and_time[1]-34200)/23400)
    # Norm nanoseconds to be fraction of a second
    delta_p_mid_and_time=delta_p_mid_and_time.at[2].set(delta_p_mid_and_time[2]/1e9)

    # set ask volume to negative (sell orders)
    mybook = mybook.at[price_levels // 2:].set(mybook[price_levels // 2:] * -1)
    mybook = jnp.concatenate((
        delta_p_mid_and_time.astype(jnp.float32),
        mybook.astype(jnp.float32) / 1000
    ))

    # return mybook.astype(jnp.float32) #/ divide_by
    return mybook


@partial(np.vectorize,signature="(c),(),()->(d)")
def transform_L2_state_numpy(
        book: np.ndarray, 
        price_levels: int,
        tick_size: int = 100,
        #divide_by: int = 1,
    ) -> np.ndarray:
    """ Transformation for data loading:
        Converts L2 book state from data to price_levels many volume
        series used as input to the model. The first element (column) of the
        input and output is the change in mid price.
        Converts sizes to negative sizes for ask side (sell orders).
    """
    delta_p_mid_and_time, book = book[:3], book[3:]
    book = book.reshape((-1,2))
    # print(book)
    mid_price = np.ceil((book[0, 0] + book[1, 0]) / (2*tick_size)).__mul__(tick_size).astype(int)
    book=book.copy()
    delta_p_mid_and_time=delta_p_mid_and_time.copy()
    book[:, 0]=((book[:, 0] - mid_price) // tick_size)
    # print(book)
    # change relative prices to indices
    book[:, 0]=(book[:, 0] + price_levels // 2)
    # set to out of bounds index, so that we can use -1 to indicate nan
    # out of bounds will be ignored in setting value in jax
    # print(book)
    book_ind = book[(book[:,0] >= 0)&(book[:,0] < 500)]
    # print(book_ind)

    mybook = np.zeros(price_levels, dtype=np.int32)
    # print(book_ind[:, 0])
    # print(book_ind[:, 1])
    # mybook[book_ind[:, 0]]=(book_ind[:, 1])
    mybook[book_ind[:, 0].astype(int)] = (book_ind[:, 1])
    
    delta_p_mid_and_time=delta_p_mid_and_time.astype(np.float32)
    # Norm seconds to be in [0,1] representing percent of day. 
    delta_p_mid_and_time[1]=((delta_p_mid_and_time[1]-34200)/23400)
    # Norm nanoseconds to be fraction of a second
    delta_p_mid_and_time[2]=(delta_p_mid_and_time[2]/1e9)


    # set ask volume to negative (sell orders)
    mybook[price_levels // 2:]=(mybook[price_levels // 2:] * -1)
    mybook = np.concatenate((
        delta_p_mid_and_time,
        mybook.astype(np.float32) / 1000
    ))

    # return mybook.astype(jnp.float32) #/ divide_by
    return mybook 





def load_message_df(m_f: str) -> pd.DataFrame:
    cols = ['time', 'event_type', 'order_id', 'size', 'price', 'direction']
    messages = pd.read_csv(
        m_f,
        names=cols,
        usecols=cols,
        index_col=False,
        dtype={
            'time': str,
            'event_type': 'int32',
            'order_id': 'int32',
            'size': 'int32',
            'price': 'int32',
            'direction': 'int32'
        }
    )
    messages.time = messages.time.apply(lambda x: Decimal(x))
    return messages






def process_message_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        filter_above_lvl: Optional[int] = None,
        skip_existing: bool = False,
    ) -> None:

    v = Vocab()
    tok = Message_Tokenizer()

    assert len(message_files) == len(book_files)
    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        m_path = save_dir + m_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        if skip_existing and Path(m_path).exists():
            print('skipping', m_path)
            continue
        
        
        messages = load_message_df(m_f)
        try:
            book = pd.read_csv(
            b_f,
            index_col=False,
            header=None
        )
        except pd.errors.EmptyDataError:
            print(b_f, " is empty")
            continue

        assert len(messages) == len(book), f'messages: {m_f} orderbook: {b_f} ;{len(messages)} != {len(book)}'

        if filter_above_lvl:
            book = book.iloc[:, :filter_above_lvl * 4]
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)
        
        print('<< pre processing >>')
        messages = messages.loc[(messages.time>=Decimal(34200)) & (messages.time<Decimal(57600))]
        book = book.loc[messages.index]
        m_ = tok.preproc(messages, book)

        # save processed messages
        np.save(m_path, m_)
        print('saved to', m_path)

def get_price_range_for_level(
        book: pd.DataFrame,
        lvl: int
    ) -> pd.DataFrame:
    assert lvl > 0
    assert lvl <= (book.shape[1] // 4)
    p_range = book[[(lvl-1) * 4, (lvl-1) * 4 + 2]]
    p_range.columns = ['p_max', 'p_min']
    return p_range

def filter_by_lvl(
        messages: pd.DataFrame,
        book: pd.DataFrame,
        lvl: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    assert messages.shape[0] == book.shape[0]
    p_range = get_price_range_for_level(book, lvl)
    messages = messages[(messages.price <= p_range.p_max) & (messages.price >= p_range.p_min)]
    book = book.loc[messages.index]
    return messages, book


def process_book_files(
        message_files: list[str],
        book_files: list[str],
        save_dir: str,
        n_price_series: int,
        filter_above_lvl: Optional[int] = None,
        allowed_events=[1, 2, 3, 4],
        skip_existing: bool = False,
        use_raw_book_repr=False,
    ) -> None:

    for m_f, b_f in tqdm(zip(message_files, book_files)):
        print(m_f)
        print(b_f)
        b_path = save_dir + b_f.rsplit('/', maxsplit=1)[-1][:-4] + '_proc.npy'
        if skip_existing and Path(b_path).exists():
            print('skipping', b_path)
            continue

        messages = load_message_df(m_f)

        try:
            book = pd.read_csv(
            b_f,
            index_col=False,
            header=None
        )
        except pd.errors.EmptyDataError:
            print(b_f, " is empty")
            continue

        # remove disallowed order types
        messages = messages.loc[messages.event_type.isin(allowed_events)]
        messages = messages.loc[(messages.time>=Decimal(34200)) & (messages.time<Decimal(57600))]
        # make sure book is same length as messages
        book = book.loc[messages.index]

        if filter_above_lvl is not None:
            messages, book = filter_by_lvl(messages, book, filter_above_lvl)



        # convert to n_price_series separate volume time series (each tick is a price level)
        # NOTE: this conversion can now be done fast in the data loader, so we can skip this step
        if use_raw_book_repr:
            book=augment_book_state(book,messages)
        else:
            book = process_book(book,messages, price_levels=n_price_series)

        np.save(b_path, book, allow_pickle=True)


def augment_book_state(book: pd.DataFrame,
                       message: pd.DataFrame):
    best_ask = book.iloc[:, 0].replace({9999999999: np.nan}).ffill().astype(int)
    best_bid = book.iloc[:, 2].replace({-9999999999: np.nan}).ffill().astype(int)
    p_ref = ((best_ask + best_bid) / 2).round(-2).astype(int)
    mid_diff = p_ref.div(100).diff().fillna(0).astype(int)

    message.insert(0, 'time_s', message.time.astype(int))
    message.rename(columns={'time': 'time_ns'}, inplace=True)
    message.time_ns = ((message.time_ns % 1) * 1000000000).astype(int)

    times=message[['time_s','time_ns']]
    times=times.values.reshape(-1,2)
    # prepend delta mid price column to book data
    book = np.concatenate((mid_diff.values.reshape(-1,1),times, book.values), axis=1)
    return book

def process_book(
        b: pd.DataFrame,
        m: pd.DataFrame,
        price_levels: int
    ) -> np.ndarray:

    # mid-price rounded to nearest tick (100)
    best_ask = b.iloc[:, 0].replace({9999999999: np.nan}).ffill().astype(int)
    best_bid = b.iloc[:, 2].replace({-9999999999: np.nan}).ffill().astype(int)
    p_ref = ((best_ask + best_bid) / 2).round(-2).astype(int)
    b_indices = b.iloc[:, ::2].sub(p_ref, axis=0).div(100).astype(int)
    b_indices = b_indices + price_levels // 2
    b_indices.columns = list(range(b_indices.shape[1]))
    vol_book = b.iloc[:, 1::2].copy()
    # convert sell volumes (ask side) to negative
    vol_book.iloc[:, ::2] = vol_book.iloc[:, ::2].mul(-1)
    vol_book.columns = list(range(vol_book.shape[1]))

    # convert to book representation with volume at each price level relative to reference price (mid)
    # i.e. at each time we have a fixed width snapshot around the mid price
    # therefore movement of the mid price needs to be a separate feature (e.g. relative to previous price)

    mybook = np.zeros((len(b), price_levels), dtype=np.int32)

    a = b_indices.values
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            price = a[i, j]
            if price >= 0 and price < price_levels:
                mybook[i, price] = vol_book.values[i, j]

    # prepend column with reference price changes (in ticks)
    mid_diff = p_ref.div(100).diff().fillna(0).astype(int).values

    m.insert(0, 'time_s', m.time.astype(int))
    m.rename(columns={'time': 'time_ns'}, inplace=True)
    m.time_ns = ((m.time_ns % 1) * 1000000000).astype(int)

    times=m[['time_s','time_ns']]
    times=times.values.reshape(-1,2)

    return np.concatenate([mid_diff[:, None],times, mybook], axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/homes/80/kang/GOOG_2017/',
		     			help="where to load data from")
    parser.add_argument("--save_dir", type=str, default='/homes/80/kang/LOBS5/proced_data/GOOG20172018/',
		     			help="where to save processed data")
    parser.add_argument("--filter_above_lvl", type=int,
                        help="filters down from levels present in the data to specified number of price levels")
    parser.add_argument("--n_tick_range", type=int, default=500,
                        help="how many ticks price series should be calculated")
    parser.add_argument("--skip_existing", action='store_true', default=False)
    parser.add_argument("--messages_only", action='store_true', default=False)
    parser.add_argument("--book_only", action='store_true', default=False)
    parser.add_argument("--use_raw_book_repr", action='store_true', default=True)
    args = parser.parse_args()

    assert not (args.messages_only and args.book_only)

    message_files = sorted(glob(args.data_dir + '*message*.csv'))
    book_files = sorted(glob(args.data_dir + '*orderbook*.csv'))

    print('found', len(message_files), 'message files')
    print('found', len(book_files), 'book files')
    print()

    if not args.book_only:
        print('processing messages...')
        process_message_files(
            message_files,
            book_files,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            skip_existing=args.skip_existing,
        )
    else:
        print('Skipping message processing...')
    print()
    
    if not args.messages_only:
        print('processing books...')
        process_book_files(
            message_files,
            book_files,
            args.save_dir,
            filter_above_lvl=args.filter_above_lvl,
            n_price_series=args.n_tick_range,
            skip_existing=args.skip_existing,
            use_raw_book_repr=args.use_raw_book_repr,
        )
    else:
        print('Skipping book processing...')
    print('DONE')
