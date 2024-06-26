python run_train.py --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=False \
                    --blocks=8 --bsz=16 --d_model=32 --dataset=lobster-prediction \
                    --dir_name='./data' --clip_eigs=True \
                    --dt_global=False --epochs=100 --jax_seed=42 --lr_factor=1 --n_layers=6 \
                    --opt_config=BandCdecay --p_dropout=0.2 --ssm_lr_base=0.0005 --ssm_size_base=32 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --masking=causal \
                    #--restore='checkpoints/avid-firefly-145_9uo36dtm/'
                    --restore='checkpoints/brisk-surf-147_yc8wpyi3/'
