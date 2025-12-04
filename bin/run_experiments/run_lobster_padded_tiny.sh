python run_train.py --USE_WANDB=True\
                    --C_init=trunc_standard_normal --prenorm=True --batchnorm=True --bidirectional=False \
                    --blocks=4 --bsz=4 --d_model=64 --dataset=lobster-prediction --merging=padded\
                    --dir_name='./data/' --clip_eigs=True --activation_fn=half_glu1 \
                    --dt_global=False --epochs=2 --jax_seed=42 --lr_factor=1 --n_layers=3 \
                    --opt_config=standard --p_dropout=0.0 --ssm_lr_base=0.0005 --ssm_size_base=256 \
                    --warmup_end=1 --weight_decay=0.05 --msg_seq_len=500 \
                    --use_book_data=True --use_simple_book=False --book_transform=True \
                    --masking=none \
                    --num_devices=6 --n_data_workers=6 \
                    --debug_loading=False \
                    --enable_profiler=True \
                    #--restore='/data1/sascha/LOBS5/checkpoints/rural-serenity-60_uxc2c11x/' #\
                    #--restore='checkpoints/eager-shadow-750_af39bb9u/'


