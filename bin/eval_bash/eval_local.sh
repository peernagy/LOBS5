 python3 run_eval.py --restore="/data1/sascha/data/checkpoints/toasty-pine-206_7lo414ol/" \
                        --dir_name="/data1/sascha/data/GOOG2017to2019/" \
                        --restore_step=9 \
                        --n_data_workers=4 \
                        --bsz=8 \
                        --num_devices=1 \
                        --USE_WANDB=True \
                        --ignore_times=False \
                        # --curtail_epoch=10 \
