CUDA_VISIBLE_DEVICES=0 python3 -u src/myrun.py \
    --file_dir data/DBP15K/$1 \
    --rate $2 \
    --lr .0005 \
    --epochs 1000 \
    --hidden_units "300,300,300" \
    --check_point 50  \
    --bsize 512 \
    --semi_learn_step 5 \
    --csls \
    --csls_k 3 \
    --seed 42 \
    --tau 0.1 \
    --tau2 4.0 \
    --structure_encoder "gat" \
    --img_dim 400 \
    --attr_dim 400 \
    --w_name \
    --w_char

