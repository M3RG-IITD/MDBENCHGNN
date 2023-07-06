export PYTHONNOUSERSITE=True    # prevent using packages from base


python mdbenchgnn/models/equiformer/main_custom.py \
    --output-dir 'output_dir_sl/equiformer/3bpa300/' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target '3bpa' \
    --data-path 'data_sl/mdsim_data/mace_data_sl/BOTNet-datasets/dataset_3BPA/nequip_format/' \
    --epochs 5000 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80