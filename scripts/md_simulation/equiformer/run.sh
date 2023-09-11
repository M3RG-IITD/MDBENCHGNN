#!/bin/bash
# activate conda neq_alg
# Edit the following variables as needed
MODEL_PATH="example/lips/equiformer/best_val_epochs@74_e@0.0833_f@0.0510.pth.tar"
INIT_CONF_PATH="example/lips/data/test/botnet.xyz"
OUT_PATH="./"
TEMP=520


# Run the Python command
python scripts/md_simulation/equiformer/md_eq.py \
  --model_path $MODEL_PATH \
  --init_conf_path $INIT_CONF_PATH \
  --out_dir $OUT_PATH \
  --temp $TEMP \

