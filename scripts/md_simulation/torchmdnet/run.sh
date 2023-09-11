#!/bin/bash
# activate conda neq_alg
# Edit the following variables as needed
MODEL_PATH="example/lips/torchmdnet/epoch=129-val_loss=0.0184-test_loss=0.0941.ckpt"
INIT_CONF_PATH="example/lips/data/test/botnet.xyz"
OUT_PATH="./"
TEMP=520
# SYSTEM="lips20"

# Run the Python command
python scripts/md_simulation/torchmdnet/md_tmd.py \
  --model_path "$MODEL_PATH" \
  --init_conf_path $INIT_CONF_PATH \
  --out_dir "$OUT_PATH" \
  --temp $TEMP \

