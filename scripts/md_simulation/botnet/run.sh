#!/bin/bash
# activate conda neq_alg
# Edit the following variables as needed
MODEL_PATH="example/lips/botnet/BOTNet_model_lips_50_swa.model"
INIT_CONF_PATH="example/lips20/data/test/botnet.xyz"
OUT_PATH="./"
TEMP=520


# Run the Python command
python scripts/md_simulation/botnet/md_bot.py \
  --model_path $MODEL_PATH \
  --init_conf_path $INIT_CONF_PATH \
  --out_path $OUT_PATH \
  --temp $TEMP \

