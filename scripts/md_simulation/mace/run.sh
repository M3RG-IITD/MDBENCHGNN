#!/bin/bash
# activate conda neq_alg
# Edit the following variables as needed
MODEL_PATH="example/lips/mace/MACE_model_500_lips_swa.model"
INIT_CONF_PATH="example/lips20/data/test/botnet.xyz"
OUT_PATH="./"
TEMP=520


# Run the Python command
python scripts/md_simulation/mace/md_mace.py \
  --model_path "$MODEL_PATH" \
  --init_conf_path $INIT_CONF_PATH \
  --out_dir "$OUT_PATH" \
  --temp $TEMP \

