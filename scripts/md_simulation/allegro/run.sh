#!/bin/bash
# activate conda neq_alg
# Edit the following variables as needed
MODEL_PATH="example/lips/allegro/deployed.pth"
INIT_CONF_PATH="example/lips20/data/test/botnet.xyz"
OUT_PATH="./"
TEMP=520


# Run the Python command
python scripts/md_simulation/allegro/md_alg.py \
  --model_path "$MODEL_PATH" \
  --init_conf_path "$INIT_CONF_PATH" \
  --out_path "$OUT_PATH" \
  --temp $TEMP \

