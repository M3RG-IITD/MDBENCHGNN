python3 /home/civil/btech/ce1180169/MDBENCHGNN/mdbenchgnn/models/mace/scripts/run_train.py \
  --name="BOTNet_model_md17_salicylic_acid" \
  --log_dir="/home/civil/btech/ce1180169/MDBENCHGNN/output_dir_sl/botnet/md17/salicylic_acid/logs" \
  --model_dir="/home/civil/btech/ce1180169/MDBENCHGNN/output_dir_sl/botnet/md17/salicylic_acid" \
  --train_file='/home/civil/btech/ce1180169/MDBENCHGNN/example/md17/salicylic_acid/train/botnet.xyz' \
  --valid_file='/home/civil/btech/ce1180169/MDBENCHGNN/example/md17/salicylic_acid/val/botnet.xyz' \
  --results_dir="/home/civil/btech/ce1180169/MDBENCHGNN/output_dir_sl/botnet/md17/salicylic_acid/results" \
  --checkpoints_dir="/home/civil/btech/ce1180169/MDBENCHGNN/output_dir_sl/botnet/md17/salicylic_acid/checkpoints" \
  --E0s='{1:-13.663181292231226, 3:-216.78673811801755, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639, 15:-1537.0898574856286, 16:-1867.8202267974733}' \
  --model="ScaleShiftMACE" \
  --hidden_irreps='16x0e+16x1o+16x2e ' \
  --r_max=5.0 \
  --correlation=1 \
  --num_interactions=4 \
  --batch_size=1 \
  --valid_batch_size=1 \
  --max_num_epochs=1000 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cuda \
  --seed=123 \
  --swa
