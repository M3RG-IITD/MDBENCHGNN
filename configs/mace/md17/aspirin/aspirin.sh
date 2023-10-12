python3 mdbenchgnn/models/mace/scripts/run_train.py \
  --name="MACE_model_500_aspirin" \
  --log_dir="output_dir_sl/mace/md17/aspirin/logs" \
  --model_dir="output_dir_sl/mace/md17/aspirin" \
  --results_dir="output_dir_sl/mace/md17/aspirin/results" \
  --checkpoints_dir="output_dir_sl/mace/md17/aspirin/checkpoints" \
  --train_file='example/md17/aspirin/train/botnet.xyz' \
  --valid_file='example/md17/aspirin/val/botnet.xyz' \
  --E0s='{1:-13.663181292231226, 3:-216.78673811801755, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639, 15:-1537.0898574856286, 16:-1867.8202267974733}' \
  --model="ScaleShiftMACE" \
  --hidden_irreps='16x0e+16x1o+16x2e ' \
  --r_max=5.0 \
  --batch_size=20 \
  --valid_batch_size=20 \
  --max_num_epochs=1500 \
  --ema \
  --ema_decay=0.99 \
  --amsgrad \
  --default_dtype="float32" \
  --device=cuda \
  --seed=123 \
  --swa