example:
[make sure to activate relevant conda environment]
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1

nequip-train configs/allegro/lips/lips.yaml

after training:
nequip-deploy build --train-dir path/to/training/session/ where/to/put/deployed_model.pth

