Specifying training arguments can either be done via a configuration yaml file or through command line arguments directly. Note that if a parameter is present both in the yaml file and the command line, the command line version takes precedence.

GPUs can be selected by setting the CUDA_VISIBLE_DEVICES environment variable. Otherwise, the argument --ngpus can be used to select the number of GPUs to train on (-1, the default, uses all available GPUs or the ones specified in CUDA_VISIBLE_DEVICES). Keep in mind that the GPU ID reported by nvidia-smi might not be the same as the one CUDA_VISIBLE_DEVICES uses.


example:
[make sure to activate relevant conda environment]
export CUDA_VISIBLE_DEVICES=0 
torchmd-train-custom  --conf configs/torchmdnet/lips/lips.yaml 

