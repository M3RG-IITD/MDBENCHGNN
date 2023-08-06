nequip-evaluate --model path/of/deployed/model --dataset-config path/to/yaml/file

# Example: 
nequip-evaluate --model output_dir_sl/nequip/lips20/run1/deployed_model.pth --dataset-config configs/nequip/lips20/lips.yaml --train-dir output_dir_sl/nequip/lips20/run1/ --repeat 10

Ouputs: 

--- Final result: ---
               f_mae =  0.065861
              f_rmse =  0.084963
               e_mae =  19575.826172
             e/N_mae =  235.853287
               f_mae =  0.065861
              f_rmse =  0.084963
               e_mae =  19575.826172
             e/N_mae =  235.853287


P.S explore more flags see: mdbenchgnn/models/nequip/nequip/scripts/evaluate.py
