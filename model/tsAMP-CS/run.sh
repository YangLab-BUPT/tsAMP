
#!/bin/bash
for train_file in /tsAMP/data/tsAMP-CS/train/*.xlsx; do

            train_filename=$(basename "$train_file" .xlsx)
            #CUDA_VISIBLE_DEVICES=0 python train.py --train_dir "$train_file" --filename_dir "$train_filename1"
            
           
            model_path="/tsAMP/model/tsAMP-CS/strain/${train_filename}.pt"
            output_path="model"
            test_dir="/tsAMP/data/tsAMP-CS/test/${train_filename}.xlsx"
            python predict.py --output_excel "$output_path" --model_path "$model_path" --test_dir "$test_dir"
        
            
done

