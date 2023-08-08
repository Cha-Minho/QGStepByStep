experiment=hotpot_comp #choose from: baseline hotpot_sub hotpot_comp
# input_data_folder=baseline #choose from: baseline subq_rel
# output dir need to be empty before running
# empty cache if needed

CUDA_VISIBLE_DEVICES=0 python QG_gpt2_train.py \
    --eval_before_start \
    --n_epochs 10 \
    --model_name_or_path /home/team5/gpt2 \
    --output_dir /home/team5/QGStepByStep/Datasets/output/QG/comp_models \
    --train_dataset_path /home/team5/QGStepByStep/DCQG_data/train.json \
    --dev_dataset_path /home/team5/QGStepByStep/DCQG_data/dev.json \
    --filetype $experiment \
    --train_dataset_cache /home/team5/QGStepByStep/Datasets/output/QG/cache/$experiment/train_hotpot_cache.txt \
    --dev_dataset_cache /home/team5/QGStepByStep/Datasets/output/QG/cache/$experiment/dev_hotpot_cache.txt \


    # --train_dataset_path /home/team5/QGStepByStep/Datasets/original/HotpotQA/$input_data_folder/dev.json \
    # --dev_dataset_path /home/team5/QGStepByStep/Datasets/original/HotpotQA/$input_data_folder/test.json \

