experiment="baseline" #choose from: baseline hotpot_sub hotpot_comp
input_data_folder="baseline" #choose from: baseline subq_rel
# empty cache if needed

output_path="/home/team5/QGStepByStep/Datasets/output/QG/generated/$experiment/"
data_file_prefix="test"
st_idx=0
ed_idx=1000
epochs=10

CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path "/home/team5/QGStepByStep/Datasets/output/QG/models/" \
    --filename "/home/team5/QGStepByStep/Datasets/original/HotpotQA/$input_data_folder/${data_file_prefix}.json" \
    --data_type $experiment \
    --output_file "$output_path${data_file_prefix}_epoch_${epochs}_generated_gpt2.json" \
    --filecache "/home/team5/QGStepByStep/Datasets/output/QG/cache/$experiment/${data_file_prefix}_hotpot_cache.txt" \
    --calc_tg_metrics\
    # --model_name  "/home/team5/QGStepByStep/Datasets/output/QG/checkpoint_mymodel_3.pth" \
    # --subq_dir "/home/team5/QGStepByStep/Datasets/output/QG/generated/hotpot_sub/${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.intermediate.json" 
    # --output_file "$output_path${data_file_prefix}_${st_idx}_${ed_idx}_generated_gpt2.json" \
