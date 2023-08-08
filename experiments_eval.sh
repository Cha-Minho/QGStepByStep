experiment="hotpot_comp1" #choose from: baseline hotpot_sub hotpot_comp


CUDA_VISIBLE_DEVICES=1 python QG_eval.py \
    --output_prefix /home/team5/QGStepByStep/Datasets/output/QG/generated/$experiment/ \
    --filename test.0.1000.generated.gpt2.json.copy.json\
    --eval_result_output /home/team5/QGStepByStep/Datasets/output/QG/generated/$experiment/output.txt

