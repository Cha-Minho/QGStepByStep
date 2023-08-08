from tqdm import tqdm
import torch
import json
from transformers import T5ForConditionalGeneration,T5Tokenizer


def paraphrase_sent(sentence, num_return_sequences):
    text =  "paraphrase: " + sentence + " </s>"

    max_len = 256

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=num_return_sequences
    )

    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs


def paraphrase(model, tokenizer, num):
    # gold
    intermediate = []
    with open("/home/team5/QGStepByStep/Datasets/output/QG/generated/hotpot_sub/test.0.1000.generated.gpt2.intermediate.json", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # intermediate = json_data.values()

    intermediate_output = {}
    for key, line in tqdm(json_data.items()):
        new_text = paraphrase_sent(line, num)
        for text in new_text:
            intermediate_output[key] = text



    with open('/home/team5/QGStepByStep/Datasets/output/QG/generated/hotpot_sub/test.0.1000.generated.gpt2.intermediate_paraphrased.json', 'w', encoding='utf-8') as f:
        json.dump(intermediate_output, f, ensure_ascii=False)



if __name__ == "__main__":
    num = 1
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    paraphrase(model, tokenizer, num)
