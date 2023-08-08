from tqdm import tqdm
import torch
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = AutoModelForSeq2SeqLM.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = AutoTokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

sentence = "Which course should I take to get started in data science?"
# sentence = "What are the ingredients required to bake a perfect cake?"
# sentence = "What is the best possible approach to learn aeronautical engineering?"
# sentence = "Do apples taste better than oranges in general?"

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


    # print ("\nOriginal Question ::")
    # print (sentence)
    # print ("\n")
    # print ("Paraphrased Questions :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent+'\n')
    return final_outputs










candidate_path = '/home/team5/MulQG-main/prediction/MulQG_BFS_checkpoint.pt/candidate.txt'
golden_path = '/home/team5/MulQG-main/prediction/MulQG_BFS_checkpoint.pt/reference.txt'
output_num = 1

new_candidates = []
print("start parphrasing")
with open(candidate_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        new_text = paraphrase_sent(line, output_num)
        new_candidates.append(line)
        for text in new_text:
            new_candidates.extend(new_text)
print("finished paraphrasing")

print("finish paraphrasing")
new_golden = []
with open(golden_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        for i in range(0, output_num+1):
            new_golden.append(line)

with open('/home/team5/MulQG-main/prediction/new_candidate.txt', 'w', encoding='utf-8') as f:
    for sent in tqdm(new_candidates):
        f.write(sent)


with open('/home/team5/MulQG-main/prediction/new_reference.txt', 'w', encoding = 'utf-8') as f:
    for sent in tqdm(new_golden):
        f.write(sent)

