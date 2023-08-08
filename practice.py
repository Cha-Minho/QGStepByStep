from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("philippelaban/keep_it_simple")
kis_model = AutoModelForCausalLM.from_pretrained("philippelaban/keep_it_simple")

paragraph = [
                    "Fat Possum Records is an American independent record label based in Water Valley and Oxford, Mississippi.",
                    " At first Fat Possum focused almost entirely on recording previously unknown Mississippi blues artists (typically from Oxford or Holly Springs, Mississippi).",
                    " Recently, Fat Possum has signed younger rock acts to its roster.",
                    " The label has been featured in \"The New York Times\", \"New Yorker\", \"The Observer\", a Sundance Channel production, a piece on NPR, and a 2004 documentary, \"You See Me Laughin\".",
                    " Fat Possum also distributes the Hi Records catalog."
            ]
new = ""
for string in paragraph:
    new += string
paragraph = new
start_id = tokenizer.bos_token_id
tokenized_paragraph = [(tokenizer.encode(text=paragraph) + [start_id])]
input_ids = torch.LongTensor(tokenized_paragraph)

output_ids = kis_model.generate(input_ids, max_length=1231, num_beams=4, do_sample=True, num_return_sequences=8)
output_ids = output_ids[:, input_ids.shape[1]:]
output = tokenizer.batch_decode(output_ids)
output = [o.replace(tokenizer.eos_token, "") for o in output]
output = output[0]
output = output.split(".")
if output[-1] == '':
    output.pop(-1)
new_list = []
for strings in output:
    new_list.append(strings.strip() + ".")
print(new_list)
