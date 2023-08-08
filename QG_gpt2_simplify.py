from tqdm import tqdm
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def simplify_sent(kis_model, tokenizer, paragraph):
    kis_model.to('cuda')
    
    # tokenizer.padding_side = 'left'
    start_id = tokenizer.bos_token_id
    tokenized_paragraph = [(tokenizer.encode(text=paragraph) + [start_id])]
    input_ids = torch.cuda.LongTensor(tokenized_paragraph)

    output_ids = kis_model.generate(input_ids, max_length=150, num_beams=4, do_sample=True, num_return_sequences=8)
    output_ids = output_ids[:, input_ids.shape[1]:]
    output = tokenizer.batch_decode(output_ids)
    output = [o.replace(tokenizer.eos_token, "") for o in output]
    # output = output[0].split(".")
    
    return output[0]

def simplify(model, tokenizer):
    # gold
    gold = []
    with open("/home/team5/QGStepByStep/DCQG_data/test.json", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    json_data = json_data[30:40]

    for i in tqdm(range(10)): #for i in tqdm(range(len(json_data))):
        torch.cuda.empty_cache()
        json_data[i]['context'][0][1] = [(simplify_sent(model, tokenizer, str(json_data[i]['context'][0][1])))]
        json_data[i]['context'][1][1] = [(simplify_sent(model, tokenizer, str(json_data[i]['context'][1][1])))]
    
    """for i in tqdm(range(1)):#for i in tqdm(range(len(json_data))):
        l1 = len(json_data[i]['context'][0][1])
        if l1 % 2 == 0:
            lst1 = []
            for j in range(0, l1, 2):
                lst1.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][0][1][j:j+2]))))
            json_data[i]['context'][0][1] = lst1
        elif l1 == 1:
            json_data[i]['context'][0][1] = [(simplify_sent(model, tokenizer, str(json_data[i]['context'][0][1])))]
        else:
            lst1 = []
            for j in range(0, l1-3, 2):
                lst1.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][0][1][j:j+2]))))
            lst1.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][0][1][l1-3:l1]))))
            json_data[i]['context'][0][1] = lst1

        l2 = len(json_data[i]['context'][1][1])
        if l2 % 2 == 0:
            lst2 = []
            for j in range(0, l2, 2):
                lst2.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][1][1][j:j+2]))))
            json_data[i]['context'][1][1] = lst2
        elif l2 == 1:
            json_data[i]['context'][1][1] = [(simplify_sent(model, tokenizer, str(json_data[i]['context'][1][1])))]
        else:
            lst2 = []
            for j in range(0, l2-3, 2):
                lst2.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][1][1][j:j+2]))))
            lst2.append((simplify_sent(model, tokenizer, str(json_data[i]['context'][1][1][l2-3:l2]))))
            json_data[i]['context'][1][1] = lst2"""


        # line['context'] = simplify_sent(line['context'])
    print("=========test=========")
    # print(json_data)
    with open('/home/team5/QGStepByStep/DCQG_data/simplified_text.json', 'a', encoding='utf-8') as f:
        torch.cuda.empty_cache()
        json.dump(json_data, f, ensure_ascii=False)



    # print(gold)
    # pred = []
    # with open("/home/team5/QGStepByStep/Datasets/output/QG/generated/baseline/pred_gpt2.txt", 'r', encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         pred.append(line)
    #         print("=============")
    #         new_text = simplify_sent(line, num)
    #         print(new_text)
    #         print("==============")
    #         for text in new_text:
    #             pred.extend(new_text)
    # print(pred)
    # with open('/home/team5/QGStepByStep/Datasets/output/QG/generated/simplified/paraphrase_pred.txt', 'w', encoding='utf-8') as f:
    #     for sent in tqdm(pred):
    #         f.write(sent)


    # with open('/home/team5/QGStepByStep/Datasets/output/QG/generated/simplified/paraphrase_gold.txt', 'w', encoding = 'utf-8') as f:
    #     for sent in tqdm(gold):
    #         f.write(sent)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    #num = 1
    tokenizer = AutoTokenizer.from_pretrained("philippelaban/keep_it_simple")
    kis_model = AutoModelForCausalLM.from_pretrained("philippelaban/keep_it_simple")
    device = torch.device("cuda")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = kis_model.to(device)
    model.eval()
    simplify(model, tokenizer)
