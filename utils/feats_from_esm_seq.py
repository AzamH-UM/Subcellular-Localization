import transformers
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification
from tqdm import tqdm
import pickle
import numpy as np

model_checkpoint = "facebook/esm2_t30_150M_UR50D"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = EsmForSequenceClassification.from_pretrained(model_checkpoint, output_hidden_states=True)
    model.eval()

    import csv
    with open("./Data/DataSplit2.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    outs = {}
    model.cuda()
    
    for i in tqdm(data[:20]):
        seq_txt = i[4]
        seq = tokenizer(seq_txt, return_tensors="pt")
        outputs = model(**seq.to(torch.device("cuda:0")))
        last_hidden_state = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
        outs[i[0]] = last_hidden_state
        del seq
        torch.cuda.empty_cache()
        
    np.save("esm_feats.npy", outs)