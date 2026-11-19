import pickle as pkl
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate ProtTrans embeddings")

parser.add_argument("-i", "--input", required=True, help="Input Protein sequence file")
parser.add_argument("-o", "--output", required=True, help="Output embedding file")

args = parser.parse_args()

Dataset = args.input
output_file = args.output

#Dataset = "/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/LLMs_embedding/ProtTrans/data/all.protein.seq.list"
with open(Dataset) as f:
	protein_list = [line.strip() for line in f]

####01.input && parameter########################
#prediction_interactions=None ##预测用的 DataFrame，若为 None 则处理训练集蛋白质
embedding_dimension=1024
# replace_dataframe=True ##布尔值，是否用新結果覆蓋 self.proteins_dataframe
# return_normalisation_conststants=False ##布尔值，如果为真，则返回归一化常数
#delimiter='\t' ##字符串,用于读取 Pandas ProtVec DataFrame 的分隔符

# if type(prediction_interactions) != type(None): ##如果使用prediction_interactions（預測模式），就从DataFrame的蛋白質序列欄位取出序列，否則使用训练集self.protein_list
# 		protein_list = list(prediction_interactions[self.protein_seq_name])
# 		replace_dataframe = False
# else:
# 	protein_list = self.protein_list

protein_embeddings = np.zeros((len(protein_list), embedding_dimension)) ##初始化 (N, 1024) 的零矩陣來儲存所有蛋白質的 embedding
length_of_protein = [0 for _ in range(len(protein_list))]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####02.导入ProtT5模型########################
# tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False) # Load the tokenizer
# model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device) # Load the model
MODEL_DIR = "/home/isou/lncRNA/LLPS/DLM/bin/LLMs_embedding/ProtTrans/ProtTrans_pre-trained_models/ProtT5-XL-U50"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, do_lower_case=False, local_files_only=True)
model = T5EncoderModel.from_pretrained(MODEL_DIR,local_files_only=True).to(device)
model.full() if device == 'cpu' else model.half() ## GPU使用model.half():float16（半精度）;CPU必须用model.full(): float32

####03.計算每个蛋白质的embedding########################
# For each protein in protein list
for idx, protein in tqdm(enumerate(protein_list)):
	protein = [protein]
	protein = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in protein] #將罕見或模糊的胺基酸（U、Z、O、B）替換成 X,用空格把每個胺基酸字母分開，例如 "ACDE" → "A C D E"（ProtT5 的輸入格式要求）,replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
	# tokenize sequences and pad up to the longest sequence in the batch
	ids = tokenizer(protein, add_special_tokens=True, padding="longest") ##將序列转成token id，加上特殊 token（如 </s>），並 padding 到 batch 內最長序列
	input_ids = torch.tensor(ids['input_ids']).to(device) ##移至 GPU/CPU
	attention_mask = torch.tensor(ids['attention_mask']).to(device)
	with torch.no_grad(): #generate embeddings，在不计算梯度的情况下，把蛋白质序列喂给模型，进行推理，反向传播更新参数
		embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
	# extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens ([0,:7]),提取 per-protein embedding
	emb_0 = embedding_repr.last_hidden_state[0, :-1]  # 去掉最後一個特殊 token，去掉結尾的 </s> special token（:-1）
	# if you want to derive a single representation (per-protein embedding) for the whole protein
	emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024),對所有 residue 取平均 → shape (1024,),對序列長度維度做 mean pooling，得到整條蛋白質的單一 1024 維向量
	protein_embeddings[idx, :] = emb_0_per_protein.cpu().numpy()

# # Normalize embeddings 
mean_protein_embeddings = np.mean(protein_embeddings, axis = 0)
centered_protein_embeddings = protein_embeddings - mean_protein_embeddings
centered_protein_embeddings_length = np.mean(np.sqrt(np.sum(centered_protein_embeddings * centered_protein_embeddings, axis = 1)))
normalized_protein_embeddings = centered_protein_embeddings / np.expand_dims(centered_protein_embeddings_length, axis = -1)

emb = np.array(normalized_protein_embeddings)
feature_cols = [f"fea_{i+1}" for i in range(emb.shape[1])]
protein_dataframe = pd.DataFrame(emb, columns=feature_cols)
protein_dataframe.insert(0, "Protein_sequence", protein_list)
protein_dataframe.to_csv(output_file,sep="\t",index=False,header=True)

# protein_dataframe = pd.DataFrame({"Protein_sequence": protein_list,"embedding": list(protein_embeddings)})
# with open(output_file, "wb") as f:
# 	pkl.dump(protein_dataframe, f)


# self.protein_embeddings = protein_embeddings
# # self.mean_protein_embeddings = np.mean(protein_embeddings, axis=0)
# # Normalize embeddings - train data

# if type(prediction_interactions) == type(None):
# 	self.protein_embeddings = protein_embeddings
# 	self.mean_protein_embeddings = np.mean(protein_embeddings, axis = 0)
# 	self.centered_protein_embeddings = protein_embeddings - self.mean_protein_embeddings
# 	self.centered_protein_embeddings_length = np.mean(np.sqrt(np.sum(self.centered_protein_embeddings * self.centered_protein_embeddings, axis = 1)))
# 	self.normalized_protein_embeddings = self.centered_protein_embeddings / np.expand_dims(self.centered_protein_embeddings_length, axis = -1)

# # Normalize for prediction data and return
# else:
# 	centered_protein_embeddings = protein_embeddings - self.mean_protein_embeddings
# 	normalized_protein_embeddings = centered_protein_embeddings / np.expand_dims(self.centered_protein_embeddings_length, axis = -1)
# 	proteins_dataframe = pd.DataFrame([protein_list, normalized_protein_embeddings]).T
# 	proteins_dataframe.columns = [self.protein_seq_name, self.protein_embedding_name]
# 	self.proteins_dataframe = proteins_dataframe
# 	return proteins_dataframe

# # self.normalized_protein_embeddings = self.protein_embeddings
# # Replace proteins dataframe with
# if replace_dataframe:
# 	self.proteins_dataframe = pd.DataFrame([protein_list, self.normalized_protein_embeddings]).T
# 	self.proteins_dataframe.columns = [self.protein_seq_name, self.protein_embedding_name]

# if return_normalisation_conststants:
# 	return self.protein_embeddings, self.centered_protein_embeddings_length, self.normalized_protein_embeddings
