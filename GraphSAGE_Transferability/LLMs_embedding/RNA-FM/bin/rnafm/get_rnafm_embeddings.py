import pickle as pkl
import numpy as np
import fm
from tqdm import tqdm
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate RNA-FM embeddings")

parser.add_argument("-i", "--input", required=True, help="Input RNA sequence file")
parser.add_argument("-o", "--output", required=True, help="Output embedding file")

args = parser.parse_args()

Dataset = args.input
output_file = args.output

#############################################################
###########          get_rnafm_embeddings              ############
#############################################################

#Dataset="all.uniq.AUCG.seq.txt"
with open(Dataset) as f:
	rna_list = [line.strip() for line in f]


#prediction_interactions = None, ##确定是训练模式还是预测模式
embedding_dimension = 640 ## 设定640 维 embedding
#replace_dataframe = True, ##是否用新的 RNA embedding DataFrame 替换类内部保存的 self.rna_dataframe，本质是指训练数据集是否被替换
#return_normalisation_conststants = False, ##是否返回归一化参数
#delimiter = '\t' ##读取 DataFrame 或文件时使用的分隔符

# if type(prediction_interactions) != type(None): ##预测模式
# 	rna_list = list(prediction_interactions[self.rna_seq_name])
# 	replace_dataframe = False ## 只计算新 RNA 的 embedding，不替换原来的训练数据
# else:
# 	rna_list = self.rna_list

rna_embeddings = np.zeros((len(rna_list), embedding_dimension)) ##初始化 embedding 矩阵,(1871, 640)
length_of_rna = [0 for _ in range(len(rna_list))] ##用来记录RNA的个数，1871个元素的列表

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model, alphabet = fm.pretrained.rna_fm_t12() ##从官方库中加载预训练好的模型,model是RNA-FM模型，alphabet是token字典
batch_converter = alphabet.get_batch_converter() ##数据转换函数,在深度学习中，我们通常不会一条条地处理数据，而是成批（Batch）处理
model = model.to(device)
model.eval() ##将模型切换到评估（Evaluation）模式,如果你不写这一句，模型可能会保留随机性，导致你对同一条 RNA 提取出的特征向量每次都不一样，这在科学研究中是不可接受的

for idx, RNA in tqdm(enumerate(rna_list)): ##遍历每条 RNA
	# 创建包含元组的列表
	if len(RNA) > 1022: ##RNA长度 >1022,RNA-FM 最大 token 长度是：1024,但要减去：<BOS><EOS>,所以最大 RNA 长度：1022
		chunk_size = 1022 # 指定每部分的长度
		chunks = [RNA[i:i + chunk_size] for i in range(0, len(RNA), chunk_size)] # 使用切片将字符串分成指定长度的部分
		for j, chunk in enumerate(chunks): # 对每个chunk部分进行操作（示例：打印每个部分）
			RNA_tuple = [("RNA1", chunk)] ##它创建了一个包含元组（Tuple）的列表,元组的第一个元素 "RNA1" 是这条序列的标签或名字（你可以随意命名）,第二个元素 chunk 是实际的 RNA 碱基字符串（如 "AUGCCG..."）
			batch_labels, batch_strs, batch_tokens = batch_converter(RNA_tuple) ##这一行是真正的**“翻译”过程**，它通过之前定义的转换器处理数据。它会返回三个变量:batch_labels:就是刚才传进去的名字，即 ['RNA1'];batch_strs:经过处理后的字符串;batch_tokens (最核心的部分):这是一个 PyTorch Tensor（张量），是一串数字,这是唯一会真正输入到模型 model() 里的变量。模型看不懂字母，只能计算这些数字。
			batch_tokens = batch_tokens.to(device)
			with torch.no_grad(): ##这段代码是整个特征提取过程中的“计算核心”。它的任务是将数字化后的序列（batch_tokens）输入到深度神经网络中，并拿到该模型最深层（第 12 层）产生的高维特征表示
				results = model(batch_tokens, repr_layers=[12]) ##results:(Batch_Size, Sequence_Length, 640),results内部通常包含：logits: 预测每个位置碱基概率的原始分值;representations: 这是一个子字典，里面存放着你要求的不同层的特征向量;attentions: （如果你请求了的话）每一层的注意力权重矩阵
			token_embeddings_chunk = results["representations"][12] ##从模型返回的字典 results 中，取出第 12 层（最后一层）的隐藏层表示,此时它是一个 PyTorch 张量（Tensor），形状通常是 (1, L, 640)
			token_embeddings_chunk_np = token_embeddings_chunk.cpu().numpy() ##将数据从 PyTorch Tensor 格式转换为 NumPy 数组
			token_embeddings_chunk_np = token_embeddings_chunk_np[0,] ##降维。取出第一个（也是唯一一个）Batch 的内容,形状从 (1, L, 640) 变成了 (L, 640)
			token_embeddings_chunk_np_RNAlength = token_embeddings_chunk_np[1:-1, :] ##去除特殊标记（Special Tokens）,切掉第 0 位，即去掉了序列开头的 <cls> 标记,切掉最后一位，即去掉了序列结尾的 <eos> 标记
			token_embeddings_chunk_np_RNAlength_mean = np.mean(token_embeddings_chunk_np_RNAlength, axis=0) ##全局平均池化（Global Average Pooling）,将形状从 (L_real, 640) 压缩成了 (640,)
			if j == 0: ##将所有chunk块的平均值拼接起来成一个矩阵
				token_embeddings_np_RNAlength = token_embeddings_chunk_np_RNAlength_mean
			else:
				token_embeddings_np_RNAlength = np.vstack([token_embeddings_np_RNAlength, token_embeddings_chunk_np_RNAlength_mean])
		# print(token_embeddings_np_RNAlength_mean.shape)
		# print(token_embeddings_np_RNAlength_mean)
		token_embeddings_np_RNAlength_mean = np.mean(token_embeddings_np_RNAlength, axis=0) ##所有 chunk 的 embedding 再平均
		# print(token_embeddings_np_RNAlength_mean.shape)
		# print(token_embeddings_np_RNAlength_mean)
		rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean ##保存 RNA embedding
	else: ##如果RNA长度 ≤1022，直接计算 embedding
		RNA_tuple = [("RNA1", RNA)]
		batch_labels, batch_strs, batch_tokens = batch_converter(RNA_tuple)
		batch_tokens = batch_tokens.to(device)
		with torch.no_grad():
			results = model(batch_tokens, repr_layers=[12])
		token_embeddings = results["representations"][12]
		token_embeddings_np = token_embeddings.cpu().numpy()
		token_embeddings_np = token_embeddings_np[0,]
		token_embeddings_np_RNAlength = token_embeddings_np[1:-1, :]
		token_embeddings_np_RNAlength_mean = np.mean(token_embeddings_np_RNAlength, axis=0)
		# print(token_embeddings_np_RNAlength_mean)
		rna_embeddings[idx, :] = token_embeddings_np_RNAlength_mean ##保存 RNA embedding

## Normalize embeddings ##
# mean_rna_embeddings = np.mean(rna_embeddings, axis=0) ##对所有RNA计算平均 embedding，unseen需要
# centered_rna_embeddings = rna_embeddings - mean_rna_embeddings ##中心化
# centered_rna_embeddings_length = np.mean(np.sqrt(np.sum(centered_rna_embeddings * centered_rna_embeddings, axis=1))) ##unseen需要，计算平均向量长度，长度指的是这个点距离原点 $(0,0,0...)$ 的几何距离
# normalized_rna_embeddings = centered_rna_embeddings / np.expand_dims(centered_rna_embeddings_length, axis=-1) ##特征缩放,将所有中心化后的 RNA 向量调整到一个统一的数值量级,normalization

# emb = np.array(normalized_rna_embeddings)
# target_dim = 1024
# padded_emb = np.zeros((emb.shape[0], target_dim), dtype=np.float32)
# padded_emb[:, :emb.shape[1]] = emb
# feature_cols = [f"fea_{i+1}" for i in range(target_dim)]
# rna_dataframe = pd.DataFrame(padded_emb, columns=feature_cols)
# rna_list_dna = [seq.replace("U", "T") for seq in rna_list]
# rna_dataframe.insert(0, "RNA_sequence", rna_list_dna)
# rna_dataframe.to_csv(output_file,sep="\t",index=False,header=True)

rna_dataframe = pd.DataFrame({"RNA_sequence": rna_list,"embedding": list(rna_embeddings)})
with open(output_file, "wb") as f:
    pkl.dump(rna_dataframe, f)

#self.rna_embeddings = rna_embeddings


# # Normalize embeddings - train data
# if type(prediction_interactions) == type(None): ##训练模式
# 	self.rna_embeddings = rna_embeddings
# 	self.mean_rna_embeddings = np.mean(rna_embeddings, axis=0) ##对所有RNA计算平均 embedding
# 	self.centered_rna_embeddings = rna_embeddings - self.mean_rna_embeddings ##中心化
# 	self.centered_rna_embeddings_length = np.mean(np.sqrt(np.sum(self.centered_rna_embeddings * self.centered_rna_embeddings, axis=1))) ##计算平均向量长度，长度指的是这个点距离原点 $(0,0,0...)$ 的几何距离
# 	self.normalized_rna_embeddings = self.centered_rna_embeddings / np.expand_dims(
# 		self.centered_rna_embeddings_length, axis=-1) ##特征缩放,将所有中心化后的 RNA 向量调整到一个统一的数值量级,normalization
# # Normalize for prediction data and return
# else: ##预测模式
# 	centered_rna_embeddings = rna_embeddings - self.mean_rna_embeddings ####统一去中心化 (Centering),这里减去的 self.mean_rna_embeddings 是在训练阶段算出来的全局平均值
# 	normalized_rna_embeddings = centered_rna_embeddings / np.expand_dims(
# 		self.centered_rna_embeddings_length, axis=-1)  ## 统一缩放 (Scaling),同样使用训练集算出来的 centered_rna_embeddings_length（那个标量数字）作为分母
# 	rna_dataframe = pd.DataFrame([rna_list, normalized_rna_embeddings]).T ##构建数据表, 将 RNA 的名称列表（rna_list）和处理后的特征向量（normalized_rna_embeddings）拼成一个表格
# 	rna_dataframe.columns = [self.rna_seq_name, self.rna_embedding_name] ##赋予清晰的列名，通常是类似 RNA_ID 和 Embedding
# 	self.rna_dataframe = rna_dataframe
# 	return rna_dataframe
# # self.normalized_rna_embeddings = self.rna_embeddings
# # Replace proteins dataframe with 它根据用户的不同需求，决定是以“表格”形式保存结果，还是直接返回“原始数值”用于后续计算
# if replace_dataframe: ##是否替换 dataframe,如果是的话：
# 	self.rna_dataframe = pd.DataFrame([rna_list, self.normalized_rna_embeddings]).T
# 	self.rna_dataframe.columns = [self.rna_seq_name, self.rna_embedding_name]

# if return_normalisation_conststants:
# 	return self.rna_embeddings, self.centered_rna_embeddings_length, self.normalized_rna_embeddings
