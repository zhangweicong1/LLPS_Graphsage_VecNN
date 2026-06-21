import os
import sys
import argparse
from datetime import datetime
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
import math
import numpy as np
import pandas as pd
import pyhocon
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score

from my_datacenter import *
from my_models_GraphSage import *
from VecNN_model import *
from my_utils import *

def fix_seed(seed): ##固定所有随机数生成器的种子，保证实验结果可复现
	""" 
	Seed all necessary random number generators.
	"""
	if seed is None:
		seed = random.randint(1, 10000) ##如果没有指定seed，则随机生成一个1到10000之间的整数作为seed
	#torch.set_num_threads(1)  ##限制 PyTorch 的 CPU 线程为 1，Python 内置 random 模块的随机性
	os.environ['PYTHONHASHSEED'] = str(seed) 

	random.seed(seed) ##Python内置random模块的随机数生成器，seed会产生可预测的固定顺序
	np.random.seed(seed) ##控制的是 NumPy 的随机模块
	torch.manual_seed(seed) ##控制 PyTorch 的 CPU 随机数
	torch.cuda.manual_seed(seed) ##控制 当前 GPU 的随机数
	torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.控制 所有 GPU 的随机数

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	## torch.use_deterministic_algorithms(True)

def get_training_graphsage_embeddings(dataSet=None, seed=64, config='./graphsage_src/experiments.conf', graphsage_embedding=1):

	fix_seed(seed)

	if dataSet==None:
		sys.exit("请输入正确数据集!")

	config = pyhocon.ConfigFactory.parse_file(config)
	ds = dataSet ## dataSet='test_random'
	dataCenter = DataCenter(config)
	#dataCenter.load_VecNN_training_dataSet(ds) #运行dataCenter
	node_map, index_to_seq, label_df = dataCenter.load_VecNN_training_dataSet(ds)
	label_dict = dict(zip(label_df.iloc[:,0], label_df.iloc[:,1]))
	RNA_index_list = getattr(dataCenter, ds + '_RNA_index_list')
	features = getattr(dataCenter, ds + '_feats')
	graphsage_model_path = getattr(dataCenter, ds + '_graphsage_model_path')

	five_folds = getattr(dataCenter, ds + '_folds')
	if graphsage_embedding == 1:
		graphsage_allRNA_nodes_embeddings = run_graphsage_model(nodes=RNA_index_list, cuda=True, graphsage_model_path=graphsage_model_path)
		assert graphsage_allRNA_nodes_embeddings.shape[0] == len(RNA_index_list)
	
	all_folds_results = []
	for fold_data in five_folds:
		fold_id = fold_data["fold"]
		train_nodes = fold_data["train_nodes"]
		test_nodes = fold_data["test_nodes"]
		# train_features = features[train_nodes]
		# test_features = features[test_nodes]
		train_features = features[train_nodes][:, :640]
		test_features = features[test_nodes][:, :640]

		if graphsage_embedding == 1:
			graphsage_train_nodes_embeddings = graphsage_allRNA_nodes_embeddings[train_nodes]
			graphsage_test_nodes_embeddings = graphsage_allRNA_nodes_embeddings[test_nodes]
			# print(graphsage_train_nodes_embeddings.shape)
			# print(graphsage_test_nodes_embeddings.shape)
			assert graphsage_train_nodes_embeddings.shape[0] == train_features.shape[0]
			graphsage_train_embeddings = np.concatenate((graphsage_train_nodes_embeddings, train_features), axis=1)
			assert graphsage_test_nodes_embeddings.shape[0] == test_features.shape[0]
			graphsage_test_embeddings = np.concatenate((graphsage_test_nodes_embeddings, test_features), axis=1)
		elif graphsage_embedding == 0:
			graphsage_train_embeddings = train_features
			graphsage_test_embeddings = test_features
		else:
			print("请输入正确的graphsage_embedding的参数!")
		
		train_seqs, train_label_array = nodes_to_seqs_and_labels(nodes=train_nodes, index_to_seq=index_to_seq, label_dict=label_dict, debug=True)
		if not (len(train_seqs) == len(train_label_array) == graphsage_train_embeddings.shape[0]):
			raise ValueError(f"Length mismatch: "f"train_seqs={len(train_seqs)}, "f"train_label_array={len(train_label_array)}, "f"embeddings={graphsage_train_embeddings.shape[0]}")
		test_seqs, test_label_array = nodes_to_seqs_and_labels(nodes=test_nodes, index_to_seq=index_to_seq, label_dict=label_dict, debug=True)
		if not (len(test_seqs) == len(test_label_array) == graphsage_test_embeddings.shape[0]):
			raise ValueError(f"Length mismatch: "f"test_seqs={len(test_seqs)}, "f"test_label_array={len(test_label_array)}, "f"embeddings={graphsage_test_embeddings.shape[0]}")

		fold_result = {"fold_id": fold_id, "train_nodes": train_nodes, "test_nodes": test_nodes, "train_seqs": train_seqs, "test_seqs": test_seqs, "train_labels": train_label_array, "test_labels": test_label_array, "train_embeddings": graphsage_train_embeddings, "test_embeddings": graphsage_test_embeddings}
		all_folds_results.append(fold_result)
	
	return all_folds_results


def Graphsage_VecNN_model_Train(K_folds_results=None, epochs=30, b_sz=128, seed=64, graphsage_embedding=1, result_path='./'):

	## 设备选择,模型训练在 GPU 可行情况下使用 CUDA
	fix_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	os.makedirs(result_path, exist_ok=True)

	# 5-fold结果
	all_auc = []
	all_auprc = []
	all_acc = []
	all_precision = []
	all_recall = []
	all_mcc = []
	all_spe = []
	fold_results = []
	
	# Iterate over 5 folds
	for fold_data in K_folds_results:

		fold_id = fold_data["fold_id"]
		seq_train = fold_data["train_seqs"]
		X_train = fold_data["train_embeddings"]
		y_train = fold_data["train_labels"]
		seq_test = fold_data["test_seqs"]
		X_test = fold_data["test_embeddings"]
		y_test = fold_data["test_labels"]

		train_df = pd.DataFrame({"RNA_sequence": seq_train,"label": y_train})
		train_df["embedding"] = list(X_train)
		test_df = pd.DataFrame({"RNA_sequence": seq_test,"label": y_test})
		test_df["embedding"] = list(X_test)
		fold_dir = os.path.join(result_path,f"fold_{fold_id}")
		#os.makedirs(fold_dir, exist_ok=True)
		## 保存训练集和测试数据集
		dataset_dir = os.path.join(fold_dir, "dataset")
		os.makedirs(dataset_dir, exist_ok=True) 
		train_df.to_pickle(os.path.join(dataset_dir,"train_dataframe.pkl"))
		test_df.to_pickle(os.path.join(dataset_dir,"test_dataframe.pkl"))
		print(f"Fold {fold_id} saved to {fold_dir}")

		## 设置日志文件SummaryWriter和weight保存路径
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		log_path = os.path.join(fold_dir,"tensorboard",timestamp)
		writer = SummaryWriter(log_dir=log_path) ##创建tensorboard实例
		weight_dir = os.path.join(fold_dir,"weight")
		os.makedirs(weight_dir, exist_ok=True)

		## Initialize dataset and dataloader
		X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train,y_train,test_size=0.2,stratify=y_train,random_state=42) ##划分训练集和验证集
		train_dataset = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.float32))
		train_loader = DataLoader(train_dataset, batch_size=b_sz, shuffle=True)
		validation_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
		validation_loader = DataLoader(validation_dataset, batch_size=b_sz, shuffle=False)


		## Build the VecNN model
		model = VecNN(rna_embed_len=X_train_sub.shape[1],graphsage_embedding=graphsage_embedding).to(device)

		# Initialize criterion and optimizer
		criterion = nn.BCELoss()
		##criterion = nn.BCEWithLogitsLoss()
		optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

		best_auc = -float("inf")
		best_epoch = 0
		best_model = None

		## 训练模型
		for epoch in range(epochs):
			print(f'----------------------FOLD {fold_id} EPOCH {epoch}-----------------------')
			epoch_loss, val_loss, val_auc = apply_VecNN_model(train_loader=train_loader, validation_loader=validation_loader, VecNN_model=model, device=device, criterion=criterion, optimizer=optimizer)
			writer.add_scalar('Train/EpochLoss', epoch_loss, epoch + 1)
			writer.add_scalar('Val/Loss', val_loss, epoch + 1)
			writer.add_scalar('Val/AUC', val_auc, epoch + 1)
			
			if val_auc > best_auc:
				best_auc = val_auc
				best_epoch = epoch + 1
				#best_weights = copy.deepcopy(model.state_dict())
				best_model = copy.deepcopy(model)
			
			writer.add_scalar('Val/BestAUC',best_auc,epoch + 1)

		#torch.save(best_weights, os.path.join(weight_dir, f"VecNN_best_fold_{fold_id}_epoch{best_epoch}.pth"))
		torch.save(best_model,os.path.join(weight_dir,f"VecNN_best_fold_{fold_id}.pth"))
		#print(f"Fold {fold_id} best_auc: {best_auc:.4f}")
		print(f"Fold {fold_id} "f"Best Epoch={best_epoch} "f"Best AUC={best_auc:.4f}")
		writer.close()

		## 验证集寻找最佳cutoff阈值
		#model.load_state_dict(best_weights)
		model = best_model
		model.eval()

		val_probs = []
		val_labels = []
		with torch.no_grad():
			for val_x, val_y in validation_loader:
				val_x = val_x.to(device)
				pred = model(val_x).squeeze(-1)
				val_probs.extend(pred.cpu().numpy())
				val_labels.extend(val_y.cpu().numpy())
		val_probs = np.array(val_probs)
		val_labels = np.array(val_labels)

		thresholds = np.linspace(0, 1, 200) ##划分阈值，0到1之间，均匀生成200个数字，把0到1这个区间切成200份，每份间距是0.005
		best_mcc = -1 ##MCC的取值范围是 [-1, 1]，Matthews Correlation Coefficient
		best_cutoff = 0.5 ##cutoff的默认值

		for threshold in thresholds: ##逐个尝试每个阈值，看哪个MCC最高
			pred_labels = (val_probs > threshold).astype(int)
			mcc = matthews_corrcoef(val_labels, pred_labels)
			if mcc > best_mcc:
				best_mcc = mcc
				best_cutoff = threshold
		#print(f"Fold {fold_id} "f"Best MCC={best_mcc:.4f} "f"Cutoff={best_cutoff:.4f}")

		##测试集预测
		test_dataset = TensorDataset(torch.tensor(X_test,dtype=torch.float32), torch.tensor(y_test,dtype=torch.float32))
		test_loader = DataLoader(test_dataset, batch_size=b_sz, shuffle=False)

		test_probs = []
		test_labels = []
		with torch.no_grad():
			for test_x, test_y in test_loader:
				test_x = test_x.to(device)
				pred = model(test_x).squeeze(-1)
				test_probs.extend(pred.cpu().numpy())
				test_labels.extend(test_y.cpu().numpy())
		test_probs = np.array(test_probs)
		test_labels = np.array(test_labels)

		## 测试集的指标
		test_auc = roc_auc_score(test_labels, test_probs)
		test_auprc = average_precision_score(test_labels, test_probs)
		test_pred_labels = (test_probs > best_cutoff).astype(int)

		test_acc = accuracy_score(test_labels,test_pred_labels)
		#test_precision = precision_score(test_labels,test_pred_labels)
		test_precision = precision_score(test_labels,test_pred_labels,zero_division=0)
		#test_recall = recall_score(test_labels,test_pred_labels) 
		test_recall = recall_score(test_labels, test_pred_labels, zero_division=0) ##所有真实正样本被验证出的概率
		test_mcc = matthews_corrcoef(test_labels,test_pred_labels) ##模型预测标签和真实标签之间的一致性

		#cm = confusion_matrix(test_labels, test_pred_labels)
		cm = confusion_matrix(test_labels, test_pred_labels, labels=[0,1])
		tn, fp, fn, tp = cm.ravel()
		test_spe = tn / (tn + fp) if (tn + fp) > 0 else 0

		all_auc.append(test_auc)
		all_auprc.append(test_auprc)
		all_acc.append(test_acc)
		all_precision.append(test_precision)
		all_recall.append(test_recall)
		all_mcc.append(test_mcc)
		all_spe.append(test_spe)
		fold_results.append({
			"fold": fold_id,
			"best_epoch": best_epoch,
			"best_auc_val": best_auc,
			"best_cutoff": best_cutoff,
			"test_auc": test_auc,
			"test_auprc": test_auprc,
			"test_acc": test_acc,
			"test_precision": test_precision,
			"test_recall": test_recall,
			"test_mcc": test_mcc,
			"test_specificity": test_spe,
			"tn": tn,
			"fp": fp,
			"fn": fn,
			"tp": tp
		})

		# ========================= ROC curve =========================
		fpr, tpr, roc_thresholds = roc_curve(test_labels, test_probs)
		roc_df = pd.DataFrame({"fpr": fpr,"tpr": tpr,"threshold": roc_thresholds})
		roc_path = os.path.join(fold_dir, "roc_curve.csv")
		roc_df.to_csv(roc_path, index=False)

		# ========================= PR curve =========================
		precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_labels, test_probs)
		pr_df = pd.DataFrame({"precision": precision_curve,"recall": recall_curve})
		pr_path = os.path.join(fold_dir, "pr_curve.csv")
		pr_df.to_csv(pr_path, index=False)

		##打印当前fold结果
		print(f"Fold {fold_id} | " f"AUC={test_auc:.4f} | " f"AUPRC={test_auprc:.4f} | " f"ACC={test_acc:.4f} | " f"Precision={test_precision:.4f} | " f"Recall={test_recall:.4f} | " f"Specificity={test_spe:.4f} | " f"MCC={test_mcc:.4f} | " f"Cutoff={best_cutoff:.4f}")

	results_df = pd.DataFrame(fold_results)
	results_df.to_csv(os.path.join(result_path, "5fold_metrics_results.csv"), index=False, sep='\t')
	print(f"Results saved to "f"{os.path.join(result_path, '5fold_metrics_results.csv')}")

	summary_mean_df = pd.DataFrame({"Metric":["AUC","AUPRC","ACC","Precision","Recall","Specificity","MCC"],
									"Mean":[np.mean(all_auc),np.mean(all_auprc),np.mean(all_acc),np.mean(all_precision),np.mean(all_recall),np.mean(all_spe),np.mean(all_mcc)],
									"Std":[np.std(all_auc),np.std(all_auprc),np.std(all_acc),np.std(all_precision),np.std(all_recall),np.std(all_spe),np.std(all_mcc)]})
	summary_mean_df.to_csv(os.path.join(result_path, "5fold_summary.csv"), index=False, sep='\t')


#def run_graphsage_model(nodes=None, seed=64, cuda=False, graphsage_model_path=None):
def run_graphsage_model(nodes=None, cuda=False, graphsage_model_path=None):

	if torch.cuda.is_available():
		if not cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			device_id = torch.cuda.current_device()
			print('using device', device_id, torch.cuda.get_device_name(device_id))
		cuda = torch.cuda.is_available()

	device = torch.device("cuda" if cuda else "cpu")
	print('DEVICE:', device)

	#fix_seed(seed)

	### 输入模型
	model_path = graphsage_model_path
	#graphSage = torch.load(model_path)
	graphSage = torch.load(model_path, map_location=device,weights_only=False)
	# print(graphSage)
	graphSage.to(device)
	graphSage.eval()
	with torch.no_grad():
		embs = get_gnn_embeddings(graphSage, nodes)
	# print(f'Output embs: {embs}')
	# print(f'Output embs shape: {embs.shape}')
	graphsage_embeddings = embs.cpu().numpy()
	#self.graphSage = graphSage
	return graphsage_embeddings

def get_gnn_embeddings(gnn_model, nodes):
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        # print(f'打印nodes_batch: {nodes_batch}')
        # print(nodes_batch)
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    # print(embs.size())
    assert len(embs) == len(nodes)
    return embs.detach()

def nodes_to_seqs_and_labels(nodes, index_to_seq, label_dict, debug=False):

	rna_seqs = []
	label_list = []
	missing_seq = 0
	missing_label = 0
	for node in nodes:
		if node not in index_to_seq:
			missing_seq += 1
			continue
		rna_seq = index_to_seq[node]

		if rna_seq not in label_dict:
			missing_label += 1
			continue
		rna_seqs.append(rna_seq)
		label_list.append(label_dict[rna_seq])
	
	label_array = np.array(label_list)
	if debug: ##是否调试输出
		print(f"missing sequence: {missing_seq}")
		print(f"missing label: {missing_label}")

	return rna_seqs, label_array

def main(DataSet, Config, Save_dir, seed=64, graphsage_embedding=1):

	K_folds_results = get_training_graphsage_embeddings(dataSet=DataSet,seed=seed,config=Config,graphsage_embedding=graphsage_embedding)
	Graphsage_VecNN_model_Train(K_folds_results=K_folds_results,epochs=50,b_sz=128,seed=seed,graphsage_embedding=graphsage_embedding,result_path=Save_dir)

	print("========== Pipeline Finished ==========")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Training models")
	parser.add_argument("--DataSet", type=str, required=True, help="The name of the dataset to be trained")
	parser.add_argument("--Config", type=str, required=True, help="Input configuration file")
	parser.add_argument("--Save_dir", type=str, required=True, help="Output directory")
	args = parser.parse_args()

	main(args.DataSet, args.Config, args.Save_dir)
