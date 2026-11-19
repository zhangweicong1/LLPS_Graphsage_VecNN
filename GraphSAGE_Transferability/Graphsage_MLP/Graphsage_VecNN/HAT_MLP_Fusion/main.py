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
#import math
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


def run_graphsage_experiment(dataSet=None, agg_func='MEAN', epochs=3, b_sz=20, seed=64, cuda=False, gcn=False, learn_method='unsup', unsup_loss='margin', max_vali_f1=0, name='debug', config='./graphsage_src/experiments.conf', save_dir=None):

	if dataSet==None:
		sys.exit("请输入正确数据集!")
	if torch.cuda.is_available():
		if not cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			device_id = torch.cuda.current_device()
			print('using device', device_id, torch.cuda.get_device_name(device_id))
		cuda = torch.cuda.is_available()

	device = torch.device("cuda" if cuda else "cpu")
	print('DEVICE:', device)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_path = os.path.join(save_dir,dataSet,"tensorboard",timestamp)
	writer = SummaryWriter(log_dir=log_path) ##创建tensorboard实例
	weight_dir = os.path.join(save_dir, dataSet, "weight")
	os.makedirs(weight_dir, exist_ok=True)

	config = pyhocon.ConfigFactory.parse_file(config) ##config='/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Pre_training_Graph_neural_network/experiments.conf'

	ds = dataSet ## dataSet='test_random'
	dataCenter = DataCenter(config)
	dataCenter.load_Graphsage_pretrain_dataSet(ds) #运行dataCenter
	features = torch.FloatTensor(getattr(dataCenter, ds + '_feats')).to(device)
	#print(f'feature in dataCenter: {features}')

	# print("features shape:", features.shape)
	# print("features std:", features.std())
	# cos = F.cosine_similarity(features[0].unsqueeze(0),features[1].unsqueeze(0))
	# print("node0-node1 cosine:", cos.item())

	graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], dataCenter,
							features, getattr(dataCenter, ds + '_adj_lists'), device, gcn=gcn, agg_func=agg_func)
	graphSage.to(device)
	### 直接输出，再保存graphsage的模型
	print(f"nn structure: {graphSage}")
	#print(graphSage.adj_lists)
	unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds + '_adj_lists'), getattr(dataCenter, ds + '_train'), device)

	if learn_method == 'sup':
		print('GraphSage with Supervised Learning')
	elif learn_method == 'plus_unsup':
		print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
	else:
		print('GraphSage with Net Unsupervised Learning')

	best_loss = float('inf')
	best_epoch = -1
	for epoch in range(epochs):
		print('----------------------EPOCH %d-----------------------' % epoch)
		#graphSage = apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method)
		graphSage, epoch_loss = apply_pregraphsage_model(dataCenter=dataCenter, ds=ds, graphSage=graphSage, unsupervised_loss=unsupervised_loss, b_sz=b_sz, unsup_loss=unsup_loss, device=device, learn_method=learn_method, writer=writer, epoch=epoch)
		
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			best_epoch = epoch
			print(f"New best model at epoch {epoch + 1}, loss = {best_loss:.4f}")
			weight_path = os.path.join(weight_dir,"Pretrain_GraphSage_best_epoch.pth")
			#weight_path = '{}/Pretrain_GraphSage_epoch{}.pth'.format(save_dir, epoch + 1)
			torch.save(graphSage,weight_path)
			#torch.save(graphSage.state_dict(),weight_path)

	writer.close()


def get_benchmark_graphsage_embeddings(dataSet=None, seed=64, config='./graphsage_src/experiments.conf', graphsage_model_path=None, graphsage_embedding=0):

	fix_seed(seed)

	if dataSet==None:
		sys.exit("请输入正确数据集!")

	config = pyhocon.ConfigFactory.parse_file(config)
	ds = dataSet ## dataSet='test_random'
	dataCenter = DataCenter(config)
	#dataCenter.load_VecNN_training_dataSet(ds) #运行dataCenter
	node_map, index_to_seq, label_df = dataCenter.load_benchmark_dataSet(ds)
	label_dict = dict(zip(label_df.iloc[:,0], label_df.iloc[:,1]))
	RNA_index_list = getattr(dataCenter, ds + '_RNA_index_list')
	#features = getattr(dataCenter, ds + '_feats')
	rna_hat_features = getattr(dataCenter, ds + '_rna_hat_feats')
	#graphsage_model_path = getattr(dataCenter, ds + '_graphsage_model_path')

	five_folds = getattr(dataCenter, ds + '_folds')
	if graphsage_embedding == 1:
		graphsage_allRNA_nodes_embeddings = run_graphsage_model(nodes=RNA_index_list, cuda=True, graphsage_model_path=graphsage_model_path)
		assert graphsage_allRNA_nodes_embeddings.shape[0] == len(RNA_index_list)
	
	all_folds_results = []
	for fold_data in five_folds:
		fold_id = fold_data["fold"]
		train_nodes = fold_data["train_nodes"]
		test_nodes = fold_data["test_nodes"]
		# Node index -> RNA sequence
		train_seqs = [index_to_seq[node] for node in train_nodes]
		test_seqs = [index_to_seq[node] for node in test_nodes]

		# RNA sequence -> RNA-FM + HAT embedding
		train_features = np.stack([rna_hat_features[rna_hat_features[:, 0] == seq, 1:][0] for seq in train_seqs]).astype(np.float32)
		test_features = np.stack([rna_hat_features[rna_hat_features[:, 0] == seq, 1:][0] for seq in test_seqs]).astype(np.float32)

		if graphsage_embedding == 1:
			graphsage_train_embeddings = graphsage_allRNA_nodes_embeddings[train_nodes]
			graphsage_test_embeddings = graphsage_allRNA_nodes_embeddings[test_nodes]

			assert graphsage_train_embeddings.shape[0] == train_features.shape[0]
			assert graphsage_test_embeddings.shape[0] == test_features.shape[0]
		elif graphsage_embedding == 0:
			graphsage_train_embeddings = None
			graphsage_test_embeddings = None
		else:
			print("请输入正确的graphsage_embedding的参数!")
			sys.exit(1)
		
		train_seqs, train_label_array = nodes_to_seqs_and_labels(nodes=train_nodes, index_to_seq=index_to_seq, label_dict=label_dict, debug=True)
		test_seqs, test_label_array = nodes_to_seqs_and_labels(nodes=test_nodes, index_to_seq=index_to_seq, label_dict=label_dict, debug=True)
		if graphsage_embedding == 1:
			assert len(train_seqs) == graphsage_train_embeddings.shape[0]
			assert len(test_seqs) == graphsage_test_embeddings.shape[0]

		fold_result = {
			"fold_id": fold_id,
			"train_nodes": train_nodes,
			"test_nodes": test_nodes,

			"train_seqs": train_seqs,
			"test_seqs": test_seqs,

			"train_labels": train_label_array,
			"test_labels": test_label_array,

			"train_rna_embeddings": train_features,
			"test_rna_embeddings": test_features,

			"train_graph_embeddings": graphsage_train_embeddings,
			"test_graph_embeddings": graphsage_test_embeddings,
		}
		all_folds_results.append(fold_result)

	return all_folds_results

def get_benchmark_validation_results(K_folds_results=None, epochs=30, b_sz=128, seed=64, graphsage_embedding=0, result_path='./'):

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
		X_train_rna = fold_data["train_rna_embeddings"]
		X_train_graph = fold_data["train_graph_embeddings"]
		y_train = fold_data["train_labels"]
		seq_test = fold_data["test_seqs"]
		X_test_rna = fold_data["test_rna_embeddings"]
		X_test_graph = fold_data["test_graph_embeddings"]
		y_test = fold_data["test_labels"]

		train_df = pd.DataFrame({"RNA_sequence": seq_train,"label": y_train})
		train_df["rna_embedding"] = list(X_train_rna)
		#train_df["graph_embedding"] = list(X_train_graph)

		test_df = pd.DataFrame({"RNA_sequence": seq_test,"label": y_test})
		test_df["rna_embedding"] = list(X_test_rna)
		#test_df["graph_embedding"] = list(X_test_graph)

		if graphsage_embedding == 1:
			train_df["graph_embedding"] = list(X_train_graph)
			test_df["graph_embedding"] = list(X_test_graph)

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
		#(X_train_rna_sub, X_val_rna, X_train_graph_sub, X_val_graph, y_train_sub, y_val) = train_test_split(X_train_rna, X_train_graph, y_train, test_size=0.2,stratify=y_train, random_state=42)
		if graphsage_embedding == 1:
			(X_train_rna_sub, X_val_rna, X_train_graph_sub, X_val_graph, y_train_sub, y_val) = train_test_split(X_train_rna, X_train_graph, y_train, test_size=0.2, stratify=y_train, random_state=42)
			train_dataset = TensorDataset(torch.tensor(X_train_rna_sub, dtype=torch.float32), torch.tensor(X_train_graph_sub, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.float32))
			validation_dataset = TensorDataset(torch.tensor(X_val_rna, dtype=torch.float32), torch.tensor(X_val_graph, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
		elif graphsage_embedding == 0:
			(X_train_rna_sub, X_val_rna, y_train_sub, y_val) = train_test_split(X_train_rna, y_train, test_size=0.2, stratify=y_train, random_state=42)
			X_train_graph_sub = None
			X_val_graph = None
			train_dataset = TensorDataset(torch.tensor(X_train_rna_sub, dtype=torch.float32), torch.tensor(y_train_sub, dtype=torch.float32))
			validation_dataset = TensorDataset(torch.tensor(X_val_rna, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
		else:
			raise ValueError("graphsage_embedding 必须为 0 或 1")

		train_loader = DataLoader(train_dataset, batch_size=b_sz, shuffle=True)
		validation_loader = DataLoader(validation_dataset, batch_size=b_sz, shuffle=False)

		## Build the VecNN model
		model = VecNN(rna_embed_len=640, graphsage_embed_len=100, graphsage_embedding=graphsage_embedding).to(device)

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
			epoch_loss, val_loss, val_auc = apply_VecNN_model(train_loader=train_loader, validation_loader=validation_loader, VecNN_model=model, device=device, criterion=criterion, optimizer=optimizer, graphsage_embedding=graphsage_embedding)
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
			if graphsage_embedding == 1:
				for val_rna, val_graph, val_y in validation_loader:
					val_rna = val_rna.to(device)
					val_graph = val_graph.to(device)

					pred = model(val_rna, val_graph).squeeze(-1)

					val_probs.extend(pred.cpu().numpy())
					val_labels.extend(val_y.cpu().numpy())
			elif graphsage_embedding == 0:
				for val_rna, val_y in validation_loader:
					val_rna = val_rna.to(device)
					pred = model(val_rna, None).squeeze(-1)
					val_probs.extend(pred.cpu().numpy())
					val_labels.extend(val_y.cpu().numpy())
			else:
				raise ValueError("graphsage_embedding 必须为 0 或 1")

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
		if graphsage_embedding == 1:
			test_dataset = TensorDataset(torch.tensor(X_test_rna, dtype=torch.float32), torch.tensor(X_test_graph, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
		elif graphsage_embedding == 0:
			test_dataset = TensorDataset(torch.tensor(X_test_rna, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
		else:
			raise ValueError("graphsage_embedding 必须为 0 或 1")
			
		test_loader = DataLoader(test_dataset, batch_size=b_sz, shuffle=False)

		test_probs = []
		test_labels = []
		with torch.no_grad():
			if graphsage_embedding == 1:
				for test_rna, test_graph, test_y in test_loader:
					test_rna = test_rna.to(device)
					test_graph = test_graph.to(device)
					pred = model(test_rna, test_graph).squeeze(-1)
					test_probs.extend(pred.cpu().numpy())
					test_labels.extend(test_y.cpu().numpy())
			elif graphsage_embedding == 0:
				for test_rna, test_y in test_loader:
					test_rna = test_rna.to(device)
					pred = model(test_rna, None).squeeze(-1)
					test_probs.extend(pred.cpu().numpy())
					test_labels.extend(test_y.cpu().numpy())
			else:
				raise ValueError("graphsage_embedding 必须为 0 或 1")

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
		pr_df = pd.DataFrame({"precision": precision_curve, "recall": recall_curve, "threshold": np.append(pr_thresholds, np.nan)})
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
	graphSage = torch.load(model_path, map_location=device, weights_only=False)
	# print(graphSage)
	graphSage.to(device)
	graphSage.eval()
	with torch.no_grad():
		embs = get_gnn_embeddings(graphSage, nodes)
	# print(f'Output embs: {embs}')
	# print(f'Output embs shape: {embs.shape}')
	graphsage_embeddings = embs.cpu().numpy()

	return graphsage_embeddings

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


def get_unseen_graphsage_embeddings(config='./graphsage_src/experiments.conf', model_path=None, model_dataset=None, unseen_dataset=None, graphsage_embedding=0):
	'''
	获取未见过节点的网络上的embeddings
	'''
	cuda = True
	device = torch.device("cuda" if cuda else "cpu") ##设置 GPU
	# print('DEVICE:', device)
	config = pyhocon.ConfigFactory.parse_file(config) ##读取 config
	ds = model_dataset ##训练的数据集'NPInter2'
	dataCenter = DataCenter(config) ##创建训练数据集'NPInter2'的 DataCenter

	if graphsage_embedding == 1:
		### 输入模型
		#graphsage = torch.load(model_path) ##传入已经训练好的 GraphSAGE 模型
		graphsage = torch.load(model_path, map_location=device, weights_only=False)
		graphsage.to(device) ####把 GraphSAGE 放到 GPU

		### 导入benchmark数据集
		node_map, feat_data = graphsage.dataCenter.load_Graphsage_pretrain_dataSet(model_dataset) ##加载 GraphSAGE 原训练集的信息
		feat_data_list = feat_data.tolist() ##feat_data_list 是benchmark训练集节点 feature

		dataCenter.load_Graphsage_pretrain_dataSet(ds) #运行dataCenter，再次加载训练数据集
		graphsage.dataCenter = dataCenter ##把当前这个 GraphSAGE 模型对象内部的 dataCenter 指针重新绑定到一个新的 DataCenter 实例，本质一致，但是为了加载unseen的数据
		unseen_indexs, feat_data, unseen_feat_data, unseen_label_df, node_map = graphsage.dataCenter.load_unseen_dataSet(unseen_dataset, graphsage.adj_lists, node_map, feat_data_list, graphsage_embedding=1)

		raw_features = torch.FloatTensor(feat_data).to(device) ##把训练集 + unseen 的 feature 转成 Tensor
		graphsage.raw_features = raw_features ##把这些 feature 放进 GraphSAGE 模型

		graphsage.to(device) ##把整个模型放到 GPU
		graphsage.eval() ##GraphSAGE 进入推理模式，而不是训练模式
		embs = get_unseen_gnn_embeddings(graphsage, graphsage.dataCenter, unseen_dataset, graphsage.adj_lists)

		graphsage_embeddings = embs.cpu().numpy() ##转成 NumPy
		print("GraphSAGE embeddings length:", len(graphsage_embeddings))
		assert graphsage_embeddings.shape[0] == unseen_feat_data.shape[0]
		assert graphsage_embeddings.shape[1] == 100
	elif graphsage_embedding == 0:
		### 不加载 GraphSAGE 模型，只加载 RNA-FM + HAT embeddings
		unseen_indexs, feat_data, unseen_feat_data, unseen_label_df, node_map = dataCenter.load_unseen_dataSet(unseen_dataset, graphsage_embedding=0)
		graphsage_embeddings = None
	else:
		raise ValueError("graphsage_embedding 必须为 0 或 1")

	index_to_seq = {v: k for k, v in node_map.items()}
	unseen_label_dict = dict(zip(unseen_label_df.iloc[:,0], unseen_label_df.iloc[:,1]))
	assert unseen_feat_data.shape[1] == 640
	#unseen_graphsage_rnas_embeddings = np.concatenate((graphsage_embeddings, unseen_feat_data), axis=1) ##unseen RNA的LLM的emb和graphsage的embeddings结合起来
	return unseen_feat_data, graphsage_embeddings, unseen_indexs, index_to_seq, unseen_label_dict


def get_TheNovel_unseen_results(config='./graphsage_src/experiments.conf', graphsage_model_path=None, model_out_dir=None, model_dataset=None, unseen_dataset=None, result_path=None, b_sz=128, graphsage_embedding=0):

	if model_dataset == None: ##如果没有指定 model_dataset，程序直接退出。
		sys.exit("请输入正确的数据集！")
	os.makedirs(result_path, exist_ok=True)
	metrics_file = os.path.join(model_out_dir, "5fold_metrics_results.csv")
	metrics_df = pd.read_csv(metrics_file, sep="\t")

	all_unseen_auc = [] ##准备保存每一个 Run 的指标
	all_unseen_auprc = []
	all_unseen_accuracy = []
	all_unseen_precision = []
	all_unseen_recall = []
	all_unseen_mcc = []
	all_unseen_spe = []
	run_results = []

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ##选择 GPU 或 CPU

	for run in range(5): ##循环5个Run

		print(20*"——") ##打印当前Run
		print(f"Run_{run}")
		print(20*"——")

		run_dir = os.path.join(result_path,f"run_{run}")
		os.makedirs(run_dir, exist_ok=True)

		### 获取graphsage的嵌入
		#graphsage_model_path = graphsage_path+'/Run_' + str(run) + '/graphSage.pth' ##找到当前Run的GraphSAGE模型
		unseen_rna_embeddings, unseen_graphsage_embeddings, unseen_indexs, index_to_seq, unseen_label_dict = get_unseen_graphsage_embeddings(config=config, model_path=graphsage_model_path, model_dataset=model_dataset, unseen_dataset=unseen_dataset, graphsage_embedding=graphsage_embedding) ##获取 unseen 节点的 GraphSAGE embedding 和 LLM的embs结合的特征
		unseen_seqs, unseen_label_array = nodes_to_seqs_and_labels(nodes=unseen_indexs, index_to_seq=index_to_seq, label_dict=unseen_label_dict, debug=True)

		if graphsage_embedding == 1:
			if not (len(unseen_seqs) == len(unseen_label_array) == unseen_rna_embeddings.shape[0] == unseen_graphsage_embeddings.shape[0]):
				raise ValueError(f"Length mismatch: "f"unseen_seqs={len(unseen_seqs)}, "f"unseen_label_array={len(unseen_label_array)}, "f"rna_embeddings={unseen_rna_embeddings.shape[0]}, "f"graphsage_embeddings={unseen_graphsage_embeddings.shape[0]}")
		elif graphsage_embedding == 0:
			if not (len(unseen_seqs) == len(unseen_label_array) == unseen_rna_embeddings.shape[0]):
				raise ValueError("Unseen RNA embeddings length mismatch")
		else:
			raise ValueError("graphsage_embedding 必须为 0 或 1")

		best_model_path = os.path.join(model_out_dir, 'fold_' + str(run), 'weight', f"VecNN_best_fold_{run}.pth") ##加载当前 Run 的 VecNN 模型

		# 加载模型
		#best_vecnet_model = torch.load(best_model_path)
		best_vecnet_model = torch.load(best_model_path, map_location=device, weights_only=False)
		best_vecnet_model.to(device)
		best_vecnet_model.eval() # 确保模型处于评估模式（不进行梯度计算）

		if graphsage_embedding == 1:
			unseen_tensor_dataset = TensorDataset(torch.tensor(unseen_rna_embeddings, dtype=torch.float32), torch.tensor(unseen_graphsage_embeddings, dtype=torch.float32), torch.tensor(unseen_label_array, dtype=torch.float32))

		elif graphsage_embedding == 0:

			unseen_tensor_dataset = TensorDataset(torch.tensor(unseen_rna_embeddings, dtype=torch.float32), torch.tensor(unseen_label_array, dtype=torch.float32))
		else:
			raise ValueError("graphsage_embedding 必须为 0 或 1")

		unseen_loader = DataLoader(unseen_tensor_dataset, batch_size=b_sz, shuffle=False)

		unseen_probs = []
		unseen_labels = []
		with torch.no_grad():
			if graphsage_embedding == 1:
				for (unseen_rna_x, unseen_graph_x, unseen_y) in unseen_loader:
					unseen_rna_x = unseen_rna_x.to(device)
					unseen_graph_x = unseen_graph_x.to(device)
					pred = best_vecnet_model(unseen_rna_x, unseen_graph_x).squeeze(-1)
					unseen_probs.extend(pred.cpu().numpy())
					unseen_labels.extend(unseen_y.cpu().numpy())
			elif graphsage_embedding == 0:
				for unseen_rna_x, unseen_y in unseen_loader:
					unseen_rna_x = unseen_rna_x.to(device)
					pred = best_vecnet_model(unseen_rna_x, None).squeeze(-1)
					unseen_probs.extend(pred.cpu().numpy())
					unseen_labels.extend(unseen_y.cpu().numpy())

		unseen_probs = np.array(unseen_probs).reshape(-1, 1)
		unseen_labels = np.array(unseen_labels).reshape(-1, 1)
		#unseen_labels_T = np.reshape(unseen_labels, (len(unseen_labels), 1)) ##处理真实标签, 使用reshape将其转置为列向量
		print("unseen_probs shape:", unseen_probs.shape)
		print("unseen_labels shape:", unseen_labels.shape)

		unseen_predictions_actual = np.concatenate((unseen_probs, unseen_labels), axis=1) ##保存预测结果和真实标签,使用concatenate函数合并列
		# print(Y_test_predictions_actual)
		unseen_predictions_actual_path_file = os.path.join(run_dir, f"UnseenNodeTopPrediction_{run}.csv")
		np.savetxt(unseen_predictions_actual_path_file, unseen_predictions_actual, delimiter=',')
		print(f"Saved unseen predictions: {unseen_predictions_actual_path_file}")

		## unseen集的指标
		best_cutoff = metrics_df.loc[metrics_df["fold"] == run, "best_cutoff"].iloc[0]
		print(f"Run_{run}: best_cutoff = {best_cutoff}")

		unseen_auc = roc_auc_score(unseen_labels, unseen_probs) # 计算 AUROC
		unseen_auprc = average_precision_score(unseen_labels, unseen_probs)
		unseen_pred_labels = (unseen_probs > best_cutoff).astype(int)
		unseen_acc = accuracy_score(unseen_labels, unseen_pred_labels)
		unseen_precision = precision_score(unseen_labels, unseen_pred_labels, zero_division=0) # 计算precision值
		unseen_recall = recall_score(unseen_labels, unseen_pred_labels, zero_division=0) ### 计算recall值，所有真实正样本被验证出的概率
		unseen_mcc = matthews_corrcoef(unseen_labels, unseen_pred_labels) ###计算MCC值，模型预测标签和真实标签之间的一致性
		cm = confusion_matrix(unseen_labels, unseen_pred_labels, labels=[0,1]) # 计算混淆矩阵
		tn, fp, fn, tp = cm.ravel() # 提取混淆矩阵的值
		unseen_spe = tn / (tn + fp) if (tn + fp) > 0 else 0 # 计算 Specificity

		all_unseen_auc.append(unseen_auc)
		all_unseen_auprc.append(unseen_auprc)
		all_unseen_accuracy.append(unseen_acc)
		all_unseen_precision.append(unseen_precision)
		all_unseen_recall.append(unseen_recall)
		all_unseen_mcc.append(unseen_mcc)
		all_unseen_spe.append(unseen_spe)
		run_results.append({
			"run": run,
			"unseen_auc": unseen_auc,
			"unseen_auprc": unseen_auprc,
			"unseen_accuracy": unseen_acc,
			"unseen_precision": unseen_precision,
			"unseen_recall": unseen_recall,
			"unseen_mcc": unseen_mcc,
			"unseen_specificity": unseen_spe,
			"tn": tn,
			"fp": fp,
			"fn": fn,
			"tp": tp
		})

		# ========================= ROC curve =========================
		fpr, tpr, roc_thresholds = roc_curve(unseen_labels, unseen_probs)
		roc_df = pd.DataFrame({"fpr": fpr,"tpr": tpr,"threshold": roc_thresholds})
		roc_path = os.path.join(run_dir, "roc_curve.csv")
		roc_df.to_csv(roc_path, index=False)

		# ========================= PR curve =========================
		precision_curve, recall_curve, pr_thresholds = precision_recall_curve(unseen_labels, unseen_probs)
		pr_df = pd.DataFrame({"precision": precision_curve,"recall": recall_curve})
		pr_path = os.path.join(run_dir, "pr_curve.csv")
		pr_df.to_csv(pr_path, index=False)

		##打印当前fold结果
		print(f"Run {run} | " f"AUC={unseen_auc:.4f} | " f"AUPRC={unseen_auprc:.4f} | " f"ACC={unseen_acc:.4f} | " f"Precision={unseen_precision:.4f} | " f"Recall={unseen_recall:.4f} | " f"Specificity={unseen_spe:.4f} | " f"MCC={unseen_mcc:.4f}")

	results_df = pd.DataFrame(run_results)
	results_df.to_csv(os.path.join(result_path, "5run_metrics_results.csv"), index=False, sep='\t')
	print(f"Results saved to "f"{os.path.join(result_path, '5run_metrics_results.csv')}")

	summary_mean_df = pd.DataFrame({"Metric":["AUC","AUPRC","ACC","Precision","Recall","Specificity","MCC"],
									"Mean":[np.mean(all_unseen_auc),np.mean(all_unseen_auprc),np.mean(all_unseen_accuracy),np.mean(all_unseen_precision),np.mean(all_unseen_recall),np.mean(all_unseen_spe),np.mean(all_unseen_mcc)],
									"Std":[np.std(all_unseen_auc),np.std(all_unseen_auprc),np.std(all_unseen_accuracy),np.std(all_unseen_precision),np.std(all_unseen_recall),np.std(all_unseen_spe),np.std(all_unseen_mcc)]})
	summary_mean_df.to_csv(os.path.join(result_path, "5run_summary.csv"), index=False, sep='\t')


def main(DataSet, Unseen_DataSet, Config, VecNN_model_dir, Save_dir, Graphsage_model_dir, data_type="pretrain_graphsage", seed=64, graphsage_embedding=0):
	
	if data_type == "pretrain_graphsage":
		run_graphsage_experiment(dataSet=DataSet,agg_func='MAX',epochs=50,b_sz=128,seed=64,cuda=True,gcn=False,learn_method='unsup',unsup_loss='margin',max_vali_f1=0,name='debug',config=Config,save_dir=Save_dir)

	elif data_type == "benchmark":
		K_folds_results = get_benchmark_graphsage_embeddings(dataSet=DataSet,seed=seed,config=Config,graphsage_model_path=Graphsage_model_dir,graphsage_embedding=graphsage_embedding)
		get_benchmark_validation_results(K_folds_results=K_folds_results,epochs=50,b_sz=128,seed=seed,graphsage_embedding=graphsage_embedding,result_path=Save_dir)

	elif data_type == "unseen":
		get_TheNovel_unseen_results(config=Config, graphsage_model_path=Graphsage_model_dir, model_out_dir=VecNN_model_dir, model_dataset=DataSet, unseen_dataset=Unseen_DataSet, result_path=Save_dir, b_sz=128, graphsage_embedding=graphsage_embedding)

	else:
		raise ValueError(f"Unknown data_type: {data_type}. ""Please use 'benchmark' or 'unseen'.")

	print("========== Pipeline Finished ==========")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="GraphSAGE + VecNN RNA classification pipeline")
	parser.add_argument("--DataSet", type=str, required=True, help="The name of the dataset to be trained")
	parser.add_argument("--Unseen_DataSet", type=str, required=False, help="The unseen dataset name")
	parser.add_argument("--Config", type=str, required=True, help="Input configuration file")
	parser.add_argument("--VecNN_model_dir", type=str, required=False, help="Directory containing VecNN models")
	parser.add_argument("--Save_dir", type=str, required=True, help="Output directory")
	parser.add_argument("--Graphsage_model_dir", type=str, required=False, help="Directory containing the GraphSAGE model")
	parser.add_argument("--data_type", type=str, required=True, help="Dataset type: benchmark or unseen")
	args = parser.parse_args()

	main(
		DataSet=args.DataSet,
		Unseen_DataSet=args.Unseen_DataSet,
		Config=args.Config,
		VecNN_model_dir=args.VecNN_model_dir,
		Save_dir=args.Save_dir,
		Graphsage_model_dir=args.Graphsage_model_dir,
		data_type=args.data_type
	)
