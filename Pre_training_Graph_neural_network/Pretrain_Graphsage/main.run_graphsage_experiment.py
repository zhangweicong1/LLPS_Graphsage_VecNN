import os
import sys
import argparse
from datetime import datetime
import torch
import random
import numpy as np
import pyhocon
from torch.utils.tensorboard import SummaryWriter

from my_datacenter import *
from my_models_GraphSage import *
from my_utils import *



def run_graphsage_experiment(dataSet=None, agg_func='MEAN', epochs=3, b_sz=20, seed=64, cuda=False, gcn=False, learn_method='unsup', unsup_loss='margin', max_vali_f1=0, name='debug', config='./graphsage_src/experiments.conf',embedding_type=None, save_dir=None):

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
		graphSage, epoch_loss = apply_model(dataCenter=dataCenter, ds=ds, graphSage=graphSage, unsupervised_loss=unsupervised_loss, b_sz=b_sz, unsup_loss=unsup_loss, device=device, learn_method=learn_method, writer=writer, epoch=epoch)
		
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			best_epoch = epoch
			print(f"New best model at epoch {epoch + 1}, loss = {best_loss:.4f}")
			weight_path = os.path.join(weight_dir,"Pretrain_GraphSage_best_epoch.pth")
			#weight_path = '{}/Pretrain_GraphSage_epoch{}.pth'.format(save_dir, epoch + 1)
			torch.save(graphSage,weight_path)
			#torch.save(graphSage.state_dict(),weight_path)

	writer.close()

def main(DataSet, Config, Save_dir):
	run_graphsage_experiment(dataSet=DataSet,agg_func='MEAN',epochs=50,b_sz=128,seed=64,cuda=True,gcn=False,learn_method='unsup',unsup_loss='margin',max_vali_f1=0,name='debug',config=Config,embedding_type=None,save_dir=Save_dir)
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Pre-training of GraphSage models")
	parser.add_argument("--DataSet", type=str, required=True, help="The name of the dataset to be pre-trained")
	parser.add_argument("--Config", type=str, required=True, help="Input configuration file")
	parser.add_argument("--Save_dir", type=str, required=True, help="Output directory")
	args = parser.parse_args()

	main(args.DataSet, args.Config, args.Save_dir)
