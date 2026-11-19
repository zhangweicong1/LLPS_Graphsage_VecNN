#!/bin/bash
#$ -S /bin/bash
#$ -cwd 
#$ -pe def_slot 1
#$ -l l4=1,s_vmem=512G
#$ -o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/run_unseen.o.txt
#$ -e /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/run_unseen.e.txt

#qsub run_modules.sh
#tensorboard --logdir=20260104_154702

date
cd /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/home/isou/Software/miniconda3/envs/pytorchenvs/bin/python -u /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Graphsage_VecNN/MLP_Fusion/main.py \
	--DataSet LLPS_benchmark \
	--Unseen_DataSet LLPS_unseen \
	--Config /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Graphsage_VecNN/MLP_Fusion/experiments.conf \
	--VecNN_model_dir /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/LLPS_Benchmark/MLP_Fusion/graphsage_embedding0 \
	--Save_dir /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/LLPS_Unseen/MLP_Fusion/graphsage_embedding0 \
	--data_type unseen
	# --Graphsage_model_dir /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Pretrain_Graphsage/MAX_agg_func/LLPS_benchmark_WALK_LEN2_N_WALK_LEN4/weight/Pretrain_GraphSage_best_epoch.pth \
date
