#!/bin/bash
#$ -S /bin/bash
#$ -cwd 
#$ -pe def_slot 1
#$ -l l4=1,s_vmem=512G
#$ -o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/run_Pretrain_Graphsage.o.txt
#$ -e /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/run_Pretrain_Graphsage.e.txt

#qsub run_Pretrain_Graphsage.sh
#tensorboard --logdir=20260104_154702

date
cd /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/home/isou/Software/miniconda3/envs/pytorchenvs/bin/python -u /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Graphsage_VecNN/main.py --DataSet LLPS_benchmark --Config /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Graphsage_VecNN/experiments.conf --Save_dir /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/Graphsage_MLP/Pretrain_Graphsage --data_type pretrain_graphsage
date
