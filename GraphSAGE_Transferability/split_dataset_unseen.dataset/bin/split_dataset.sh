#!/bin/bash
#$ -S /bin/bash
#$ -cwd 
#$ -pe def_slot 4
#$ -N split_dataset
#$ -l s_vmem=16G
#$ -o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/bin/split_dataset.o.txt
#$ -e /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/bin/split_dataset.e.txt

# export OMP_NUM_THREADS=8
# export MKL_NUM_THREADS=8
# export OPENBLAS_NUM_THREADS=8
# export NUMEXPR_NUM_THREADS=8
# export TORCH_NUM_THREADS=8

date
cd /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/bin
/home/isou/Software/miniconda3/envs/pytorchenvs/bin/python /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/bin/Filter_sequence_similarity.split_dataset.py
date
