#!/bin/bash
#$ -S /bin/bash
#$ -cwd 
#$ -pe def_slot 4
#$ -l l4=1,s_vmem=64G
#$ -o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/HAT_result/runhatFM.o.txt
#$ -e /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/HAT_result/runhatFM.e.txt

#qsub GPU.get_rnafm_embeddings.sh
#tensorboard --logdir=20260104_154702

date
cd /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/HAT_result
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
/home/isou/Software/miniconda3/envs/pytorchenvs/bin/python /hshare1/ZETTAI_path_WA_slash_home_KARA/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/bin/hierarchical-transformers/models/get_hat_rnafm_embeddings.py\
 -i /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/data/unseen_rna_AUCG.txt\
 -o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/HAT_result/unseen_hat_rnafm_embeddings.pkl

 #-i /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/data/train_rna_AUCG.txt\
 #-o /home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/LLMs_embedding/RNA-FM/HAT_result/train_hat_rnafm_embeddings.pkl
date
