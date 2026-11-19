import pandas as pd
import random
from collections import defaultdict

# ============================================================
# 参数
# ============================================================
interaction_file = "/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/Filter_sequence_similarity_interactions_seq_list.txt"
unseen_ratio = 0.2
random_seed = 42
random.seed(random_seed)

# ============================================================
# 1. 读取 RNA-Protein interaction
# ============================================================
interactions = []
with open(interaction_file, "r") as f:
    for line in f:
        line = line.strip()
        rna, protein = line.split()
        interactions.append((rna, protein))

# ============================================================
# 2. 构建 RNA -> Protein 邻接关系
# ============================================================
rna_to_proteins = defaultdict(set)
for rna, protein in interactions:
    rna_to_proteins[rna].add(protein)

all_rnas = list(rna_to_proteins.keys())
print("Total RNA:", len(all_rnas))
target_unseen_size = int(len(all_rnas) * unseen_ratio)
print("Target unseen RNA size:", target_unseen_size)

# ============================================================
# 3. 随机初始化 Benchmark / unseen
# ============================================================
random.shuffle(all_rnas)
unseen_rnas = set(all_rnas[:target_unseen_size])
benchmark_rnas = set(all_rnas[target_unseen_size:])

# ============================================================
# 4. 获取 Benchmark RNA 已覆盖的 Protein
# ============================================================
def get_protein_set(rna_set):
    proteins = set()
    for rna in rna_set:
        proteins.update(rna_to_proteins[rna])
    return proteins

benchmark_proteins = get_protein_set(benchmark_rnas)
unseen_proteins = get_protein_set(unseen_rnas)
print("Initial Benchmark proteins:", len(benchmark_proteins))
print("Initial Unseen proteins:", len(unseen_proteins))

# ============================================================
# 5. 检查 Unseen RNA 是否连接到 Benchmark 中不存在的 Protein
# ============================================================
invalid_unseen_rnas = []
for rna in unseen_rnas:
    rna_proteins = rna_to_proteins[rna]
    # 如果该 RNA 存在任何 protein
    # 没有被 Benchmark 覆盖
    if not rna_proteins.issubset(benchmark_proteins):
        invalid_unseen_rnas.append(rna)
print("Invalid unseen RNAs:", len(invalid_unseen_rnas))

# ============================================================
# 6. 把不满足条件的 Unseen RNA 移回 Benchmark
# ============================================================
for rna in invalid_unseen_rnas:
    unseen_rnas.remove(rna)
    benchmark_rnas.add(rna)

# ============================================================
# 7. 尝试从 Benchmark 中补充新的 RNA 到 Unseen,条件：把 candidate 移到 Unsenn 后，candidate 的所有 Protein 仍然至少存在于 Benchmark 中
# ============================================================
remaining_benchmark_rnas = list(benchmark_rnas)
random.shuffle(remaining_benchmark_rnas)

def get_protein_degree(rna_set):
    protein_count = defaultdict(int)
    for rna in rna_set:
        for protein in rna_to_proteins[rna]:
            protein_count[protein] += 1
    return protein_count

protein_degree = get_protein_degree(benchmark_rnas) # 计算 Benchmark 中每个 Protein 被多少个 RNA 连接的字典
for rna in remaining_benchmark_rnas:
    if len(unseen_rnas) >= target_unseen_size: ##如果达到目标数量
        break ##立即结束当前的 for 循环，不再继续遍历后面的 RNA
    proteins = rna_to_proteins[rna]

    # --------------------------------------------------------
    # 判断：如果把这个 RNA 从 Benchmark 移走，它连接的每个 Protein 是否仍然至少被另一个 Benchmark RNA 连接
    # --------------------------------------------------------
    can_move = True
    for protein in proteins:
        if protein_degree[protein] <= 1:
            can_move = False
            break
    if can_move:
        benchmark_rnas.remove(rna) # 移动 RNA
        unseen_rnas.add(rna)
        for protein in proteins: # 更新 Protein degree
            protein_degree[protein] -= 1


# ============================================================
# 8. 最终检查
# ============================================================
final_benchmark_proteins = get_protein_set(benchmark_rnas)
final_unseen_proteins = get_protein_set(unseen_rnas)

print("Benchmark RNA:", len(benchmark_rnas))
print("Unseen RNA:", len(unseen_rnas))
print("Benchmark Protein:", len(final_benchmark_proteins))
print("Unseen Protein:", len(final_unseen_proteins))

print("Unseen proteins subset of Benchmark proteins:",final_unseen_proteins.issubset(final_benchmark_proteins))
assert final_unseen_proteins.issubset(final_benchmark_proteins)
print("Actual unseen ratio:",len(unseen_rnas) / len(all_rnas))

# ============================================================
# 9. 保存 RNA 列表
# ============================================================
pd.DataFrame(sorted(benchmark_rnas)).to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/benchmark_rna.txt", sep="\t", index=False, header=False)
pd.DataFrame(sorted(unseen_rnas)).to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/unseen_rna.txt", sep="\t", index=False, header=False)

# ============================================================
# 10. 保存 interaction
# ============================================================
benchmark_interactions = []
unseen_interactions = []

for rna, protein in interactions:
    if rna in benchmark_rnas:
        benchmark_interactions.append((rna, protein))
    elif rna in unseen_rnas:
        unseen_interactions.append((rna, protein))

pd.DataFrame(benchmark_interactions).to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/benchmark_interactions.txt", sep="\t", index=False, header=False)
pd.DataFrame(unseen_interactions).to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/unseen_interactions.txt", sep="\t", index=False, header=False)


###split label文件
benchmark_rna_file = "/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/benchmark_rna.txt"
unseen_rna_file = "/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/unseen_rna.txt"
label_file = "/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/Filter_sequence_similarity_RNA_labels.txt"

with open(benchmark_rna_file, "r") as f:
    benchmark_rnas = set(line.strip() for line in f if line.strip())

with open(unseen_rna_file, "r") as f:
    unseen_rnas = set(line.strip() for line in f if line.strip())

label_df = pd.read_csv(label_file, sep="\t", header=None)

benchmark_label_df = label_df[label_df.iloc[:, 0].isin(benchmark_rnas)]
unseen_label_df = label_df[label_df.iloc[:, 0].isin(unseen_rnas)]

benchmark_label_df.to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/LLPS_benchmark_label.txt", sep="\t", index=False, header=False)
unseen_label_df.to_csv("/home/isou/lncRNA/LLPS/DLM/LLPS_DLM/Graphsage/GraphSAGE_Transferability/split_dataset_unseen.dataset/Filter_sequence_similarity/LLPS_unseen_label.txt", sep="\t", index=False, header=False)

def print_label_distribution(name, df):
    counts = df.iloc[:, 1].value_counts()
    positive = counts.get(1, 0)
    negative = counts.get(0, 0)
    total = positive + negative

    print(f"\n{name}")
    print("Total:", total)
    print("Positive:", positive)
    print("Negative:", negative)
    print("Positive ratio:", round(positive / total * 100, 2), "%")
    print("Negative ratio:", round(negative / total * 100, 2), "%")

print_label_distribution("Benchmark", benchmark_label_df)
print_label_distribution("Unseen", unseen_label_df)

# Check whether any labels were omitted
print(
    "\nUnassigned labeled RNAs:",
    len(label_df) - len(benchmark_label_df) - len(unseen_label_df)
)
