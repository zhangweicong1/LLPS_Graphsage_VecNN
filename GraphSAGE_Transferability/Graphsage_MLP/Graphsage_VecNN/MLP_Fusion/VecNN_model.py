import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F

class VecNN(nn.Module):
    def __init__(self, rna_embed_len=640, graphsage_embed_len=100, rna_hidden_dim=256, graph_hidden_dim=32, graphsage_embedding=1):
        super(VecNN, self).__init__()
        
        self.rna_layer = nn.Sequential(
            #nn.LayerNorm(rna_embed_len),
            nn.Linear(rna_embed_len, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, rna_hidden_dim),
            nn.ReLU(),
        )
        self.graphsage_embedding = graphsage_embedding

        if graphsage_embedding == 1:
            self.graph_layer = nn.Sequential(
                nn.Linear(graphsage_embed_len, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, graph_hidden_dim),
                nn.ReLU(),
            )
            fusion_input_dim = rna_hidden_dim + graph_hidden_dim

        if graphsage_embedding == 0:
            fusion_input_dim = rna_hidden_dim

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, rnas_input, graphsage_input):
        rna_embedding = self.rna_layer(rnas_input)
        if self.graphsage_embedding == 1:
            graph_embedding = self.graph_layer(graphsage_input)
            combined = torch.cat((rna_embedding, graph_embedding), dim=1)

        else:
            combined = rna_embedding
        fused_embedding = self.fusion_layer(combined)
        output = self.output_layer(fused_embedding)

        return output
