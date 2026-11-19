import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn
#from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
#from torch.nn.functional import normalize

from transformers.modeling_outputs import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaIntermediate, RobertaOutput
#from transformers.activations import gelu
import numpy as np
from transformers import RobertaConfig


@dataclass ##返回所有的Sentence的Attention的输出, 整个序列经过HAT后的所有Token表示
class BaseModelOutputWithSentenceAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    sentence_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass ##基于BaseModelOutputWithSentenceAttentions结果后的Pooling操作
class SequenceRepresentationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    representations: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HATLayer(nn.Module): ##继承的是nn.Module，定义一个新的神经网络层
    def __init__(self, config):
        super().__init__()
        self.document_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(config.max_sentences+1, config.hidden_size) ##这里的position_embeddings是构建一个 Position Embedding 层（Embedding Layer），初始值是随机的，这里第0行是position_embeddings（Embedding Layer 本质上就是维护一张可以训练的矩阵（Embedding Table）），也就是建立一张Position Embedding Table（max_sentences+1 * hidden_size）用来区分每一个sentence
    def forward(
        self,
        hidden_states, ##上一层Encoder输出，对于第一层 HATLayer来说：hidden_states = HATEmbeddings 的输出
        output_attentions=False, ##模型在前向传播时,除了返回主要的隐藏状态(hidden_states)之外,要不要额外返回每一层计算出来的注意力权重矩阵(attention weight)
    ):
        document_outputs = (None, None) ##初始化一个变量，用来保存 Document Encoder 的输出
        sentence_global_tokens = hidden_states ##直接使用 Mean Pooling 后的 Segment embeddings (1, num_segments, 640)
        num_segments = hidden_states.size(1)
        sentence_positions = torch.arange(1, num_segments + 1, device=hidden_states.device).unsqueeze(0)
        sentence_global_tokens = sentence_global_tokens + self.position_embeddings(sentence_positions) ##把sentence_global_tokens的每个Segment的token和每个Segment的Position相加，让后面的Document Encoder知道segment的顺序
        document_outputs = self.document_encoder(sentence_global_tokens, output_attentions=output_attentions) ##把多个segment进行transformer，输入数据维度：(batch,num_segments,640)
        hidden_states = document_outputs[0] ##返回更新后的 Segment embeddings
        if output_attentions: ##如果用户需要Attention矩阵,返回Document Attention
            return hidden_states, document_outputs[1]
        return hidden_states, None ##否则只返回outputs(B, num_segments, 640)


class TransformerLayer(nn.Module): ##标准的Transformer Encoder Layer，没有任何HAT特有的东西，可以直接用nn.TransformerEncoderLayer代替
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder ##记录当前模块是否配置为 Decoder
        self.intermediate = RobertaIntermediate(config) ##初始化 Transformer 中的前馈神经网络（Feed-Forward Network，FFN）的中间层
        self.output = RobertaOutput(config) ##初始化 FFN 的输出模块
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0] ##得到最终的attention_output，shape:(256,128,768)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights，除非output_attentions设置为True，提取可选的 Attention 权重
        intermediate_output = self.intermediate(attention_output) ##Transformer FFN层，intermediate_output(256,128,3072)，每个token从768维扩展到3072维。为什么扩大？因为FFN负责：非线性特征转换，Attention负责：token之间的信息交流，FFN负责：每个token自己的feature transformation
        layer_output = self.output(intermediate_output, attention_output) ##FFN第二部分(Linear,Dropout,Residual,LayerNorm),输入：intermediate_output(256,128,3072),先恢复维度:(256,128,768),Residual connection加上原来的attention_output(output=FFN(x)+x),最终layer_output shape:(256,128,768)
        outputs = (layer_output,) + outputs ##重新组合返回值
        return outputs ##返回一个元组


class HATEncoder(nn.Module): ## 把很多个HATLayer串联起来，形成完整的Hierarchical Transformer Encoder
    def __init__(self, config):
        super().__init__()
        self.config = config ##设置配置
        self.layer = nn.ModuleList([HATLayer(config) for idx in range(config.num_hidden_layers)]) ##创建所有 HATLayer，然后把它们保存到一个 nn.ModuleList 中，组成整个 HATEncoder，对于我的RNA-FM模型只需要堆叠12/N层Document Encoder，不需要更新堆叠sentence_encoder的结构
        self._tie_weights()
    def forward(
        self,
        hidden_states, ## HATEmbeddings的输出(B, num_segments, 640)
        output_attentions=False, ##是否返回 Attention Matrix，默认不返回
        output_hidden_states=False, ##是否保存每层输出，默认不返回因为占用内存
    ):
        all_hidden_states = () if output_hidden_states else None ##如果用户需要保存所有层输出，创建一个空tuple
        all_sentence_attentions = () if output_attentions else None ##Document Encoder Attention的初始化变量设置
        for i, layer_module in enumerate(self.layer): ##开始循环所有HATLayer,依次取出12个HATLayer
            if output_hidden_states: ##如果开启 output_hidden_states，保存每层输出
                all_hidden_states = all_hidden_states + (hidden_states,) ##保存当前层输入，进入这一层之前的 hidden_states
            layer_outputs = layer_module(hidden_states, output_attentions) ####否则正常Forward，调用HATLayer.forward()，返回(outputs, None)
            hidden_states = layer_outputs[0] ##下一层输入就是上一层输出，更新隐藏状态
            if output_attentions: ##保存Document的Attention
                all_sentence_attentions = all_sentence_attentions + (layer_outputs[1],)
        if output_hidden_states: ##加上保存最后一层的输出
            all_hidden_states = all_hidden_states + (hidden_states,)
        return BaseModelOutputWithSentenceAttentions(
            last_hidden_state=hidden_states, ##返回last_hidden_state：最后一层输出
            hidden_states=all_hidden_states, ##hidden_states：所有层输出
            sentence_attentions=all_sentence_attentions, ##sentence_attentions：Document Encoder Attention
        ) ##最后实际应该是只有最后一层输出，因为其他的都设置成false了没有输出
    def _tie_weights(self): ##一般在模型初始化完成后调用，不会进行 Forward，也不会参与训练过程，只是修改模型参数之间的引用关系，让所有 HATLayer 的 Document Encoder 使用同一套 Sentence Position Embedding 参数，而不是每一层都学习一套新的 Position Embedding
        original_position_embeddings = self.layer[0].position_embeddings
        for layer in self.layer[1:]:
            layer.position_embeddings.weight = (original_position_embeddings.weight)

class HATModel(nn.Module): ##定义一个完整的 HAT 模型，并继承 HATPreTrainedModel
    def __init__(self, config):  ##继承HATPreTrainedModel父类的config，不需要
        super().__init__()
        self.config = config ##设置配置信息
        self.encoder = HATEncoder(config) ##建立真正的建立真正的 HAT层
    def forward(
        self,
        embedding_input=None, ## RNA-FM Mean Pooling 后的输入：(B, N, 640)
        output_attentions=None, ##是否返回Attention权重
        output_hidden_states=None, ##是否返回每层输出
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions ##确定是否输出 Attention，如果调用模型时明确指定 output_attentions=True，就输出 Attention；如果没有指定，就使用 config 中的默认设置。
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states) ##确定是否返回每层输出
        encoder_outputs = self.encoder( ##整个模型最核心，真正进入HATEncoder部分
            hidden_states=embedding_input,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return encoder_outputs

# class AttentivePooling(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attn_dropout = config.hidden_dropout_prob
#         self.lin_proj = nn.Linear(config.hidden_size, config.hidden_size)
#         self.v = nn.Linear(config.hidden_size, 1, bias=False)

#     def forward(self, inputs):
#         lin_out = self.lin_proj(inputs)
#         attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
#         attention_weights_normalized = torch.softmax(attention_weights, -1)
#         return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), ##(640, 640)
            nn.Tanh(), ##Tanh 激活函数，Tanh 将数值映射到 \((-1,1)\) 范围内，为注意力评分引入非线性。
            nn.Linear(hidden_size, 1) ##(640, 1)
        )
    def forward(self, x):
        scores = self.attention(x).squeeze(-1) # 计算每个 Segment 的注意力分数; x: (B, num_segments, hidden_size)例子：(1, 3, 640)
        weights = torch.softmax(scores, dim=1) ##使用 Softmax 计算注意力权重
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1) ## 对片段特征进行加权求和,(B, num_segments)
        return pooled  # (1, 640),weights：(1, 10)

