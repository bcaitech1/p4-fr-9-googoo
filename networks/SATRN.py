import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

from dataset import START, PAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''

from torch.nn.modules.loss import _WeightedLoss


# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#     def forward(self, x, target, smoothing=0.1):
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss
#         return loss.mean()


# Maihon 버전 커스텀
class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1, ignore_index=-100, num_classes=245):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0, ignore_index=-100):
        assert 0 <= smoothing < 1
        # targets: [36, 94] batch x # of sentence
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes, targets.size(1)),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(-1, targets.data.unsqueeze(1), 1. - smoothing) # dim=-1로 수정
                
        # print(targets[0].shape, targets[0]) # [94, 245] => sentence, token
        print(targets[0][0].shape, targets[0][0]) # [245] => token => 1개만 1.
        # print(targets[0][0][0])
        # print(targets.shape, targets) # 36, 245, 04 => batch, token, sentence
#             one_hot_smooth = torch.full((targets.size(1), n_classes,), smoothing / (n_classes-2), device=device) # size 만큼 smoothing_value로 채우기
#             if ignore_index != -100:
#                 one_hot_smooth[ignore_index] = 0.0 # pad
#             one_hot_smooth = one_hot_smooth.repeat(targets.size(0), 1)
#             one_hot_smooth = one_hot_smooth.scatter(-1, targets.unsqueeze(-1), 1.  - smoothing)
        return targets
    
    def forward(self, inputs, targets): # batch, sentence_size, num_classes
        # print('inputs', inputs.shape) # [36, 245, 94] : batch x # of tokens x # of sentence
        # print('targets', targets.shape) # [36, 94] batch x # of sentence
        # print('self.num_classes', self.num_classes)
        targets = LabelSmoothingCrossEntropy._smooth_one_hot(targets, self.num_classes, self.smoothing, self.ignore_index) # inputs.size(-1)
        lsm = F.log_softmax(inputs, -1)
        # print('after targets', targets.shape) # 36 94 245 => batch x sentence x token
        # print('lsm', lsm.shape) # 36 245 94 => batch x token x sentence
        # print(targets.shape, lsm.shape)
        # print(targets, lsm) 
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        # torch.reshape(targets, (targets.size(0), targets.size(2), targets.size(1))) 
        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

# # https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()

#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))

#         self.confidence = 1.0 - label_smoothing

#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

#         return F.kl_div(output, model_prob, reduction='sum')

'''
class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2, num_bn=3):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate # 중간에 num_bn * growth_rate 만큼 ch 생성 후 //num_bn 줄여줌
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False # size : input => inter
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False # size : inter => groth_rate
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1) # 입력에 출력 덧붙여 리턴


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False # 채널 output_size만큼 (보통 절반) 축소
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # feature map size 절반으로 축소

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """

    def __init__(
        self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2, depth=16, growth_rate=24
    ):
        super(DeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d(
            input_channel,  # 3
            num_in_features,  # 48
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features

        self.block1 = DenseBlock(
            num_features,  # 48
            growth_rate=growth_rate,  # 48 + growth_rate(24)*depth(16) -> 432
            depth=depth,  # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate # conv layer 통과 할때마다 g_r만큼 ch 증가
        self.trans1 = TransitionBlock(num_features, num_features // 2)  # 16 x 16, Denseblock 사이에 존재, 줄여주는 역할
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features,  # 128
            growth_rate=growth_rate,  # 16
            depth=depth,  # 8
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False  # 128
        )

    def forward(self, input):
        out = self.conv0(input)  # (H, V, )
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)
        return out_A  # 128 x (16x16)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature # 벡터사이즈의 루트값
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        # linear 통해 q, k, v 생성
        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.attention = ScaledDotProductAttention(
            temperature=(self.head_num * self.head_dim) ** 0.5, dropout=dropout
        )
        self.out_linear = nn.Linear(self.head_num * self.head_dim, q_channels) # 여러개의 HEAD를 하나로 합치는 역할
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = (
            self.q_linear(q)
            .view(b, q_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(k)
            .view(b, k_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(v)
            .view(b, v_len, self.head_num, self.head_dim)
            .transpose(1, 2)
        )

        if mask is not None: # Decoder에서 mask 사용 => 학습이 뒤에 나온 단어 가리
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, q_len, self.head_num * self.head_dim)
        )
        out = self.out_linear(out)
        out = self.dropout(out)

        return out


class Feedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(Feedforward, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, filter_size, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(filter_size, hidden_dim, True),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

    def forward(self, input):
        return self.layers(input)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)
        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input):

        att = self.attention_layer(input, input, input)
        out = self.attention_norm(att + input)

        ff = self.feedforward_layer(out)
        out = self.feedforward_norm(ff + out)
        return out


class PositionalEncoding2D(nn.Module):
    # SATRN 논문 : w & h 방향 각각 포지션 인코딩 후 학습 가능한 α, β 곱
    # feature map에서 global Avgpooling한 벡터에서 α, β 회귀
    # baseline code => 단순히 둘로 나뉘어 한쪽은 w, 한쪽은 h 포지션 인코딩 후 concat
    def __init__(self, in_channels, max_h=64, max_w=128, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()

        self.h_position_encoder = self.generate_encoder(in_channels // 2, max_h)
        self.w_position_encoder = self.generate_encoder(in_channels // 2, max_w)

        self.h_linear = nn.Linear(in_channels // 2, in_channels // 2)
        self.w_linear = nn.Linear(in_channels // 2, in_channels // 2)

        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len): # 포지션 인코딩에 활용되는 수식
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder  # (Max_len, In_channel)

    def forward(self, input):
        ### Require DEBUG
        b, c, h, w = input.size()
        h_pos_encoding = (
            self.h_position_encoder[:h, :].unsqueeze(1).to(input.get_device())
        )
        h_pos_encoding = self.h_linear(h_pos_encoding)  # [H, 1, D]

        w_pos_encoding = (
            self.w_position_encoder[:w, :].unsqueeze(0).to(input.get_device())
        )
        w_pos_encoding = self.w_linear(w_pos_encoding)  # [1, W, D]

        h_pos_encoding = h_pos_encoding.expand(-1, w, -1)   # h, w, c/2
        w_pos_encoding = w_pos_encoding.expand(h, -1, -1)   # h, w, c/2

        pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=2)  # [H, W, 2*D]

        pos_encoding = pos_encoding.permute(2, 0, 1)  # [2*D, H, W]

        out = input + pos_encoding.unsqueeze(0)
        out = self.dropout(out)

        return out


class TransformerEncoderFor2DFeatures(nn.Module):
    """
    Transformer Encoder for Image
    1) ShallowCNN : low-level visual feature identification and dimension reduction
    2) Positional Encoding : adding positional information to the visual features
    3) Transformer Encoders : self-attention layers for the 2D feature maps
    """

    def __init__(
        self,
        input_size,
        hidden_dim,
        filter_size,
        head_num,
        layer_num,
        dropout_rate=0.1,
        checkpoint=None,
    ):
        super(TransformerEncoderFor2DFeatures, self).__init__()

        # 원조 shallow_cnn 대신 DeepCNN 사용 (DenseNet)
        self.shallow_cnn = DeepCNN300(
            input_size,
            num_in_features=48,
            output_channel=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.positional_encoding = PositionalEncoding2D(hidden_dim) # 
        self.attention_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate) # layer 갯수만큼 transformer 통과
                for _ in range(layer_num)
            ]
        )
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):

        out = self.shallow_cnn(input)  # [b, c, h, w]
        out = self.positional_encoding(out)  # [b, c, h, w]

        # flatten, SATRN 논문에서는 hxw처럼 합치지 않고 2-d feature 활용
        b, c, h, w = out.size()
        out = out.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]

        for layer in self.attention_layers:
            out = layer(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, src_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=input_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(
            q_channels=input_size,
            k_channels=src_size,
            head_num=head_num,
            dropout=dropout_rate,
        )
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(
            filter_size=filter_size, hidden_dim=input_size
        )
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask):

        if tgt_prev == None:  # Train, tf시 모든 입력이 한 번에 들어가기 때문에 무조건 None
            att = self.self_attention_layer(tgt, tgt, tgt, tgt_mask) # q, k, v
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(tgt, src, src)
            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)
        else:
            tgt_prev = torch.cat([tgt_prev, tgt], 1)
            att = self.self_attention_layer(tgt, tgt_prev, tgt_prev, tgt_mask) # q, k, v
            out = self.self_attention_norm(att + tgt)

            att = self.attention_layer(tgt, src, src)
            out = self.attention_norm(att + out)

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out)
        return out


class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len)
        self.position_encoder = self.position_encoder.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)

        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def forward(self, x, point=-1):
        if point == -1:
            out = x + self.position_encoder[:, : x.size(1), :].to(x.get_device())
            out = self.dropout(out)
        else:
            out = x + self.position_encoder[:, point, :].unsqueeze(1).to(x.get_device())
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        hidden_dim,
        filter_dim,
        head_num,
        dropout_rate,
        pad_id,
        st_id,
        layer_num=1,
        checkpoint=None,
    ):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num

        self.pos_encoder = PositionEncoder1D(
            in_channels=hidden_dim, dropout=dropout_rate
        )

        self.attention_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_dim, src_dim, filter_dim, head_num, dropout_rate
                )
                for _ in range(layer_num)
            ]
        )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def pad_mask(self, text):
        pad_mask = text == self.pad_id # 패딩 된 것을 찾아서 마스크를 지워주는  역할
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, length):
        order_mask = torch.triu(torch.ones(length, length), diagonal=1).bool() # triu : upper triangle을 0으로
        order_mask = order_mask.unsqueeze(0).to(device)
        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(
        self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0
    ):

        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text) # 정답 전체 임베딩
            tgt = self.pos_encoder(tgt) # 포지션 인코딩
            tgt_mask = self.pad_mask(text) | self.order_mask(text.size(1)) # order_mask : 뒤를 보지 않도록 하는 마스크
            for layer in self.attention_layers:
                tgt = layer(tgt, None, src, tgt_mask) # 타겟, 이전출력, 소스, 타겟마스크
            out = self.generator(tgt)
        else:
            out = []
            num_steps = batch_max_length - 1
            target = torch.LongTensor(src.size(0)).fill_(self.st_id).to(device) # [START] token
            features = [None] * self.layer_num

            for t in range(num_steps):
                target = target.unsqueeze(1)
                tgt = self.text_embedding(target)
                tgt = self.pos_encoder(tgt, point=t)
                tgt_mask = self.order_mask(t + 1)
                tgt_mask = tgt_mask[:, -1].unsqueeze(1)  # [1, (l+1)], 마스킹을 실제론 안 하는 것
                for l, layer in enumerate(self.attention_layers):
                    tgt = layer(tgt, features[l], src, tgt_mask)
                    # 앞에서 계산한 Attention을 features에 저장하여 reuse 하기 위함
                    features[l] = (
                        tgt if features[l] == None else torch.cat([features[l], tgt], 1)
                    )

                _out = self.generator(tgt)  # [b, 1, c]
                target = torch.argmax(_out[:, -1:, :], dim=-1)  # [b, 1], 다음 타겟으로 넘김
                target = target.squeeze()   # [b]
                out.append(_out)
            
            out = torch.stack(out, dim=1).to(device)    # [b, max length, 1, class length]
            out = out.squeeze(2)    # [b, max length, class length]

        return out


class SATRN(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None):
        super(SATRN, self).__init__()

        self.encoder = TransformerEncoderFor2DFeatures(
            input_size=FLAGS.data.rgb,
            hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
            filter_size=FLAGS.SATRN.encoder.filter_dim,
            head_num=FLAGS.SATRN.encoder.head_num,
            layer_num=FLAGS.SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
        )

        self.decoder = TransformerDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.SATRN.decoder.src_dim,
            hidden_dim=FLAGS.SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.SATRN.decoder.filter_dim,
            head_num=FLAGS.SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.SATRN.decoder.layer_num,
        )

#         if FLAGS.loss == 'Label_Smoothing':
#             self.criterion = (
#                 LabelSmoothingCrossEntropy(
#                     weight=None,
#                     reduction='mean',
#                     smoothing=FLAGS.Label_Smoothing.label_smoothing,
#                     ignore_index=train_dataset.token_to_id[PAD],
#                     num_classes=len(train_dataset.id_to_token),
#                 )
#             )
#         else:    
        self.criterion = (
            nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
        )  # without ignore_index=train_dataset.token_to_id[PAD]

        if checkpoint:
            self.load_state_dict(checkpoint)

    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        enc_result = self.encoder(input) # Attention은 flatten 후 reshape 하여 decoder에 전달 => Satrn은 reshape이 내부에서 진행
        dec_result = self.decoder(
            enc_result,
            expected[:, :-1],
            is_train,
            expected.size(1),
            teacher_forcing_ratio,
        )
        return dec_result