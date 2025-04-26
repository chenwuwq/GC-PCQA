import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Attention_Block(nn.Module):
    def __init__(self, input_channels):
        super(Attention_Block, self).__init__()

        self.Conv1x1 = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(input_channels, input_channels // 16, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(input_channels // 16, input_channels, kernel_size=1, bias=False)

    def forward(self, x):
        res = x

        x1 = self.Conv1x1(x)
        att1 = self.sigmoid(x1)

        x2 = self.avgpool(x)
        x2 = self.Conv_Squeeze(x2)
        x2 = self.Conv_Excitation(x2)
        att2 = self.sigmoid(x2)

        out = (att1 * att2) * res + res
        out = self.avgpool(out)
        return out


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            init.constant_(self.bias.data, 0.1)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, img_len):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(2304, 512)
        self.bn1 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.gc2 = GraphConvolution(512, 128)
        self.bn2 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.gc3 = GraphConvolution(128, 32)
        self.bn3 = nn.BatchNorm1d(img_len, eps=1e-05, momentum=0.1, affine=True)
        self.gc4 = GraphConvolution(32, 1)
        self.relu = nn.Softplus()

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def norm_adj(self, matrix):
        D = torch.diag_embed(matrix.sum(2))
        D = D ** 0.5
        D = D.inverse()
        # D(-1/2) * A * D(-1/2)
        normal = D.bmm(matrix).bmm(D)
        return normal.detach()

    def forward(self, feature, A):
        adj = self.norm_adj(A)
        gc1 = self.gc1(feature, adj)
        gc1 = self.bn1(gc1)
        gc1 = self.relu(gc1)

        gc2 = self.gc2(gc1, adj)
        gc2 = self.bn2(gc2)
        gc2 = self.relu(gc2)

        gc3 = self.gc3(gc2, adj)
        gc3 = self.bn3(gc3)
        gc3 = self.relu(gc3)

        gc4 = self.gc4(gc3, adj)
        gc4 = self.relu(gc4)
        return gc4, gc3, gc2, gc1


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Att1 = Attention_Block(256)
        self.Att4 = Attention_Block(2048)

        self.GCN = GCN(img_len=10)
        self.GCN2 = GCN(img_len=10)

        # Quality Prediction
        self.QP_l = nn.Linear(512, 1)
        self.QP_l2 = nn.Linear(128, 1)
        self.QP_l3 = nn.Linear(32, 1)
        self.QP_l4 = nn.Linear(10, 1)

        self.QP_l_2 = nn.Linear(512, 1)
        self.QP_l2_2 = nn.Linear(128, 1)
        self.QP_l3_2 = nn.Linear(32, 1)
        self.QP_l4_2 = nn.Linear(10, 1)

        self.QP_final = nn.Linear(10, 1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, A):
        batch_size = x.size(0)
        img_len = x.size(1)
        x = x.view(-1, 3, 224, 224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        g1 = self.Att1(x1)
        g4 = self.Att4(x4)

        g1 = g1.view(batch_size, img_len, -1, 1, 1).squeeze(3).squeeze(3)
        g4 = g4.view(batch_size, img_len, -1, 1, 1).squeeze(3).squeeze(3)

        x = torch.cat((g1, g4), dim=2)
        x1, x2 = x[:, 0:10], x[:, 10:20]

        gc4, gc3, gc2, gc1 = self.GCN(x1, A)
        gc1 = torch.mean(gc1, dim=1)
        gc2 = torch.mean(gc2, dim=1)
        gc3 = torch.mean(gc3, dim=1)
        s1 = self.QP_l(gc1)
        s2 = self.QP_l2(gc2)
        s3 = self.QP_l3(gc3)
        s4 = self.QP_l4(gc4.squeeze(2))

        out = gc4.view(batch_size, -1)
        score_h = torch.mean(out, dim=1).unsqueeze(1)

        gc4_2, gc3_2, gc2_2, gc1_2 = self.GCN2(x2, A)
        gc1_2 = torch.mean(gc1_2, dim=1)
        gc2_2 = torch.mean(gc2_2, dim=1)
        gc3_2 = torch.mean(gc3_2, dim=1)
        s1_2 = self.QP_l_2(gc1_2)
        s2_2 = self.QP_l2_2(gc2_2)
        s3_2 = self.QP_l3_2(gc3_2)
        s4_2 = self.QP_l4_2(gc4_2.squeeze(2))

        out_2 = gc4_2.view(batch_size, -1)
        score_v = torch.mean(out_2, dim=1).unsqueeze(1)

        s = torch.cat((score_h, s1, s2, s3, s4, score_v, s1_2, s2_2, s3_2, s4_2), dim=1)
        score_final = self.QP_final(s)
        return score_final


def Model(pretrained=False, progress=True, **kwargs):
    """ Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # resnet101
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = torch.load('pre_weight/resnet101-5d3b4d8f.pth')
        pre_train_model = {k: v for k, v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
        for name, para in model.named_parameters():
            if "GCN" not in name:
                para.requires_grad_(False)
    return model
