"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import timm
import torch.nn.functional as F
from timm.models.layers.drop import drop_path
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    #yolov3 detection heads
    (384, 1, 1),
    (768, 3, 1),
    "S",
    (192, 1, 1),
    "U",
    (192, 1, 1),
    (384, 3, 1),
    "S",
    (96, 1, 1),
    "U",
    (96, 1, 1),
    (192, 3, 1),
    "S",
        ]

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.gelu(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d,groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                        nn.BatchNorm2d(mid_c), nn.GELU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                    nn.BatchNorm2d(out_c), nn.GELU())
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn  =  nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.conv(x)
            x = self.bn(x)
            x = self.gelu(x)
            return x
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes
            
    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    def __init__(self, in_channels = 3, backbone = 'convnext_tiny', num_classes = 80, config = config):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.num_classes = num_classes
        enc = timm.create_model(self.backbone, pretrained = False,drop_rate=0.0005)
        self.econv0 = enc.stem
        self.econv1 = enc.stages[0]   
        self.econv2 = enc.stages[1]   
        self.econv3 = enc.stages[2]
        self.econv4 = enc.stages[3]
        self.in_channels = self.econv4.blocks[2].mlp.fc2.out_features
        self.od_layers   = self._od_create_conv_layers()

        
    def forward(self, x):
        outputs = []           #for each scale
        route_connections = [] #for residual connection
        x = self.econv0(x)
        x = self.econv1(x)
        x = self.econv2(x)
        route_connections.append(x)
        x = self.econv3(x)
        route_connections.append(x)
        x = self.econv4(x)
        
        for layer in self.od_layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _od_create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2, mode = 'bilinear'))
                    in_channels = in_channels * 3
        return layers



class YOLOv3_ASPP(nn.Module):
    def __init__(self, in_channels = 3, backbone = 'convnext_tiny', num_classes = 80,strides=[3,2,1], config = config):
        super().__init__()
        self.backbone = backbone
        self.config   =   config
        self.num_classes = num_classes
        self.strides = strides
        enc = timm.create_model(self.backbone, pretrained = False, drop_rate=0.0005)
        self.econv0 = enc.stem
        self.econv1 = enc.stages[0]   
        self.econv2 = enc.stages[1]   
        self.econv3 = enc.stages[2]   
        self.econv4 = enc.stages[3]
        
        self.in_channels = self.econv4.blocks[2].mlp.fc2.out_features
        self.od_layers   = self._od_create_conv_layers()
        self.aspp        = self._assp_layers()

        
    def forward(self, x):
        outputs = []           #for each scale
        route_connections = [] #for residual connection
        x = self.econv0(x)
        x = self.econv1(x)
        x = self.econv2(x)  #192
        route_connections.append(x)
        x = self.econv3(x)  #384
        route_connections.append(x)
        x = self.econv4(x)  #768
        
        id = 0
        for layer in self.od_layers:
            if isinstance(layer, ScalePrediction):
                x = self.aspp[id](x)
                outputs.append(layer(x))
                id +=1
                continue
            x = layer(x)
            if isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _od_create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2, mode = 'bilinear'))
                    in_channels = in_channels * 3
        return layers
    
    
    def _assp_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for indx, channels in enumerate([in_channels//2, in_channels//4, in_channels//8]):
            layers.append(nn.Sequential(ASPP(inplanes=channels, mid_c=channels//2, out_c=channels,
                             dilations = [int(self.strides[indx]*1),
                                          int(self.strides[indx]*2),
                                          int(self.strides[indx]*3),
                                          int(self.strides[indx]*4)]),
    
                            nn.Dropout2d(0.25))
                            )
        return layers
        

# #
# if __name__ == "__main__":
#     num_classes = 20
#     IMAGE_SIZE = 416
#     device  = torch.device('cuda:'+ str(0))
#     model = YOLOv3_ASPP(num_classes=num_classes, backbone = 'convnext_tiny').cuda(device)
#     print(model)
#     cudnn.benchmark = True
#     print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
#
#     x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
#     x = x.type(torch.cuda.FloatTensor)
#     out = model(x)
#     print("Success!")




