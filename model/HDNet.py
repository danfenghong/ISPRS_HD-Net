import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

BN_MOMENTUM = 0.1
ALIGN_CORNERS = True


# generate flow_field
class Generate_Flowfield(nn.Module):
    def __init__(self, inplane, stage=2):
        super(Generate_Flowfield, self).__init__()
        self.stage = stage
        self.flow_make_stage2 = nn.Conv2d(inplane * 2, 2, kernel_size=3, padding=1, bias=False)
        self.flow_make_stage3 = nn.Conv2d(inplane * 3, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, x_down):
        size = x.size()[2:]
        if self.stage == 2:
            flow = self.flow_make_stage2(torch.cat([x, x_down], dim=1))
        if self.stage == 3:
            flow = self.flow_make_stage3(torch.cat([x, x_down], dim=1))
        grid = self.Generate_grid(x, flow, size)
        return grid

    def Generate_grid(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)  # [out_h, 1] -> [out_h, out_w]
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)  # [1, out_w] -> [out_w, out_h]
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)  # [out_w, out_h, 2]

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return grid


class FusionModule(nn.Module):
    def __init__(self, basechannel, stage=2):
        super(FusionModule, self).__init__()
        self.stage = stage
        self.edge_fusion = nn.Conv2d(basechannel * 2, basechannel, kernel_size=1, bias=False)
        self.edge_out = nn.Conv2d(basechannel, 1, kernel_size=1, bias=False)
        self.seg_mid_fusion1 = nn.Conv2d(basechannel * 2, basechannel, kernel_size=1, bias=False)
        self.seg_mid_fusion2 = nn.Conv2d(basechannel * 3, basechannel, kernel_size=1, bias=False)
        self.seg_mid_out = nn.Conv2d(basechannel, 1, kernel_size=1, bias=False)
        self.seg_out = nn.Identity()
        self.seg_out1 = nn.Identity()
        self.seg_out2 = nn.Identity()

    def forward(self, input, grid, x_fine, x_abstract):
        seg_body, seg_edge = self.Decouple_seg(input, grid)
        seg_body = self.seg_out1(seg_body)
        seg_edge = self.seg_out2(seg_edge)
        seg_edge = self.edge_fusion(torch.cat([seg_edge, x_fine], dim=1))
        if self.stage == 2:
            seg_mid = self.seg_mid_fusion1(torch.cat([seg_body, x_abstract], dim=1))
        if self.stage == 3:
            seg_mid = self.seg_mid_fusion2(torch.cat([seg_body, x_abstract], dim=1))

        seg_mid_out = self.seg_mid_out(seg_mid)
        seg_edge_out = self.edge_out(seg_edge)

        seg_map = 1 * (torch.sigmoid(seg_mid_out).detach()).float()
        edge_map = 1 * (torch.sigmoid(seg_edge_out).detach()).float()
        seg_out = seg_edge * edge_map * (1 - seg_map) + seg_mid * seg_map * (1 - edge_map)
        seg_out = self.seg_out(seg_out)

        return seg_out, seg_edge, seg_edge_out

    # decouple
    def Decouple_seg(self, input, grid):
        seg_body = F.grid_sample(input, grid)
        seg_edge = input - seg_body
        return seg_body, seg_edge


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, multi_grid=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid,
                               dilation=dilation * multi_grid, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Stagethree_decouple(nn.Module):
    def __init__(self, input_branches, output_branches, c, dilation=1):
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        if dilation == 1:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=2, multi_grid=1),
                    BasicBlock(w, w, dilation=2, multi_grid=2),
                    BasicBlock(w, w, dilation=2, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 2:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=4, multi_grid=1),
                    BasicBlock(w, w, dilation=4, multi_grid=2),
                    BasicBlock(w, w, dilation=4, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 3:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=8, multi_grid=1),
                    BasicBlock(w, w, dilation=8, multi_grid=2),
                    BasicBlock(w, w, dilation=8, multi_grid=4)
                )
                self.branches.append(branch)
        elif dilation == 4:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w, dilation=16, multi_grid=1),
                    BasicBlock(w, w, dilation=16, multi_grid=2),
                    BasicBlock(w, w, dilation=16, multi_grid=4)
                )
                self.branches.append(branch)
        else:
            for i in range(self.input_branches):
                w = c * (2 ** i)
                branch = nn.Sequential(
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w),
                    BasicBlock(w, w)
                )
                self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='bilinear')
                        )
                    )
                else:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=False)
        # decouple operation
        self.generate_flow = Generate_Flowfield(c, stage=3)
        self.Fusion = FusionModule(c, stage=3)

    def forward(self, x, x_fine):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []
        x1 = self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[0][1](x[1])
        x3 = self.fuse_layers[0][2](x[2])
        x_down = torch.cat([x2, x3], dim=1)
        flow_field = self.generate_flow(x1, x_down)
        seg, seg_edge, seg_edge_out = self.Fusion(x1, flow_field, x_fine, x_down)
        for i in range(len(self.fuse_layers)):
            if i == 0:
                x_256 = seg + self.fuse_layers[0][0](x[0])
                xs = self.relu(x_256)
                x_fused.append(xs)

            if i == 1:
                x_1 = self.fuse_layers[1][1](x[1])
                x_2 = self.fuse_layers[1][0](x_256)
                x_128 = x_1 + x_2
                xs = self.relu(x_128)
                x_fused.append(xs)

            if i == 2:
                x_1 = self.fuse_layers[2][2](x[2])
                x_2 = self.fuse_layers[2][0](x_256)
                x_64 = x_1 + x_2
                xs = self.relu(x_64)
                x_fused.append(xs)

        return x_fused, seg_edge, seg_edge_out


class Stagetwo_decouple(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='bilinear')
                        )
                    )
                else:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=False)
        # decouple
        self.Fusion = FusionModule(c, stage=2)
        self.generate_flow = Generate_Flowfield(c, stage=2)

    def forward(self, x, x_fine):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []
        x1 = self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[0][1](x[1])
        flow_field = self.generate_flow(x1, x2)
        seg, seg_edge, seg_edge_out = self.Fusion(x1, flow_field, x_fine, x2)
        for i in range(len(self.fuse_layers)):
            if i == 0:
                x_256 = seg + self.fuse_layers[0][0](x[0])
                xs = self.relu(x_256)
                x_fused.append(xs)

            if i == 1:
                x_1 = self.fuse_layers[1][1](x[1])
                x_2 = self.fuse_layers[1][0](x_256)
                x_128 = x_1 + x_2
                xs = self.relu(x_128)
                x_fused.append(xs)

        return x_fused, seg_edge, seg_edge_out


class HighResolutionDecoupledNet(nn.Module):
    def __init__(self, base_channel: int = 48, num_classes: int = 1):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 512×512×64
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)  # 256×256×64
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 256×256×64
        self.bn3 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        self.getfine = nn.Conv2d(256, base_channel, kernel_size=1, bias=False)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2_1 = Stagetwo_decouple(input_branches=2, output_branches=2, c=base_channel)
        self.stage2_2 = Stagetwo_decouple(input_branches=2, output_branches=2, c=base_channel)
        # Stage3
        self.stage3_1 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=1)
        self.stage3_2 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=2)
        self.stage3_3 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=3)
        self.stage3_4 = Stagethree_decouple(input_branches=3, output_branches=3, c=base_channel, dilation=4)

        # Deep supervision
        self.classifier_seg1 = nn.Sequential(nn.Conv2d(base_channel * 3, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg2 = nn.Sequential(nn.Conv2d(base_channel * 3, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg3 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg4 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg5 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.classifier_seg6 = nn.Sequential(nn.Conv2d(base_channel * 7, base_channel, 1, 1),
                                             nn.BatchNorm2d(base_channel), nn.ReLU(inplace=True),
                                             nn.Conv2d(base_channel, num_classes, 1, 1))
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')
        # Final layer
        self.final_layer_seg = nn.Conv2d(6 * num_classes, num_classes, 1, 1)
        self.final_layer_bd = nn.Conv2d(6 * num_classes, num_classes, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer1(x)
        x_fine = self.getfine(x)

        x = [trans(x) for trans in self.transition1]  # Since now, x is a list  256×256×48/128×128×96

        x2_1, fine1, bd1 = self.stage2_1(x, x_fine)
        seg1 = self.classifier_seg1(self.concat_seg(x2_1))

        x2_2, fine2, bd2 = self.stage2_2(x2_1, fine1)
        seg2 = self.classifier_seg2(self.concat_seg(x2_2))

        x3 = [
            self.transition2[0](x2_2[0]),
            self.transition2[1](x2_2[1]),
            self.transition2[2](x2_2[-1])
        ]

        x3_1, fine3, bd3 = self.stage3_1(x3, fine2)
        seg3 = self.classifier_seg3(self.concat_seg(x3_1))
        x3_2, fine4, bd4 = self.stage3_2(x3_1, fine3)
        seg4 = self.classifier_seg4(self.concat_seg(x3_2))
        x3_3, fine5, bd5 = self.stage3_3(x3_2, fine4)
        seg5 = self.classifier_seg5(self.concat_seg(x3_3))
        x3_4, fine6, bd6 = self.stage3_4(x3_3, fine5)
        seg6 = self.classifier_seg6(self.upscore2(self.concat_seg(x3_4)))

        x_seg = torch.cat([seg1, seg2, seg3, seg4, seg5], 1)
        x_seg = self.upscore2(x_seg)
        x_seg = self.final_layer_seg(torch.cat([x_seg, seg6], dim=1))

        bd6 = self.upscore2(bd6)
        x_bd = torch.cat([bd1, bd2, bd3, bd4, bd5], 1)
        x_bd = self.upscore2(x_bd)
        x_bd = self.final_layer_bd(torch.cat([x_bd, bd6], dim=1))

        return x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6

    # deep supervision
    def concat_seg(self, x):
        if len(x) == 3:
            h, w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            x2 = F.interpolate(x[2], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            return torch.cat([x[0], x1, x2], 1)
        if len(x) == 2:
            h, w = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
            return torch.cat([x[0], x1], 1)
