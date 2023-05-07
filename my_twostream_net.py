import torch
import torch.nn as nn
import models
from models import vit_model as vm
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# 定义双流网络类
class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()

        self.rgb_branch = models.__dict__["rgb_resnet101"](pretrained=False, num_classes=101)
        rbg_weights = "D:\\Two_stream\\checkpoints\\frame_pth_0\\model_best.pth.tar"
        rgb_weights_dict = torch.load(rbg_weights, map_location=device)
        new_weights_dict = OrderedDict()
        for k, v in rgb_weights_dict["state_dict"].items():
            name = k[:7]
            if name == "module.":
                new_name = k[7:]
                new_weights_dict[new_name] = v
            else:
                new_weights_dict[k] = v
        self.rgb_branch.load_state_dict(new_weights_dict)

        self.opticalFlow_branch = vm.vit_base_patch16_224(num_classes=101, flow_channels=10 * 2)
        flow_weights = "D:\\Two_stream\\checkpoints\\flow_pth\\model_best.pth.tar"
        flow_weights_dict = torch.load(flow_weights, map_location=device)
        new_weights_dict1 = OrderedDict()
        for k, v in flow_weights_dict["state_dict"].items():
            name = k[:7]
            if name == "module.":
                new_name = k[7:]
                new_weights_dict1[new_name] = v
            else:
                new_weights_dict1[k] = v
        self.opticalFlow_branch.load_state_dict(new_weights_dict1)

    def forward(self, x_rgb, x_opticalFlow):
        rgb_out = self.rgb_branch(x_rgb)
        opticalFlow_out = self.opticalFlow_branch(x_opticalFlow)
        # print(rgb_out.shape)
        # print(rgb_out)
        # print(opticalFlow_out.shape)
        # print(opticalFlow_out)
        # 相加融合，并采用softmax函数
        final_out = nn.Softmax(dim=1)(rgb_out + opticalFlow_out * 0.1)
        return final_out
