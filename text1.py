import torch
from collections import OrderedDict

new_dict = OrderedDict()

#.pth路径
# pthfile = "C:\\Users\\vive\\Desktop\\vit_base_patch16_224_in21k.pth"
pthfile = "D:\\Two_stream\\epoch=26_val_acc=0.0161.pth"
model = dict(torch.load(pthfile, torch.device('cuda:0')))
# model = torch.load(pthfile)
# print(type(model))
# for i in model["state_dict"].keys():
#     print(i)
# print(model.state_dict())
# print(model)
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#查看模型字典里面的key
for k in model.keys():
    print(k)

#原pth文件有的fc.bias 将其换成fc_action.bias
# for k, v in model.items():
#     name = k[7:]
#     new_dict[name] = v
# # for k in new_dict.keys():
# #     print(k)
# cur_path = "C:\\Users\\vive\\Desktop\\epoch=26_val_acc=0.0161"
# torch.save(new_dict, cur_path)

# model['fc_action.weight']  = model.pop('fc.weight')
# torch.save(model, pthfile)
# Missing key(s) in state_dict: "fc_action.weight", "fc_action.bias".
# Unexpected key(s) in state_dict: "fc.weight", "fc.bias".