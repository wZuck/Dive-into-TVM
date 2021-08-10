
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision

model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

input_name = "input0"
shape_list = [(input_name, img.shape)]
sym, params = relay.frontend.from_pytorch(scripted_model, shape_list)
target = tvm.target.create('llvm')
target = tvm.target.create('llvm -mtriple=aarch64-linux-gnu -mattr=+neon')

# 这里利用TVM构建出优化后模型的信息
with relay.build_config(opt_level=2):
    graph, lib, params = relay.build_module.build(sym, target, params=params)

dtype = 'float32'

from tvm.contrib import graph_runtime

# 下面的函数导出我们需要的动态链接库 地址可以自己定义
print("Output model files")
libpath = "/home/wzy/tvm_output_lib/mobilenet.so"
lib.export_library(libpath)

# 下面的函数导出我们神经网络的结构，使用json文件保存
graph_json_path = "/home/wzy/tvm_output_lib/mobilenet.json"
with open(graph_json_path, 'w') as fo:
    fo.write(graph)

# 下面的函数中我们导出神经网络模型的权重参数
param_path = "/home/wzy/tvm_output_lib/mobilenet.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
# -------------至此导出模型阶段已经结束--------

# 接下来我们加载导出的模型去测试导出的模型是否可以正常工作
loaded_json = open(graph_json_path).read()
loaded_lib = tvm.runtime.load_module(libpath)
loaded_params = bytearray(open(param_path, "rb").read())

# 这里执行的平台为CPU
ctx = tvm.cpu()

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
from torchvision import transforms

my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)
x = img

module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input("input0", x)
module.run()
out_deploy = module.get_output(0).asnumpy()

print(out_deploy)