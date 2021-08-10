import os

from PIL import Image
import datetime
import numpy as np

import torch
from torchvision.models.quantization import mobilenet as qmobilenet

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm import rpc
from tvm.contrib import utils, graph_runtime as runtime

def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def get_synset():
    synset_url = "".join(
            [
                "https://gist.githubusercontent.com/zhreshold/",
                "4d0b62f3d01426887599d4f7ede23ee5/raw/",
                "596b27d23537e5a1b5751d2b0481ef172f58b539/",
                "imagenet1000_clsid_to_human.txt",
            ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_runtime.GraphModule(lib["default"](tvm.context(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).asnumpy(), runtime

synset = get_synset()
inp = get_imagenet_input()

def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()

pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()

input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
# print(mod) # comment in to see the QNN IR dump


# ARM
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(
        func.body), None, func.type_params, func.attrs)

batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

local_demo = False

if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.arm_cpu("rasp3b")
    # The above line is a simple form of
    # target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon')


with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

tmp = utils.tempdir()
lib_fname = './net.tar'
lib.export_library(lib_fname)

fsize = os.path.getsize(lib_fname)
print("Size: ", fsize/float(1024*1024))


func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(
        func.body), None, func.type_params, func.attrs)


batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape


local_demo = False

if local_demo:
    target = tvm.target.Target("llvm")
else:
    target = tvm.target.arm_cpu("rasp3b")
    # The above line is a simple form of
    # target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon')
start = datetime.datetime.now()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)
end = datetime.datetime.now()
print("Compile use: ", end-start)
# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = './net.tar'
lib.export_library(lib_fname)

fsize = os.path.getsize(lib_fname)
print("Size: ", fsize/float(1024*1024))
# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "114.212.82.145"
    port = 9090
    remote = rpc.connect(host, port)

# upload the library to remote device and load it
start = datetime.datetime.now()
remote.upload(lib_fname)
end = datetime.datetime.now()

print("Upload use: ", end-start)

start = datetime.datetime.now()
rlib = remote.load_module("net.tar")
end = datetime.datetime.now()

print("load module use: ", end-start)
# create the remote runtime module
ctx = remote.cpu(0)
module = runtime.GraphModule(rlib["default"](ctx))
# set input data
module.set_input("input", tvm.nd.array(inp.astype("float32")))
# run
start = datetime.datetime.now()
times = 100
[module.run() for i in range(times)]
end = datetime.datetime.now()

print("run 1 pic  use: ", (end-start)/times)
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())
print("TVM prediction top-1: {}".format(synset[top1]))

# tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target="llvm")
#
# pt_top3_labels = np.argsort(pt_result[0])[::-1][:3]
# tvm_top3_labels = np.argsort(tvm_result[0])[::-1][:3]
#
# print("PyTorch top3 labels:", [synset[label] for label in pt_top3_labels])
# print("TVM top3 labels:", [synset[label] for label in tvm_top3_labels])
#
# print("%d in 1000 raw floating outputs identical." % np.sum(tvm_result[0] == pt_result[0]))
#
#
#
# n_repeat = 100  # should be bigger to make the measurement more accurate
# ctx = tvm.cpu(0)
# ftimer = rt_mod.module.time_evaluator("run", ctx, number=1, repeat=n_repeat)
# prof_res = np.array(ftimer().results) * 1e3
# print("Elapsed average ms:", np.mean(prof_res))