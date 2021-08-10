import os
import datetime
import time
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, graph_runtime as runtime
from tvm.contrib.download import download_testdata

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np

# one line to get the model
# block = get_model("resnet18_v1", pretrained=True)
block = get_model("mobilenetv2_1.0", pretrained=True)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)

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
    synset = eval(f.read())


# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
shape_dict = {"data": x.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
# we want a probability so add a softmax operator


func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(
    func.body), None, func.type_params, func.attrs)


batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape


local_demo = False

if local_demo:
    target = "c"
    # target = tvm.target.Target("llvm")
else:
    target = tvm.target.arm_cpu("rasp3b")
    # The above line is a simple form of
    # target = tvm.target.Target('llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon')
start = datetime.datetime.now()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)
    _,lbttext,_=relay.build_module.build(func, target, params=params)

print(lbttext.get_source())


end = datetime.datetime.now()
print("Compile use: ", end-start)
# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = './net.tar'
lib.export_library(lib_fname)


# exit()
fsize = os.path.getsize(lib_fname)
print("Size: ", fsize/float(1024*1024))
# obtain an RPC session from remote device.
if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "114.212.85.77"
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
module.set_input("data", tvm.nd.array(x.astype("float32")))
# run
start = datetime.datetime.now()
module.run()
end = datetime.datetime.now()

print("run 1 pic  use: ", (end-start))
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())
print("TVM prediction top-1: {}".format(synset[top1]))
