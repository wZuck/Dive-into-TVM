from os import remove
import numpy as np
import tvm
from tvm import te, rpc
from tvm import target
from tvm.contrib import utils


n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i]+1.0, name="B")
s = te.create_schedule(B.op)


local_demo = False

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=arm-linux-gnueabihf"

func = tvm.build(s, [A, B], target=target, name="add_one")
temp = utils.tempdir()
# path = temp.relpath("lib.tar")
path = "./lib.tar"
print("lib.tar path is: ", path)
func.export_library(path)


if local_demo:
    remote = rpc.LocalSession()
else:
    host = "114.212.82.145"
    port = 9090
    remote = rpc.connect(host, port)

remote.upload(path)
print(remote)
func = remote.load_module("lib.tar")

ctx = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)

func(a, b)

np.testing.assert_equal(b.asnumpy(), a.asnumpy()+1)

time_f = func.time_evaluator(func.entry_name, ctx, number=10)
cost = time_f(a, b).mean

print("%g secs/op" % cost)
