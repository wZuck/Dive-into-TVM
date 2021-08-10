import numpy as np
import tvm
from tvm import relay

data_type = "float32"
data_shape = (1, 1, 9, 9)
weight_shape = (1, 1, 3, 3)
strides = (2, 2)
padding = (0, 0, 0, 0)
layout = "NCHW"
kernel_layout = "OIHW"
data = relay.var('data', shape=data_shape, dtype=data_type)
weight = relay.var('weight', shape=weight_shape, dtype=data_type)
out = relay.nn.conv2d(data, weight, padding=padding, data_layout=layout, kernel_layout=kernel_layout)
module = tvm.IRModule.from_expr(out)


print(module)
print(out)

from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib

module = partition_for_arm_compute_lib(module)
print(module)
target = "llvm -mtriple=armv7l-linux-gnueabihf -mattr=+neon"
with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    lib = relay.build(module, target=target)
print(lib)
lib_path = './lib_acl.so'
# cross_compile = 'aarch64-linux-gnu-c++'
cross_compile = 'arm-linux-gnueabihf-g++'
# cross_compile = 'arm-none-eabi-g++'

lib.export_library(lib_path, cc=cross_compile)

ctx = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('./lib_acl.so')
gen_module = tvm.contrib.graph_runtime.GraphModule(loaded_lib['default'](ctx))

print(gen_module)

d_data = np.random.uniform(0, 1, data_shape).astype(data_type)
d_kernel = np.random.uniform(0, 1, weight_shape).astype(data_type)
map_inputs = {'data': d_data,'weight':d_kernel}
gen_module.set_input(**map_inputs)
gen_module.run()

print('input: ',gen_module._get_input(0).asnumpy().shape)
print('kernel: ',gen_module._get_input(1).asnumpy().shape)
print('output: ',gen_module._get_output(0).asnumpy().shape)

for i in range(9):
    for j in range(9):
        print("({},{}) {:.3f}\t".format(i, j, gen_module._get_input(0).asnumpy()[0][0][i][j]), end='')
        # print('(', i, ',', j, ') ', gen_module._get_input(0).asnumpy()[0][0][i][j], end='')
    print('')
print('')

for i in range(3):
    for j in range(3):
        print("({},{}) {:.3f}\t".format(i, j, gen_module._get_input(1).asnumpy()[0][0][i][j]), end='')
        # print('(', i, ',', j, ') ', gen_module._get_input(0).asnumpy()[0][0][i][j], end='')
    print('')
print('')

for i in range(7):
    for j in range(7):
        print("({},{}) {:.3f}\t".format(i, j, gen_module._get_output(0).asnumpy()[0][0][i][j]), end='')
        # print('(', i, ',', j, ') ', gen_module._get_output(0).asnumpy()[0][0][i][j], end='')
    print('')

# print(gen_module._get_input(0).asnumpy()[0][0][0][0])
# print(gen_module._get_input(0).asnumpy()[0][0][1][0])
# print(gen_module._get_input(0).asnumpy()[0][1][0][0])
# print(gen_module._get_input(0).asnumpy()[0][1][1][0])
# print(gen_module._get_output(0).asnumpy()[0][0][0][0])

# print(d_data)

print('run done')
