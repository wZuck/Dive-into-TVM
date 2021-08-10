from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
import tvm
from tvm import relay


data_type = "float32"
data_shape = (1, 14, 14, 512)
strides = (2, 2)
padding = (0, 0, 0, 0)
pool_size = (2, 2)
layout = "NHWC"
output_shape = (1, 7, 7, 512)

data = relay.var('data', shape=data_shape, dtype=data_type)
out = relay.nn.max_pool2d(data, pool_size=pool_size,
                          strides=strides, layout=layout, padding=padding)

module = tvm.IRModule.from_expr(out)

module = partition_for_arm_compute_lib(module)
print(module)
target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
# target = "c"
with tvm.transform.PassContext(opt_level=3,disabled_pass=['AlterOpLayout']):
    lib = relay.build(module,target=target)

print(lib)

exit()

lib_path = './lib_acl.so'
cross_compile = 'aarch64-linux-gnu-c++'
lib.export_library(lib_path,cc=cross_compile)   

ctx = tvm.cpu(0)
loaded_lib = tvm.runtime.load_module('./lib_acl.so')
gen_module = tvm.contrib.graph_runtime.GraphModule(loaded_lib['default'](ctx))



import numpy as np
d_data= np.random.uniform(0,1,data_shape).astype(data_type)
map_inputs = {'data':d_data}
gen_module.set_input(**map_inputs)
gen_module.run()
