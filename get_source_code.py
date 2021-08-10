# all codes refer to https://zhuanlan.zhihu.com/p/108071133
# you may check tvm source code files

# python: python/tvm/contrib/
# cpp: src/runtime/contrib/ -> add module.cc (now easy, if complex -> refer to other backend)

# Add OPUModule src/runtime/contrib/opu
# CodeGen 代码生成


import tvm
from tvm import te

A = te.placeholder((10, 10))
B = te.compute((10, 10), lambda i, j: A[i, j])
s = te.create_schedule(B.op)
f = tvm.build(s, [A, B], tvm.target.arm_cpu("rasp3b"))
print(f.get_source())