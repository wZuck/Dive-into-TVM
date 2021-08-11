import os
from rich.console import Console

console = Console(width=120)


def pprint(chapter, object):
    console.rule(chapter)
    console.print(object)


import tvm

# target = tvm.target.Target("llvm", host="llvm")
target = tvm.target.arm_cpu("rasp4b64")
pprint("Target", target)
