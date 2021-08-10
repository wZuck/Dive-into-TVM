import os
from rich.console import Console

console = Console(width=120)


def pprint(chapter, object):
    console.rule(chapter)
    console.print(object)


import torch
import torchvision

# TVM的pytorch前端只接受scripted_model（静态图模型），因此需要先转换为JIT模型

if not os.path.exists("scripted_model.pth"):
    model_name = "resnet18"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    # Grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    scripted_model.save("scripted_model.pth")

# 剖析JIT scripted model

scripted_model = torch.load("scripted_model.pth")

pprint("scripted_model.graph", scripted_model.graph)
pprint("scripted_model.graph", scripted_model.graph)
pprint("scripted_model.graph.nodes()", list(scripted_model.graph.nodes()))
pprint("scripted_model.state_dict().keys()", scripted_model.state_dict().keys())
pprint("list(scripted_model.graph.inputs())", list(scripted_model.graph.inputs()))
pprint(
    "So we need use [1] for module-class scripted_model",
    list(scripted_model.graph.inputs())[1],
)
pprint(
    "Show the output Attr of a graph node",
    list(list(scripted_model.graph.nodes())[2].outputs()),
)
pprint("Return nodes of graph", scripted_model.graph.return_node())
