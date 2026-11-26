"""eraser.pt → docs/eraser.onnx (웹 UI용). 모델을 다시 학습했으면 이걸 실행해서 웹에도 반영하세요.
  python export_onnx.py
"""
import torch

from model import EraserUNet

ck = torch.load("eraser.pt", map_location="cpu")
net = EraserUNet(ck.get("width", 16)).eval()
net.load_state_dict(ck["model"])
x = torch.rand(1, 1, 512, 512)
torch.onnx.export(net, x, "docs/eraser.onnx", input_names=["image"], output_names=["clean", "logit"],
                  dynamic_axes={"image": {2: "h", 3: "w"}, "clean": {2: "h", 3: "w"}, "logit": {2: "h", 3: "w"}},
                  opset_version=17, dynamo=False)
print("저장: docs/eraser.onnx")
