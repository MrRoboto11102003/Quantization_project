import torch
import torch.nn as nn
from resnet import resnet20
import torch.ao.quantization.quantize_fx as quantize_fx

model = resnet20()
model.eval()

qconfig_dict = {"": torch.ao.quantization.get_default_qconfig("fbgemm")}
prepared_model = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=torch.randn(1, 3, 32, 32))
quantized_model = quantize_fx.convert_fx(prepared_model)

out = quantized_model(torch.randn(1, 3, 32, 32))
print("FX Quantization successful, output shape:", out.shape)
