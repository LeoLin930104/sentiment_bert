print("Importing Dependencies")
# convert_to_safetensors.py  ─── run this inside convert-env
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file
import os, shutil

SOURCE = "cl-tohoku/bert-base-japanese-v3"
TARGET = "bert-base-japanese-v3"   # folder you will copy back

print("▶ downloading original .bin weights …")
tok   = AutoTokenizer.from_pretrained(SOURCE)
model = AutoModel.from_pretrained(SOURCE)         # uses torch.load (allowed here)

print("▶ saving safetensors …")
os.makedirs(TARGET, exist_ok=True)
tok.save_pretrained(TARGET)
save_file({k: v.cpu() for k, v in model.state_dict().items()},
          f"{TARGET}/model.safetensors")
model.config.to_json_file(f"{TARGET}/config.json")
print("✅  wrote", TARGET)
