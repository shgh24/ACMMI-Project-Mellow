# -*- coding: utf-8 -*-
"""VLM_experiments.ipynb

Original file is located at
    https://colab.research.google.com/drive/1HASvaZI8LVcbDHOV6GmPk4kNQrS_93AN
"""

# code borrowed from VLM homework 9
!pip install transformers accelerate bitsandbytes pillow torch -q

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = torch.randn(2, 3, device=device)
KEY = 42
cipher_bytes = [99, 10, 102, 101, 124, 111, 10, 107, 105, 103, 103, 99]

if t.is_cuda:
    cipher = torch.tensor(cipher_bytes, dtype=torch.uint8, device=device)
    decoded = torch.bitwise_xor(cipher, KEY)
    torch.cuda.synchronize()
    secret = bytes(decoded.tolist()).decode("ascii")
    print("SECRET_WORD:", secret)
else:
    print("SECRET_WORD: (not on GPU)")

import os, shutil, zipfile
from pathlib import Path

URL = "https://drive.google.com/file/d/1M1qdEF_Bw-grHqOgzMCR6S4PNDOXW5XM/view?usp=sharing"

DATA_DIR = Path("/content/")
DATA_DIR.mkdir(parents=True, exist_ok=True)

!pip -q install gdown
import gdown

print("Downloading…")
downloaded_path = gdown.download(URL, output=None, quiet=False, fuzzy=True)  # returns local filepath
if not downloaded_path or not os.path.exists(downloaded_path):
    raise RuntimeError("Download failed. Check the URL or your Drive permissions.")

src = Path(downloaded_path)
dst = DATA_DIR / src.name
if src.resolve() != dst.resolve():
    shutil.move(str(src), str(dst))

print(f"\nFile saved to: {dst}")

if zipfile.is_zipfile(dst):
    extract_dir = DATA_DIR / dst.stem
    extract_dir.mkdir(exist_ok=True)
    print(f"Unzipping into: {extract_dir}")
    with zipfile.ZipFile(dst, "r") as zf:
        zf.extractall(extract_dir)
    print("Unzip complete.")

if dst.suffix.lower() == ".jsonl":
    print("\nSet this in your training cell:")
    print(f'DATA_JSONL = "{dst}"')

import io, requests, torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# 1) Load model + processor (processor handles BOTH text + vision)
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
print("Model and tokenizer loaded successfully.")

import io
import os
import requests
import torch
from typing import Optional, Dict, Any
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ============================================================
# ######################## CHANGE ME #########################
# ============================================================
IMAGE_URL: str = "https://ik.imagekit.io/b6al9bcwl/MIT001R_baseline.png?updatedAt=1762842563090"
QUESTION: str = "Guess whether these physiological recordings were taken from participants during a neutral baseline task (1), a stress-inducing task (2), or a recovery task after experiencing stress (3)."
SYSTEM_PROMPT: str = "Present your choice as a number and explain your reasoning in 1-2 sentences."
MAX_NEW_TOKENS: int = 128
# ============================================================
# ###################### END CHANGE ME #######################
# ============================================================


# SYSTEM CONFIG
DO_SAMPLE: bool = False           # set True for non-greedy decoding
TEMPERATURE: float = 0.7          # used only if DO_SAMPLE=True
TOP_P: float = 0.9                # used only if DO_SAMPLE=True
MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"
FORCE_CPU: bool = False           # force CPU even if CUDA is available
DTYPE_IF_GPU = torch.bfloat16     # prefer bfloat16 on recent GPUs/Colab
DTYPE_IF_CPU = torch.float32

def get_device_and_dtype() -> tuple[torch.device, torch.dtype, Optional[Dict[str, Any]]]:
    """Choose device/dtype and (optionally) a device_map for accelerate-style placement."""
    use_cuda = torch.cuda.is_available() and not FORCE_CPU
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    torch_dtype = DTYPE_IF_GPU if use_cuda else DTYPE_IF_CPU
    device_map = "auto" if use_cuda else None
    return device, torch_dtype, device_map


def load_image_from_url(url: str) -> Image.Image:
    """Fetch image from URL and return a RGB PIL.Image with robust fallback."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    try:
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((128, 128)) # resize the image
        return img
    except UnidentifiedImageError:
        # Fallback: write to disk then reopen (sometimes fixes truncated headers)
        tmp_path = "temp_image.jpg"
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        img = Image.open(tmp_path).convert("RGB")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return img


def build_chat_messages(image: Image.Image, question: str) -> list[dict]:
    """Create a single-turn, image+text chat for Qwen-VL processors."""
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]


def main() -> None:
    device, torch_dtype, device_map = get_device_and_dtype()

    # 1) Load model + processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    print("Model and processor loaded successfully.")

    # 2) Load image
    image = load_image_from_url(IMAGE_URL)

    # 3) Build chat
    messages = build_chat_messages(image, QUESTION)

    # 4) Apply chat template and preprocess
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    # 5) Move to the right device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 6) Generate
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
    if DO_SAMPLE:
        gen_kwargs.update(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P))

    with torch.no_grad():
        gen_ids = model.generate(**inputs, **gen_kwargs)

    # 7) Decode
    out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print("\n=== MODEL OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()

# ==== Qwen2.5-VL-3B-Instruct • FP16 LoRA ====

from IPython.display import display, HTML
import os, io, json, requests, torch, random, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# Environment hygiene
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ============================================================
# ######################## CHANGE ME #########################
# ============================================================
# Training hyperparameters
NUM_EPOCHS: int  = 100
LR: float        = 1e-4
BSZ_PER_DEV: int = 1
GRAD_ACCUM: int  = 1
EVAL_SPLIT: float = 0.1
SEED: int        = 42

# Collator / sequence shaping
MAX_SEQ_LEN: int = 512   # try 384 if VRAM is tight

# Image preprocessing
SHORTEST_EDGE: int = 288  # smaller saves VRAM
# ============================================================
# ###################### END CHANGE ME #######################
# ============================================================


# SYSTEM CONFIG
# Paths
DATA_JSONL: str  = "/content/acmmi-data/data.jsonl"
OUTPUT_DIR: str  = "/content/qwen2_5_vl_lora_fp16_t4"

MODEL_ID: str     = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR: str    = "/content/cache_images"
IMAGE_TIMEOUT: int = 15

# LoRA configuration (attention-only keeps memory low)
LORA_R: int          = 4
LORA_ALPHA: int      = 8
LORA_DROPOUT: float  = 0.05
LORA_TARGET: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Device / dtype policy
FORCE_CPU: bool   = False
DTYPE_IF_GPU      = torch.float16
DTYPE_IF_CPU      = torch.float32


# Repro and cache dirs
torch.manual_seed(SEED); random.seed(SEED)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------
# Demo data (create if missing)
# --------------------
def _ensure_sample_data(path: str):
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
      display(HTML(
          "<div style='color:white; background-color:#2e7d32; padding:10px; border-radius:6px;'>"
          "<strong>Using custom training data:</strong> "
          f"Loaded dataset from <code>{path}</code>. "
          "Proceeding with user-provided images and JSONL file."
          "</div>"
      ))
      return

    demo = [
        {
            "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
            "question": "List objects you see.",
            "answer": "cat, sofa, blanket, remote, cushion, tail, paw"
        },
        {
            "image": "http://images.cocodataset.org/val2017/000000001532.jpg",
            "question": "List objects you see.",
            "answer": "car, truck, road, bridge, exit sign, lamppost, building, sky"
        },
    ]
    with open(path, "w") as f:
        for r in demo: f.write(json.dumps(r) + "\n")

    # Print a red warning box (works in Colab/Jupyter)
    display(HTML(
        "<div style='color:white; background-color:#b71c1c; padding:10px; border-radius:6px;'>"
        "<strong>Warning:</strong> No dataset found — using built-in <code>sample data</code> (2 demo images). "
        "Please replace with your own dataset of at least 20 images for training."
        "</div>"
    ))

_ensure_sample_data(DATA_JSONL)


# --------------------
# Minimal JSONL dataset
# --------------------
class JsonlVisionLangDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                ex = json.loads(line)
                if {"image","question","answer"} - set(ex.keys()): continue
                self.samples.append(ex)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int) -> Dict[str, Any]: return self.samples[idx]

full_ds = JsonlVisionLangDataset(DATA_JSONL)

# Manual split
n = len(full_ds); n_val = max(1, int(n * EVAL_SPLIT))
idx = list(range(n)); random.shuffle(idx)
val_idx = set(idx[:n_val])
train_data = [full_ds[i] for i in range(n) if i not in val_idx]
val_data   = [full_ds[i] for i in range(n) if i in val_idx]

class ListDataset(Dataset):
    def __init__(self, data_list): self.data_list = data_list
    def __len__(self): return len(self.data_list)
    def __getitem__(self, i): return self.data_list[i]


# --------------------
# Cache images locally (avoid network hiccups)
# --------------------
BASE_DIR = os.path.dirname(DATA_JSONL)

def cache_image(url_or_path: str) -> str:
    # Remote URL: download and cache
    if url_or_path.startswith(("http://", "https://")):
        h = hashlib.md5(url_or_path.encode()).hexdigest()
        local = os.path.join(CACHE_DIR, f"{h}.jpg")
        if not os.path.exists(local):
            r = requests.get(url_or_path, timeout=IMAGE_TIMEOUT); r.raise_for_status()
            with open(local, "wb") as f: f.write(r.content)
        return local

    # Local path: make absolute relative to the JSONL file
    candidate = url_or_path
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, url_or_path)

    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Image not found: {candidate} (from '{url_or_path}'). "
            f"Expected under {BASE_DIR}/"
        )
    return candidate


for ex in train_data: ex["image"] = cache_image(ex["image"])
for ex in val_data:   ex["image"] = cache_image(ex["image"])

train_ds = ListDataset(train_data)
val_ds   = ListDataset(val_data)


# --------------------
# Image loader
# --------------------
def load_image(img: str) -> Image.Image:
    return Image.open(img).convert("RGB")


# --------------------
# Processor + Model (FP16 on GPU, FP32 on CPU)
# --------------------
use_cuda = torch.cuda.is_available() and not FORCE_CPU
torch_dtype = DTYPE_IF_GPU if use_cuda else DTYPE_IF_CPU
device_map = "auto" if use_cuda else None

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch_dtype,            # transformers v5 uses 'dtype'
    device_map=device_map,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# Smaller images to save VRAM
try:
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        processor.image_processor.size = {"shortest_edge": int(SHORTEST_EDGE)}
        print(f"Set image shortest_edge to {SHORTEST_EDGE}")
except Exception as e:
    print("Skip image size tweak:", e)

# Enable gradient checkpointing; avoid k-bit prep (saves VRAM)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False


# --------------------
# LoRA (attention-only)
# --------------------
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# --------------------
# Collator (truncate to keep sequences small)
# --------------------
@dataclass
class VLDataCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images, texts = [], []
        for ex in features:
            img = load_image(ex["image"])
            messages = [
                {"role": "user", "content": [
                    {"type":"image","image": img},
                    {"type":"text","text": ex["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type":"text","text": ex["answer"]},
                ]},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            images.append(img); texts.append(text)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        for im in images:
            try: im.close()
            except: pass

        return batch

collator = VLDataCollator(processor)


# --------------------
# FP16 loss trainer to avoid fp32 upcast OOM
# --------------------
class FP16CLMTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,   # v5 may pass this
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # keep fp16 path if available

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


# --------------------
# TrainingArguments (Transformers v5+ naming)
# --------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BSZ_PER_DEV,
    per_device_eval_batch_size=1,
    dataloader_num_workers=0,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,

    eval_strategy="no",              
    save_strategy="steps",
    save_steps=10_000,

    fp16=use_cuda, bf16=False,       
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to=[],
    remove_unused_columns=False,
)

trainer = FP16CLMTrainer(
    model=model,
    args=args,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# Save LoRA adapters + processor
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Training complete. LoRA adapters saved to:", OUTPUT_DIR)

# --------------------
# Inference with adapters
# --------------------
from peft import PeftModel
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    # device_map="cuda:0",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(base, OUTPUT_DIR)
ft_model.eval()
print("LoRA adapters loaded. Ready for inference.")

from IPython.display import display
import PIL.Image
import pandas as pd

def qa_pair(input_image, question, max_new_tokens=128):
  # ============================================================
  # ######################## CHANGE ME #########################
  # ============================================================
  TEST_IMAGE: str = input_image
  TEST_QUESTION: str = question
  MAX_NEW_TOKENS: int = max_new_tokens
  # ============================================================
  # ###################### END CHANGE ME #######################
  # ============================================================

  # Load image
  loaded_img = load_image_from_url(TEST_IMAGE)

  # Display image
  print("Image:")
  display(loaded_img)

  # Prepare messages
  messages = [{"role": "user", "content": [
      {"type": "image", "image": loaded_img},
      {"type": "text", "text": TEST_QUESTION}
  ]}]

  # Encode and generate
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(text=[text], images=[loaded_img], return_tensors="pt").to(ft_model.device)

  with torch.no_grad():
      out_ids = ft_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

  # Decode output
  output_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
  return output_text

phase_map = {'Baseline' : 'baseline',
             'Stroop': 'stress',
             'Math': 'stress',
             'MollyIntervention' : 'recovery',
             'Post-Relaxation': 'recovery'}

# 5-class mapping
phase_map_5 = {
    "Baseline": "baseline",
    "Stroop": "stroop_stress",
    "Math": "math_stress",
    "MollyIntervention": "molly_intervention",
    "Post-Relaxation": "post_relaxation",
}

# canonical substrings mapping
label_substrings_5 = {
    "baseline": ["baseline"],
    "stroop_stress": ["stroop"],
    "math_stress": ["math"],
    "molly_intervention": ["molly"],
    "post_relaxation": ["post-relaxation", "relaxation", "recovery"],
}

# 5 class question
q_5 = (
    "Which exact phase does this physiological recording correspond to? "
    "Choose one of: Baseline, Stroop_Stress, Math_Stress, MollyIntervention, or Post-Relaxation. "
    "Answer using the phase name."
)

test_images = ['https://ik.imagekit.io/b6al9bcwl/MIT001_R_Post-Relaxation_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Math_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Math_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Baseline_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Stroop_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Math_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Baseline_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Stroop_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Math_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Baseline_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_Stroop_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Stroop_Stress_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_MollyIntervention_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Post-Relaxation_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_R_MollyIntervention_combined.png',
               'https://ik.imagekit.io/b6al9bcwl/MIT001_L_Baseline_combined.png',]
results = pd.DataFrame(columns=["image", "ground_truth", "prediction"])
q = "Guess whether this physiological recording was taken from participants during a neutral baseline task (1), a stress-inducing task (2), or a recovery task after experiencing stress (3)."
for img in test_images:
  phase = img.split('_')[2]
  if phase in phase_map.keys():
    new_row = {'image': img,
              'ground_truth': phase_map[phase],
              'prediction': qa_pair(img, q, max_new_tokens=128)}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

results

# --------- begin project-specific edits ----------

"""# Addition #1: sweep configuration and output directory"""

# lora hyperparameter grid search and fine-tuning

import os
import pandas as pd
import matplotlib.pyplot as plt

from peft import LoraConfig, get_peft_model, PeftModel

SWEEP_OUTPUT_ROOT = "/content/lora_ablation_runs"
os.makedirs(SWEEP_OUTPUT_ROOT, exist_ok=True)

# LoRA sweep configurations
# target_type is just a label used later for plotting.
SWEEP_CONFIGS = [
    {
        "name": "r2",
        "r": 2,
        "alpha": 8,
        "dropout": 0.05,
        "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_type": "attn_only",
    },
    {
        "name": "r4",
        "r": 4,
        "alpha": 8,
        "dropout": 0.05,
        "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_type": "attn_only",
    },
    {
        "name": "r8",
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_type": "attn_only",
    },
    {
        "name": "r16",
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_type": "attn_only",
    },
    {
        "name": "r8_do15",
        "r": 8,
        "alpha": 16,
        "dropout": 0.15,
        "target": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "target_type": "attn_only",
    },
    {
        "name": "r8_mlp",
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "target_type": "attn_plus_mlp",
    },
]

# shared training hyperparameters for all sweeps
NUM_EPOCHS = 50    
LR = 1e-4
BSZ_PER_DEV = 1
GRAD_ACCUM = 1

"""# Addition #2: Evaluation + plotting helpers"""

import io
import torch
import random
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments

# helper
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def qa_pair_with_model(model, processor, image_url, question, max_new_tokens=128):
    """
    Run VLM QA for a single (image_url, question) pair using a given model+processor.
    """
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    img.close()
    # assistant's response
    assistant_prefix = "assistant\n"
    if assistant_prefix in output_text:
        output_text = output_text.split(assistant_prefix, 1)[1]
    return output_text


def load_ft_model_and_processor(adapter_dir):
    """Utility to load base VLM + LoRA adapter and processor."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base, adapter_dir)
    ft_model.eval()
    return ft_model, processor


def evaluate_model_on_heldout(adapter_dir):
    """
    Load base model + LoRA adapter and evaluate on global `test_images` and `phase_map`.

    Accuracy is computed as:
      (# images where ground-truth label substring appears in output) / (# evaluated images)
    """
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base.to("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    ft_model = PeftModel.from_pretrained(base, adapter_dir)
    ft_model.eval()

    correct = 0
    total = 0

    for img_url in test_images:
        phase = img_url.split("_")[2]
        if phase not in phase_map:
            # skip images whose phase doesn't map to baseline/stress/recovery
            continue

        gt = phase_map[phase]  # "baseline", "stress", "recovery"
        total += 1

        pred = qa_pair_with_model(ft_model, processor, img_url, q, max_new_tokens=128)
        if gt in pred.lower():
            correct += 1

    if total == 0:
        return 0.0
    return correct / total


# ---------- plot helpers ----------

def plot_loss_curves(loss_records, out_path):
    """
    loss_records: list of dicts with
      { "config": str, "steps": [..], "losses": [..] }
    """
    plt.figure()
    for rec in loss_records:
        if len(rec["steps"]) == 0:
            continue
        plt.plot(rec["steps"], rec["losses"], label=rec["config"])
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves for LoRA Configurations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy_bar(df, out_path):
    plt.figure()
    plt.bar(df["config"], df["accuracy"])
    plt.xlabel("LoRA Configuration")
    plt.ylabel("Accuracy on Held-Out Set")
    plt.title("LoRA Ablation: Accuracy per Configuration")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rank_vs_accuracy(df, out_path):
    plt.figure()
    plt.scatter(df["rank"], df["accuracy"])
    for _, row in df.iterrows():
        plt.text(row["rank"], row["accuracy"], row["config"])
    plt.xlabel("LoRA Rank (r)")
    plt.ylabel("Accuracy")
    plt.title("Effect of LoRA Rank on Accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_target_vs_accuracy(df, out_path):
    grouped = df.groupby("target_type")["accuracy"].mean().reset_index()
    plt.figure()
    plt.bar(grouped["target_type"], grouped["accuracy"])
    plt.xlabel("Target Module Type")
    plt.ylabel("Mean Accuracy")
    plt.title("Attention-Only vs Attention+MLP (Mean Accuracy)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

"""# Addition #3: Main sweep loop (training + evaluation + plots)"""

# LoRA sweep

results_rows = []
loss_records = []

for cfg in SWEEP_CONFIGS:
    cfg_name = cfg["name"]
    run_dir = os.path.join(SWEEP_OUTPUT_ROOT, cfg_name)
    os.makedirs(run_dir, exist_ok=True)

    print("\n===============================")
    print(f"[INFO] Starting config: {cfg_name}")
    print(f" r={cfg['r']}, alpha={cfg['alpha']}, dropout={cfg['dropout']}")
    print(f" target={cfg['target']}")
    print("===============================\n")

    # ---- Base model + processor (fresh for each run) ----
    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # match training setup
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # lora config for checkpoint
    lora_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # training arguments 
    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BSZ_PER_DEV,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",          
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],              
        remove_unused_columns=False,
        eval_strategy="no",         
    )

    trainer = FP16CLMTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()

    # lora adapter + processor for this config
    model.save_pretrained(run_dir)
    processor.save_pretrained(run_dir)

    # loss history
    history = [h for h in trainer.state.log_history if "loss" in h]
    steps = [h.get("step", i) for i, h in enumerate(history)]
    losses = [h["loss"] for h in history]
    loss_records.append(
        {"config": cfg_name, "steps": steps, "losses": losses}
    )

    # evaluate physio images
    print(f"[INFO] Evaluating config {cfg_name} on held-out set...")
    acc = evaluate_model_on_heldout(run_dir)
    print(f"[RESULT] Config {cfg_name}: accuracy={acc:.3f}")

    results_rows.append(
        {
            "config": cfg_name,
            "rank": cfg["r"],
            "alpha": cfg["alpha"],
            "dropout": cfg["dropout"],
            "target_type": cfg["target_type"],
            "accuracy": acc,
        }
    )

# results table 
df = pd.DataFrame(results_rows)
results_csv = os.path.join(SWEEP_OUTPUT_ROOT, "results.csv")
df.to_csv(results_csv, index=False)
df

"""# Addition #4: Generate figures from logged results"""

# save ablation accuracy figues

print("[INFO] Generating plots from sweep results...")

loss_curves_path = os.path.join(SWEEP_OUTPUT_ROOT, "loss_curves.png")
accuracy_bar_path = os.path.join(SWEEP_OUTPUT_ROOT, "accuracy_bar.png")
rank_vs_acc_path  = os.path.join(SWEEP_OUTPUT_ROOT, "rank_vs_accuracy.png")
target_vs_acc_path = os.path.join(SWEEP_OUTPUT_ROOT, "target_vs_accuracy.png")

plot_loss_curves(loss_records, loss_curves_path)
plot_accuracy_bar(df, accuracy_bar_path)
plot_rank_vs_accuracy(df, rank_vs_acc_path)
plot_target_vs_accuracy(df, target_vs_acc_path)

print("Saved:")
print(" ", loss_curves_path)
print(" ", accuracy_bar_path)
print(" ", rank_vs_acc_path)
print(" ", target_vs_acc_path)

"""# Evaluate a given adapter on the 5-class task"""

BEST_ADAPTER_DIR = "/content/lora_ablation_runs/r8"

import io

def strict_correct_5(gt_label, pred_text):
    """
    Stricter correctness:
    - identifies ALL phase labels mentioned in the model's output,
    - requires EXACTLY ONE label to appear,
    - and that single label must match the ground truth.
    """
    pred_lower = pred_text.lower()

    # find every relevant label the predicted output
    mentioned = []
    for label, substrings in label_substrings_5.items():
        if any(sub in pred_lower for sub in substrings):
            mentioned.append(label)

    # substring-match for correctness
    return len(mentioned) == 1 and mentioned[0] == gt_label


def evaluate_adapter_5class(adapter_dir, question):
    """
    Evaluate a LoRA adapter on the 5-class task.
    Returns accuracy and a DataFrame of per-image results.
    """
    # ensure previous models memory cleared 
    global ft_model, processor # global 
    if 'ft_model' in globals() and ft_model is not None:
        del ft_model
        ft_model = None
    if 'processor' in globals() and processor is not None:
        del processor
        processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # clear cache

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base, adapter_dir)
    ft_model.eval()

    rows = []
    correct = 0
    total = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for img_url in test_images:
        # phase token is still the 3rd substring
        phase = img_url.split("_")[2]
        if phase not in phase_map_5:
            # skip phases that are not being used
            continue

        gt_label = phase_map_5[phase]  # e.g. "stroop_stress"
        total += 1

        # load and run model
        resp = requests.get(img_url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
        ).to(ft_model.device)

        with torch.no_grad():
            out_ids = ft_model.generate(**inputs, max_new_tokens=128)

        pred_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        img.close()

        # extract assistant  
        assistant_prefix = "assistant\n"
        if assistant_prefix in pred_text:
            pred_text = pred_text.split(assistant_prefix, 1)[1]

        # check correctness. substring-matching again
        is_correct = strict_correct_5(gt_label, pred_text)

        if is_correct:
            correct += 1

        rows.append(
            {
                "image": img_url,
                "phase_token": phase,
                "ground_truth": gt_label,
                "prediction_text": pred_text,
                "correct": is_correct,
            }
        )

    acc = correct / total if total > 0 else 0.0
    df_results = pd.DataFrame(rows)
    return acc, df_results

BEST_ADAPTER_DIR = "/content/lora_ablation_runs/r8_mlp"  
dirs = ["/content/lora_ablation_runs/r16",
        "/content/lora_ablation_runs/r2",
        "/content/lora_ablation_runs/r4",
        "/content/lora_ablation_runs/r8",
        "/content/lora_ablation_runs/r8_do15",
        "/content/lora_ablation_runs/r8_mlp", ]

scores = {}
for dir in dirs:
  acc_5, df_5 = evaluate_adapter_5class(dir, q_5)
  print(f"5-class accuracy ({dir}): {acc_5:.3f}")
  scores[dir.split('/')[3]] = acc_5

scores

"""# Prompt Robustness Experiments"""

PROMPT_VARIANTS_5 = {
    "short_phase_name": (
        "Classify this recording into one exact phase: "
        "Baseline, Stroop_Stress, Math_Stress, MollyIntervention, or Post-Relaxation. "
        "Answer with only the phase name."
    ),
    "number_only": (
        "Classify this recording into one phase: "
        "1 = Baseline, 2 = Stroop_Stress, 3 = Math_Stress, 4 = MollyIntervention, 5 = Post-Relaxation. "
        "Respond with only the number."
    ),
    "explain": (
        "Look at the patterns in the physiological signals and determine which phase this recording corresponds to. "
        "You must choose exactly one of: Baseline, Stroop_Stress, Math_Stress, MollyIntervention, Post-Relaxation. "
        "State the phase name and then explain your reasoning in 1-2 sentences."
    ),
    "few_shot": (
        "We define phases as follows:\n"
        "- Baseline: calm, stable signals before any stress task.\n"
        "- Stroop_Stress: elevated arousal during a Stroop color-word interference task.\n"
        "- Math_Stress: elevated arousal during a mental arithmetic task.\n"
        "- MollyIntervention: intervention phase after stress.\n"
        "- Post-Relaxation: recovery period after relaxation.\n\n"
        "Given this definition, classify this recording into one exact phase. "
        "Answer with the phase name only."
    ),
    "noisy_synonyms": (
        "Is this segment more like a resting baseline, a high-stress challenge "
        "(Stroop or math), an intervention (Molly), or a cooldown/relaxation period "
        "(Post-Relaxation)? Answer with one of: Baseline, Stroop_Stress, Math_Stress, MollyIntervention, Post-Relaxation."
    ),
}

BEST_ADAPTER_DIR = "/content/lora_ablation_runs/r8_mlp"

def evaluate_prompt_variants_5class(adapter_dir, prompt_dict):
    """
    For a fixed adapter (best LoRA config), evaluate multiple prompt styles
    on the same 5-class task. Returns a DataFrame with accuracy per style.
    """
    rows = []
    for style, question in prompt_dict.items():
        print(f"\n[INFO] Evaluating prompt style: {style}")
        acc, df_res = evaluate_adapter_5class(adapter_dir, question)
        print(f"  accuracy = {acc:.3f}")
        rows.append(
            {
                "prompt_style": style,
                "accuracy": acc,
            }
        )
    return pd.DataFrame(rows)

df_prompt_5 = evaluate_prompt_variants_5class(BEST_ADAPTER_DIR, PROMPT_VARIANTS_5)
df_prompt_5

plt.figure()
plt.bar(df_prompt_5["prompt_style"], df_prompt_5["accuracy"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy (5-class)")
plt.title("Prompt Robustness for Fine-Tuned VLM (5-class)")
plt.tight_layout()
plt.show()

df_5[["image", "ground_truth", "prediction_text"]]