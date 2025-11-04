"""
Script to apply multiple persona steering vectors to a Llama‑3 model on a high‑performance computing (HPC) node.

This script assumes that you have previously trained persona vectors on Google Colab and saved them as
```${layer}_persona_vectors_respavg_{symptom}.pt``` files under ``persona_dir``. Each file contains a
dictionary called ``vectors`` mapping layer indices to hidden‑state offset tensors.  The HPC script
loads **all** of these files, combines them into a single dictionary, and then applies a selected
subset of the vectors simultaneously during generation.

Key features:

* Loads the model and tokenizer from a local directory (``model_dir``) to avoid network access.
* Loads every ``*.pt`` file in ``persona_dir`` and merges their ``vectors`` dicts.  Later
  definitions override earlier ones if two files provide a vector for the same layer.
* Computes a group of 7 consecutive layers to steer starting from ``layer_idx`` rounded down to
  the nearest multiple of 7.  For example, if ``layer_idx`` is 5, the script uses layers
  ``[0,1,2,3,4,5,6]``; if ``layer_idx`` is 13, it uses ``[7,8,9,10,11,12,13]``.
* Registers a forward hook for each selected layer and adds ``strength * vector`` to the hidden
  activations.  This means that a **negative** strength pushes the model away from the persona
  direction.
* Generates a continuation of the supplied prompt.  The default prompt uses the ``base_instruction``
  for a male or female veteran and asks for a life story.  You can modify ``user_input`` and
  ``base_instruction_*`` to suit your needs.

Usage example (run on HPC):

```bash
python hpc_persona_apply.py 1 -1.5 3 0
```

The positional arguments mean:

1. Index into the PANSS variable list (0=negative, 1=positive, 2=cognitive).  Currently this
   only influences the output file name.
2. Coefficient for the persona steering strength (float).  Negative values drive the model
   away from the persona direction.
3. Starting layer index (integer).  This determines which block of seven layers will be steered.
4. Gender flag (0=male, 1=female).  Selects the appropriate base instruction.

The script prints ten steered generations to STDOUT and also saves each output to a separate
``.txt`` file under ``save_word_dir``.  Files are named according to the variable, layer
block, coefficient, and gender.
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------------------
# Configuration parameters
# --------------------------------------------------------------------------------------

# Local model directory (assumes weights have been downloaded to this path)
model_dir = '/N/scratch/lleckie/models/llama-3.2-3B-Instruct'

# Working directory on the HPC scratch space
directory = '/N/u/lleckie/Quartz/work/SZ_steering/'

# Directory containing persona vector files (``*.pt``)

# PANSS variable names used for naming outputs.  They do not influence generation.
PANSS = [
    'PANSS_negative_sx_total_LysakerFactor',
    'PANSS_positive_sx_total_LysakerFactor',
    'PANSS_cognitive_sx_total_LysakerFactor',
]

# Base system instructions for male and female personas
base_instruction_m = (
    'Do not ask questions. Provide a single continuous narrative in the first person. '
    'You are a male veteran who attends health care appointments at a VA hospital in Indianapolis.'
)
base_instruction_f = (
    'Do not ask questions. Provide a single continuous narrative in the first person. '
    'You are a female veteran who attends health care appointments at a VA hospital in Indianapolis.'
)

# The user query that follows the system instruction
user_input = (
    'Tell me the story of your life, in as much detail as you can, from as early as you can remember up to now.'
)

# Number of layers to steer simultaneously.  Must match the value used during training.
sample_size = 7

# The granularity at which we shift the block of layers (e.g., 0–6, 7–13, etc.)
layer_increment = sample_size

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


panss_idx = int(sys.argv[1])
strength = float(sys.argv[2])
layer_idx = int(sys.argv[3])
gender_flag = int(sys.argv[4])
variable = PANSS[panss_idx]

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id


import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PersonaVectorConfig:
    layers: List[int]                  # Which layers to train vectors for
    batch_size: int = 8
    max_length: int = 512
    n_samples: Optional[int] = None    # Limit num samples per side if needed

def _validate_layers(layers: List[int], model) -> List[int]:
    n_layers = model.config.num_hidden_layers
    valid = []
    for L in layers:
        if L < 0 or L >= n_layers:
            print(f"[warn] Requested layer {L} is out of range [0..{n_layers-1}] and will be skipped.")
        else:
            valid.append(L)
    if not valid:
        raise ValueError("After validation, no valid layers remain.")
    return valid

def prepare_contrast_pairs(
    negative_responses: List[str],
    positive_responses: List[str],
    n_samples: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Returns list of (positive, negative) tuples for contrastive averaging.
    We operate on responses directly, so 'response-avg' == 'sequence-avg' here.
    """
    if n_samples:
        negative_responses = negative_responses[:n_samples]
        positive_responses = positive_responses[:n_samples]

    m = min(len(negative_responses), len(positive_responses))
    if m == 0:
        raise ValueError("Need at least 1 positive and 1 negative example.")
    negative_responses = negative_responses[:m]
    positive_responses = positive_responses[:m]
    return list(zip(positive_responses, negative_responses))

def _batch_mean_hidden(
    texts: List[str],
    model,
    tokenizer,
    layer_idx: int,
    max_length: int,
) -> torch.Tensor:
    """
    Get mean hidden state vector for a batch of texts at a given layer.
    Mean is over sequence tokens and then averaged across the batch.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1), each [B, T, H]
        # layer 0 == embeddings, layers 1..n == blocks. For HF Llama, using block index is OK.
        hs = out.hidden_states[layer_idx]  # [B, T, H]
        # Mask out padding so it doesn't dilute the signal
        # tokenizer pads with pad_token_id; build attention mask (1=valid,0=pad)
        attn = enc.get("attention_mask", torch.ones(hs.shape[:2], device=hs.device))  # [B, T]
        attn = attn.unsqueeze(-1).type_as(hs)  # [B, T, 1]
        summed = (hs * attn).sum(dim=1)                      # [B, H]
        lens = attn.squeeze(-1).sum(dim=1).clamp_min(1.0)    # [B]
        per_example_mean = summed / lens.unsqueeze(-1)       # [B, H]
        batch_mean = per_example_mean.mean(dim=0).cpu()      # [H]
        return batch_mean

def measure_projection_during_generate(
    model,
    tokenizer,
    chat_text: str,
    layer_idx: int,
    persona_vec: torch.Tensor,
    gen_kwargs: dict,
    strength: float = 0.0
) -> float:
    """
    Returns mean projection <h_t, v> across generated tokens at layer_idx.
    If strength != 0, applies steering at that layer while measuring.
    """
    layer_idx = _validate_layers([layer_idx], model)[0]
    v = persona_vec / (persona_vec.norm() + 1e-12)
    hooks = []
    if strength != 0.0:
        hook = model.model.layers[layer_idx].register_forward_hook(
            _make_layer_hook(v, strength)
        )
        hooks.append(_LayerHook(hook))
    try:
        enc = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        steps = out.decoder_hidden_states
        if steps is None or len(steps) == 0:
            raise RuntimeError("No decoder hidden states captured; check gen_kwargs.")
        projs = []
        for s in steps:
            h = s[layer_idx][:, -1, :]          # last position new token: [B,H]
            p = torch.matmul(h, v.to(h.device)) # [B]
            projs.append(p.mean().item())
        return float(np.mean(projs))
    finally:
        for h in hooks:
            h.remove()


def _response_avg_hidden_generate(
    prompts: list[str],
    model,
    tokenizer,
    layer_idx: int,
    gen_kwargs: dict
) -> torch.Tensor:
    """
    Capture per-step, per-batch hidden states at the target layer via a forward hook
    during generation, take the last token at each step, average over steps (response-avg),
    then average over batch. Returns [H].
    """
    layer_idx = _validate_layers([layer_idx], model)[0]
    buf = []  # will collect [B, H] per generation step

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output  # [B, T, H]
        buf.append(hidden[:, -1, :].detach().cpu())  # new token is the last position
        return output  # don't modify

    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    try:
        enc = tokenizer(prompts, return_tensors="pt#", padding=True).to(model.device)
        with torch.no_grad():
            # We don't need output_hidden_states here; the hook captures what we need
            _ = model.generate(**enc, **{k:v for k,v in gen_kwargs.items() if k not in ("output_hidden_states","return_dict_in_generate")})
    finally:
        handle.remove()

    if len(buf) == 0:
        raise RuntimeError("No steps captured. Check that max_new_tokens > 0 and the hook is attached to the correct layer.")

    tbh = torch.stack(buf, dim=0)   # [T, B, H]
    resp_avg = tbh.mean(dim=0)      # [B, H]
    batch_avg = resp_avg.mean(dim=0)  # [H]
    return batch_avg

def prepare_contrast_pairs(
    negative_responses: List[str],
    positive_responses: List[str],
    n_samples: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Returns list of (positive, negative) tuples for contrastive averaging.
    We operate on responses directly, so 'response-avg' == 'sequence-avg' here.
    """
    if n_samples:
        negative_responses = negative_responses[:n_samples]
        positive_responses = positive_responses[:n_samples]

    m = min(len(negative_responses), len(positive_responses))
    if m == 0:
        raise ValueError("Need at least 1 positive and 1 negative example.")
    negative_responses = negative_responses[:m]
    positive_responses = positive_responses[:m]
    return list(zip(positive_responses, negative_responses))

def _batch_mean_hidden(
    texts: List[str],
    model,
    tokenizer,
    layer_idx: int,
    max_length: int,
) -> torch.Tensor:
    """
    Get mean hidden state vector for a batch of texts at a given layer.
    Mean is over sequence tokens and then averaged across the batch.
    """
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1), each [B, T, H]
        # layer 0 == embeddings, layers 1..n == blocks. For HF Llama, using block index is OK.
        hs = out.hidden_states[layer_idx]  # [B, T, H]
        # Mask out padding so it doesn't dilute the signal
        # tokenizer pads with pad_token_id; build attention mask (1=valid,0=pad)
        attn = enc.get("attention_mask", torch.ones(hs.shape[:2], device=hs.device))  # [B, T]
        attn = attn.unsqueeze(-1).type_as(hs)  # [B, T, 1]
        summed = (hs * attn).sum(dim=1)                      # [B, H]
        lens = attn.squeeze(-1).sum(dim=1).clamp_min(1.0)    # [B]
        per_example_mean = summed / lens.unsqueeze(-1)       # [B, H]
        batch_mean = per_example_mean.mean(dim=0).cpu()      # [H]
        return batch_mean

def train_persona_vectors_response_avg(
    model,
    tokenizer,
    negative_prompts: list[str],   # chat-formatted strings for NEGATIVE system/persona
    positive_prompts: list[str],   # chat-formatted strings for POSITIVE system/persona
    layers: list[int],
    gen_kwargs: dict,
    n_samples: Optional[int] = None,
    save_path: Optional[str] = None
) -> Dict[int, torch.Tensor]:
    layers = _validate_layers(layers, model)
    if n_samples:
        negative_prompts = negative_prompts[:n_samples]
        positive_prompts = positive_prompts[:n_samples]
    m = min(len(negative_prompts), len(positive_prompts))
    if m == 0:
        raise ValueError("Need at least 1 positive and 1 negative prompt.")
    negative_prompts = negative_prompts[:m]
    positive_prompts = positive_prompts[:m]

    vectors = {}
    model.eval()
    for L in layers:
        #This is the generation, where we get the token-averaged hidden states for each pair, and then subtract them. I found this works better without nrm
        pos_avg = _response_avg_hidden_generate(positive_prompts, model, tokenizer, L, gen_kwargs)
        neg_avg = _response_avg_hidden_generate(negative_prompts, model, tokenizer, L, gen_kwargs)
        vec = pos_avg - neg_avg
    # nrm = vec.norm(p=2)
        vectors[L] = vec #if nrm.item() == 0 else vec / nrm
        print(f"[response-avg] layer {L}: norm={vectors[L].norm().item():.4f}")
    if save_path:
        save_persona_vectors(vectors, save_path)
    return vectors

def chat_prompt(tokenizer, system, user):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

def generate_only_new_text(model, tokenizer, text, **gen_kwargs):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def default_gen_kwargs(tokenizer, max_new_tokens=200, temperature=0.5, top_p=0.9):
    # Prefer an end-of-turn token if available
    eot_id = None
    for tok in ("<|eot_id|>", "<|eot|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                eot_id = tid
                break
        except Exception:
            pass
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        #top_p=top_p,
        eos_token_id=(eot_id if eot_id is not None else tokenizer.eos_token_id),
        return_dict_in_generate=True,  # for instrumentation
        output_hidden_states=True,     # for instrumentation
    )

# Build POS/NEG *chat* prompts around your trait (example: reality-contact vs delusional style)
# system_pos = "You are a 35-year-old man from Indiana. Answer factually, grounded in shared reality. Be concise and practical."
# system_neg = "You are a 35-year-old man from Indiana. Answer in a grandiose, conspiratorial, reality-detached style."


def save_persona_vectors(vectors: Dict[int, torch.Tensor], path: str):
    safe = {int(k): v.cpu() for k, v in vectors.items()}
    torch.save({
        "vectors": safe,
        "metadata": {
            "layers": sorted(safe.keys()),
            "hidden_size": next(iter(safe.values())).shape[0] if safe else None
        }
    }, path)
    print(f"Saved persona vectors to {path}")

def load_persona_vectors(path: str) -> Dict[int, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    return {int(k): v for k, v in obj["vectors"].items()}

def build_chat_pairs(tokenizer, symptom_pair: str, question: str, copies_per_prompt: int = 1):
    """
    Given a cluster name (e.g., 'negative'), return pos_chat and neg_chat lists.
    Each chat is repeated `copies_per_prompt` times for stability.
    """
    if symptom_pair not in clusters:
        raise ValueError(f"Unknown symptom_pair '{symptom_pair}'. Choose from {list(clusters.keys())}")

    pos_list, neg_list = clusters[symptom_pair]
    pos_chat, neg_chat = [], []

    for pos_prompt, neg_prompt in zip(pos_list, neg_list):
        for _ in range(copies_per_prompt):
            pos_chat.append(chat_prompt(tokenizer, pos_prompt, question))
            neg_chat.append(chat_prompt(tokenizer, neg_prompt, question))

    return pos_chat, neg_chat


# Pick a layer; you can sweep later


base_instruction_m='You are a male veteran who attends health care appointments at a VA hospital in Indianapolis.'
base_instruction_f='You are a female veteran who attends health care appointments at a VA hospital in Indianapolis.'

base_instructions = [base_instruction_m, base_instruction_f]
base_instruction = base_instructions[gender_flag]
positive_symptoms=[f"{base_instruction} You are experiencing delusions and these are interfering with your life.",
f"{base_instruction} You are experiencing sensory hallucinations and these are interfering with your life.",
f"{base_instruction} You have bizarre ideas and illogical beliefs.",
f"{base_instruction} You are experiencing delusions that you have a severe physical disease or illness.",
f"{base_instruction} You falsely believe that others are plotting against you and intend to harm you.",
f"{base_instruction} You have unrealistic beliefs that you have special abilities, wealth, knowledge, or power."
]


negative_symptoms =[f"{base_instruction} You are uninterested and apathetic towards social interactions.",
f"{base_instruction} You are uninterested in participating in daily activities and you feel distant and detached.",
f"{base_instruction} You are emotionally unresponsive and you have blunted facial and vocal expression.",
f"{base_instruction} Your verbal responses are brief and your communication lacks flow.",
f"{base_instruction} You are distant towards others and are closed off in conversation.",
f"{base_instruction} You struggle to make decisions and control your own behavior.",
f"{base_instruction} You are absorbed with your own thoughts and feelings and you are not paying attention to other people or what is happening around you.",
f"{base_instruction} Your movements and speech are slowed and you spend much of your time lying down.",
]


cognitive_symptoms =[f"{base_instruction} You are unable to think abstractly and you interpret what people say in an overly concrete manner.",
f"{base_instruction} Your thinking is stereotyped, rigid, inflexible, and repetitious.",
f"{base_instruction} Your thinking is disorganized and your speech is tangential.",
f"{base_instruction} You lack judgement and insight into your own life situation and do not recognize that you have a mental illness.",
f"{base_instruction} You are experiencing physical symptoms of anxiety and agitation such as tension, restlessness, shaking, fidgeting, and pacing.",
f"{base_instruction} Your movements, posture, and mannerisms are awkward and unnatural."
]


# Example: Here input your prompt pairs. I use 4 symptom clusters, so I have created a dictionary of 4 pairs. But only one is necessary
clusters = {
    "positive": (positive_symptoms, [base_instruction for i in range(0,len(positive_symptoms))]),
    "negative": (negative_symptoms, [base_instruction for i in range(0,len(negative_symptoms))]),
    "cognitive": (cognitive_symptoms, [base_instruction for i in range(0,len(cognitive_symptoms))])}


save_dir_m = os.path.join(directory, 'persona_vectors_3/')
os.makedirs(save_dir_m, exist_ok=True)
save_dir_f = os.path.join(directory, 'persona_vectors_f_3/')
os.makedirs(save_dir_f, exist_ok=True)

save_dirs=[save_dir_m, save_dir_f]
save_dir=save_dirs[gender_flag]
symptom_pairs=['negative', 'positive', 'cognitive']
symptom_pair=symptom_pairs[panss_idx]
# Example usage:
  #question = "Tell me your life story."
user_input = "Tell me the story of your life, in as much detail as you can, from as early as you can remember up to now."
pos_chat, neg_chat = build_chat_pairs(tokenizer, symptom_pair, user_input, copies_per_prompt=3)



#Generate steering vectors
print(len(pos_chat), len(neg_chat))  # should be same length
print(pos_chat[0])  # inspect an example
#Control the model hyperparameters
gen_kwargs = default_gen_kwargs(tokenizer, max_new_tokens=5000)
#layers = list(range(1, 28))

for layers in [[i,i+1] for i in range(1,30-1,1)]:
# Train vectors with response-avg (recommended)
    print('Training layers:', layers)
    vectors = train_persona_vectors_response_avg(
        model, tokenizer,
        negative_prompts=neg_chat,
        positive_prompts=pos_chat,
        layers=layers,
        gen_kwargs=gen_kwargs,
        n_samples=None,
        save_path=f"{save_dir}{layers[0]}_persona_vectors_respavg_{symptom_pair}.pt"
    )