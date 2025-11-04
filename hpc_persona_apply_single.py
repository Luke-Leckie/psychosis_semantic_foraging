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
persona_dir_m = os.path.join(directory, 'persona_vectors_3')
persona_dir_f = os.path.join(directory, 'persona_vectors_f_3')

save_direc='/N/scratch/lleckie/SZ_steering/'
# Directory where generated outputs will be saved
save_word_dir = os.path.join(save_direc, 'steered_outputs_single')
os.makedirs(save_word_dir, exist_ok=True)

# PANSS variable names used for naming outputs.  They do not influence generation.
PANSS = [
    'PANSS_negative_sx_total_LysakerFactor',
    'PANSS_positive_sx_total_LysakerFactor',
    'PANSS_cognitive_sx_total_LysakerFactor',
]

# Base system instructions for male and female personas
base_instruction_m = (
 #   'Do not ask questions. Provide a single continuous narrative in the first person. '
    'You are a male veteran who attends health care appointments at a VA hospital in Indianapolis.'
)
base_instruction_f = (
#    'Do not ask questions. Provide a single continuous narrative in the first person. '
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

def chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat prompt from system and user messages."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_persona_vectors_all(directory: str, symptom: str) -> dict:
    """
    Load every ``*.pt`` persona vector file in ``directory`` and return a unified mapping.

    Each file is expected to contain a dict with a key ``"vectors"`` pointing to a mapping
    of layer indices (integers) to tensors of shape [hidden_size].  If multiple files define
    a vector for the same layer, the vector from the later file in alphabetical order wins.
    """
    all_vecs = {}
    files_list = sorted([ f for f in os.listdir(directory) if symptom in f ])
    print(files_list)
    for fname in files_list:
        if not fname.endswith('.pt'):
            continue
        path = os.path.join(directory, fname)
        try:
            obj = torch.load(path, map_location='cpu')
        except Exception:
            continue
        if not isinstance(obj, dict) or 'vectors' not in obj:
            continue
        vectors = obj['vectors']
        for layer, vec in vectors.items():
            if not isinstance(vec, torch.Tensor):
                vec = torch.tensor(vec)
            all_vecs[int(layer)] = vec
    return all_vecs


class _LayerHook:
    """RAII wrapper to manage forward hooks on model layers."""
    def __init__(self, handle):
        self.handle = handle

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _make_layer_hook(persona_vec: torch.Tensor, strength: float):
    """
    Create a forward hook that adds ``strength * persona_vec`` to the hidden state
    at every position in the decoder layer output.
    """
    persona_vec = persona_vec.detach().clone()

    def hook(module, inputs, output):
        # HF ``LlamaDecoderLayer`` returns either a Tensor or a tuple where the first element
        # is the hidden state.  We must preserve the structure of the output.
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        v = persona_vec.to(hidden.device).view(1, 1, -1)
        steered = hidden + strength * v

        if rest is None:
            return steered
        else:
            return (steered,) + rest

    return hook


def apply_persona_vectors(
    model,
    tokenizer,
    prompt: str,
    vectors: dict,
    layers: list,
    strength: float,
    max_new_tokens: int = 4000,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Apply multiple persona vectors to their corresponding layers simultaneously and generate text.

    Only layers present in ``layers`` and ``vectors`` will be steered.  Missing vectors are ignored.
    """
    n_layers = model.config.num_hidden_layers
    valid_layers = [L for L in layers if 0 <= L < n_layers]
    if not valid_layers:
        raise ValueError(f'No valid layers in {layers} for model with {n_layers} layers.')

    hooks = []
    for L in valid_layers:
        if L not in vectors:
            continue
        vec = vectors[L]
        handle = model.model.layers[L].register_forward_hook(
            _make_layer_hook(vec, strength)
        )
        hooks.append(_LayerHook(handle))
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=0.5#,
                #top_p=top_p,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()


def main():
    if len(sys.argv) != 5:
        print(
            'Usage: python hpc_persona_apply.py <panss_index> <strength> <layer_idx> <gender>',
            file=sys.stderr,
        )
        sys.exit(1)

    panss_idx = int(sys.argv[1])
    strength = float(sys.argv[2])
    layer_idx = int(sys.argv[3])
    gender_flag = int(sys.argv[4])
    symptom_pairs=['negative', 'positive', 'cognitive']
    symptom = symptom_pairs[panss_idx]
    if panss_idx < 0 or panss_idx >= len(PANSS):
        raise ValueError(f'PANSS index must be 0–{len(PANSS) - 1}')

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

    if gender_flag == 0:
        system_prompt = base_instruction_m
        gender_label = 'male'
        persona_dir=persona_dir_m
    elif gender_flag == 1:
        system_prompt = base_instruction_f
        gender_label = 'female'
        persona_dir=persona_dir_f
    else:
        raise ValueError('gender must be 0 (male) or 1 (female)')

    prompt = chat_prompt(tokenizer, system_prompt, user_input)

    persona_vectors = load_persona_vectors_all(persona_dir,symptom)
    if not persona_vectors:
        raise RuntimeError(f'No persona vector files found in {persona_dir}')

    # Determine which block of layers to steer
    start_layer = layer_idx * layer_increment
    layers_to_steer = list(range(start_layer, start_layer + sample_size))

    # Print configuration summary
    print(f'Variable    : {variable}')
    print(f'Strength    : {strength}')
    print(f'Layers      : {layers_to_steer}')
    print(f'Gender      : {gender_label}')
    print(f'Prompt (truncated to 80 chars): {prompt[:80]}...')
    print('-' * 60)

    for i in range(30):
        for steering_layer in [k for k in  range(0, 28)]:
            fname = (
                f'{variable}_block{steering_layer}_coeff{strength:+.2f}_gen{i + 1}_{gender_label}.txt'
            )
            if os.path.exists(os.path.join(save_word_dir, fname)):
                print(f'Skipping generation {i + 1}: {fname} already exists.')
                continue
            result = apply_persona_vectors(
                model,
                tokenizer,
                prompt,
                persona_vectors,
                [steering_layer],
                strength,
                max_new_tokens=5000,
            )
            print(f'--- Generation {i + 1} ---')
            print(result)

            out_path = os.path.join(save_word_dir, fname)
            with open(out_path, 'w') as fh:
                fh.write(result)


if __name__ == '__main__':
    main()
