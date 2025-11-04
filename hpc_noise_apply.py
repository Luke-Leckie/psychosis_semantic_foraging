"""
Script to apply uniform noise to selected decoder layers of a Llama‑3 model on a high‑performance
computing (HPC) node.

This script mirrors the overall structure of ``hpc_persona_apply.py``.  Instead of loading
pretrained steering vectors and adding them to layer activations, it injects uniform random
noise into the activations of a contiguous block of layers during generation.  The noise
amplitude is controlled by a command‑line coefficient.  Larger values produce larger
perturbations.  The noise is sampled independently on every forward pass, and is broadcast
across the token dimension (i.e., the same noise is added to all positions in the sequence for
each layer).  Negative coefficients are permitted but only their absolute value is used to
determine the amplitude.

Key features:

* Loads the model and tokenizer from a local directory (``model_dir``) to avoid network
  access on the HPC node.
* Selects a block of seven consecutive layers to perturb, starting from ``layer_idx``
  multiplied by the block size (e.g., block 0 covers layers 0–6, block 1 covers 7–13, etc.).
* Registers a forward hook for each selected layer.  The hook samples uniform noise in
  ``[-|strength|, |strength|]`` for each hidden dimension and adds it to the decoder layer
  output.  The same noise vector is broadcast across all tokens in the sequence.
* Generates multiple continuations of the supplied prompt.  The default prompt uses the
  ``base_instruction`` for a male or female veteran and asks for a life story.  You can modify
  ``user_input`` and ``base_instruction_*`` to suit your needs.
* Prints the noise‑perturbed generations to STDOUT and also saves each output to a separate
  ``.txt`` file under ``save_word_dir``.  Files are named according to the PANSS variable
  (unused here beyond naming), the layer block index, the coefficient, the generation index,
  and the gender flag.

Usage example (run on HPC):

```bash
python hpc_noise_apply.py 1 0.5 3 0
```

The positional arguments mean:

1. Index into the PANSS variable list (0=negative, 1=positive, 2=cognitive).  Currently this
   only influences the output file name.
2. Coefficient determining the amplitude of the uniform noise (float).  The absolute value of
   this coefficient is used; negative values simply flip the sign but have no effect on the
   magnitude of the noise.
3. Block index of layers to perturb (integer).  If ``layer_idx`` is 0, layers ``[0,1,2,3,4,5,6]``
   are perturbed; if ``layer_idx`` is 2, layers ``[14,15,16,17,18,19,20]`` are perturbed, etc.
4. Gender flag (0=male, 1=female).  Selects the appropriate base instruction.

The script prints ten perturbed generations to STDOUT and also saves each output to a separate
``.txt`` file under ``save_word_dir``.  Files are named according to the variable, layer
block index, coefficient, and gender.
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
save_direc='/N/scratch/lleckie/SZ_steering/'
# Directory where generated outputs will be saved
save_word_dir = os.path.join(save_direc, 'noise_outputs')
os.makedirs(save_word_dir, exist_ok=True)

# PANSS variable names used for naming outputs.  They do not influence generation.
PANSS = [
    'PANSS_negative_sx_total_LysakerFactor',
    'PANSS_positive_sx_total_LysakerFactor',
    'PANSS_cognitive_sx_total_LysakerFactor',
]

# Base system instructions for male and female personas
base_instruction_m = (
    #'Do not ask questions. Provide a single continuous narrative in the first person. '
    'You are a male veteran who attends health care appointments at a VA hospital in Indianapolis.'
)
base_instruction_f = (
   # 'Do not ask questions. Provide a single continuous narrative in the first person. '
    'You are a female veteran who attends health care appointments at a VA hospital in Indianapolis.'
)

# The user query that follows the system instruction
user_input = (
    'Tell me the story of your life, in as much detail as you can, from as early as you can remember up to now.'
)

# Number of layers to perturb simultaneously.  Must match the value used when selecting blocks.
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


class _LayerHook:
    """RAII wrapper to manage forward hooks on model layers."""

    def __init__(self, handle):
        self.handle = handle

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _make_noise_hook(strength: float):
    """
    Create a forward hook that adds uniform noise to the hidden state at every position in
    the decoder layer output.

    ``strength`` controls the amplitude of the noise.  The noise for each hidden dimension
    is sampled independently from a uniform distribution in ``[-|strength|, |strength|]``.  The
    same noise vector is broadcast across the token dimension.  The absolute value of
    ``strength`` is used to determine the range; negative values simply flip the sign of the
    range but have no effect on the magnitude.
    """
    amp = abs(strength)

    def hook(module, inputs, output):
        # HF ``LlamaDecoderLayer`` returns either a Tensor or a tuple where the first element
        # is the hidden state.  We must preserve the structure of the output.
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Sample noise: one vector per hidden dimension, broadcast over tokens.
        # hidden shape: [batch, seq_len, hidden_size]
        # We create shape [1, 1, hidden_size] so the same noise is added to all positions.
        device = hidden.device
        dtype = hidden.dtype
        hidden_size = hidden.shape[-1]
        noise = torch.empty(1, 1, hidden_size, dtype=dtype, device=device).uniform_(-amp, amp)
        perturbed = hidden + noise

        if rest is None:
            return perturbed
        else:
            return (perturbed,) + rest

    return hook


def apply_uniform_noise(
    model,
    tokenizer,
    prompt: str,
    layers: list,
    strength: float,
    max_new_tokens: int = 4000,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Apply uniform noise to the activations of the specified layers and generate text.

    Only layers present in ``layers`` will be perturbed.  If a layer index is outside the
    range of the model (e.g., larger than the number of layers), it is ignored.  Noise hooks
    are registered before generation and removed afterwards.
    """
    n_layers = model.config.num_hidden_layers
    valid_layers = [L for L in layers if 0 <= L < n_layers]
    if not valid_layers:
        raise ValueError(f'No valid layers in {layers} for model with {n_layers} layers.')

    hooks = []
    for L in valid_layers:
        handle = model.model.layers[L].register_forward_hook(
            _make_noise_hook(strength)
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
            'Usage: python hpc_noise_apply.py <panss_index> <strength> <layer_idx> <gender>',
            file=sys.stderr,
        )
        sys.exit(1)

    panss_idx = int(sys.argv[1])
    strength = float(sys.argv[2])
    layer_idx = int(sys.argv[3])
    gender_flag = int(sys.argv[4])

    if panss_idx < 0 or panss_idx >= len(PANSS):
        raise ValueError(f'PANSS index must be 0–{len(PANSS) - 1}')

    variable = PANSS[panss_idx]

    # Load tokenizer and model from local directory.
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

    # Ensure pad token is defined for left padding during generation.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Select base system prompt according to gender flag.
    if gender_flag == 0:
        system_prompt = base_instruction_m
        gender_label = 'male'
    elif gender_flag == 1:
        system_prompt = base_instruction_f
        gender_label = 'female'
    else:
        raise ValueError('gender must be 0 (male) or 1 (female)')

    prompt = chat_prompt(tokenizer, system_prompt, user_input)

    # Determine which block of layers to perturb.
    start_layer = layer_idx * layer_increment
    layers_to_perturb = list(range(start_layer, start_layer + sample_size))

    # Print configuration summary
    print(f'Variable    : {variable}')
    print(f'Strength    : {strength}')
    print(f'Layers      : {layers_to_perturb}')
    print(f'Gender      : {gender_label}')
    print(f'Prompt (truncated to 80 chars): {prompt[:80]}...')
    print('-' * 60)

    for i in range(30):
        fname = (
            f'noise_block{layer_idx}_coeff{strength:+.3f}_gen{i + 1}_{gender_label}.txt'
        )
        if os.path.exists(os.path.join(save_word_dir, fname)):
            print(f'Skipping generation {i + 1}: {fname} already exists.')
            continue
        result = apply_uniform_noise(
            model,
            tokenizer,
            prompt,
            layers_to_perturb,
            strength,
            max_new_tokens=5000,
        )
        print(f'--- Generation {i + 1} ---')
        print(result)
        print()

        out_path = os.path.join(save_word_dir, fname)
        with open(out_path, 'w') as fh:
            fh.write(result)


if __name__ == '__main__':
    main()
