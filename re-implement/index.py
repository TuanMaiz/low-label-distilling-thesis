import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from models.student_config import load_student_config

CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "students"
    / "qwen3_reranker_0_6b.json"
)
# Keep the model architecture, answer tokens, LoRA settings, and input limit in
# one shared config so this walkthrough uses the same contract as the real run.
config = load_student_config(CONFIG_PATH)

TARGETS = (
    REPO_ROOT
    / "data"
    / "cache"
    / "wdc_products"
    / "targets"
    / "train_128.gold_random.targets.jsonl"
)
RECORD_A_MARKER = "\n\nRecord A:\n"
RECORD_B_MARKER = "\n\nRecord B:\n"

# Start with one cached, labeled pair. This script demonstrates one training
# example rather than iterating over the complete 128-row training dataset.
with TARGETS.open(encoding='utf-8') as file:
    row = json.loads(next(file))

print("pair_id:", row["pair_id"])
print("target_text:", row["target_text"])
print("input_text:", row["input_text"])

def split_records(input_text: str) -> tuple[str, str]:
    # Cached examples contain both product records in one serialized string.
    # Recover them before mapping them to Qwen's Query/Document interface.
    if RECORD_A_MARKER not in input_text:
        raise ValueError("Record A marker is missing")
    if RECORD_B_MARKER not in input_text:
        raise ValueError("Record B marker is missing")

    _, pair_text = input_text.split(RECORD_A_MARKER, maxsplit=1)
    record_a, record_b = pair_text.split(RECORD_B_MARKER, maxsplit=1)

    return (
        f"Record A:\n{record_a}",
        f"Record B:\n{record_b}",
    )

def format_reranker_pair(input_text, instruction)-> str:
    # Entity matching is expressed as a reranking task: Record A is the query,
    # Record B is the document, and the instruction defines what "relevant"
    # means (the same real-world product).
    record_a, record_b = split_records(input_text)

    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {record_a}\n"
        f"<Document>: {record_b}"
    )

RERANKER_PREFIX = (
    # These are Qwen chat-template control tokens, not ordinary prompt words.
    # The system message constrains the next answer to exactly "yes" or "no".
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n'
    "<|im_start|>user\n"
)

RERANKER_SUFFIX = (
    # End the user turn and place the model at the assistant answer position.
    # The reranker checkpoint expects this empty-thinking suffix before its
    # final relevance answer.
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

payload = format_reranker_pair(row["input_text"], config.reranker_instruction)

prompt = RERANKER_PREFIX + payload + RERANKER_SUFFIX
print("\n--- COMPLETE PROMPT ---")
print(prompt)

# Convert the complete prompt into model token IDs and verify that each class
# answer occupies exactly one vocabulary position.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
	config.model_name,
	use_fast=config.tokenizer_use_fast,
	padding_side = 'left'
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def enforce_one_token(tokenizer, text: str) -> str:
    # Later we select one logit for each answer. That operation is valid only
    # when each configured answer is represented by exactly one token.
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r} must be exactly one token, got {token_ids}"
        )
    return token_ids[0]

assert config.reranker_negative_token is not None
assert config.reranker_positive_token is not None

no_token_id = enforce_one_token(tokenizer, config.reranker_negative_token)
yes_token_id = enforce_one_token(tokenizer, config.reranker_positive_token)

if no_token_id == yes_token_id:
    raise ValueError("'no' and 'yes' resolved to the same token")

print("no token ID:", no_token_id)
print("yes token ID:", yes_token_id)
print("decoded no:", tokenizer.decode([no_token_id]))
print("decoded yes:", tokenizer.decode([yes_token_id]))

# Do not silently truncate either product. The experiment contract requires
# complete inputs and fails explicitly when a pair exceeds the configured cap.
inputs = tokenizer(
    prompt,
    add_special_tokens=False,
    return_tensors="pt",
    truncation=False
)

token_count = inputs["input_ids"].shape[1]
print("Input token count:", token_count)
print("Maximum allowed:", config.max_input_length)


if token_count > config.max_input_length:
    raise ValueError(
        f"Input requires {token_count} tokens, "
        f"but the limit is {config.max_input_length}"
    )

print(
    "Final prompt tokens:",
    tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0, -20:].tolist()
    ),
)

# Load the public base reranker. The fresh LoRA adapter is attached below; this
# walkthrough intentionally does not load an already-trained adapter.
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
# Key/value caching accelerates generation but conflicts with checkpointed
# training, so it is disabled for this backward-pass demonstration.
base_model.config.use_cache = False

if config.gradient_checkpointing:
    # Save GPU memory by recomputing selected activations during backward().
    base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "enable_input_require_grads"):
        # The base model is frozen by LoRA. Marking embedding outputs for
        # gradient tracking keeps the graph connected through checkpointed
        # layers so gradients can reach the trainable LoRA matrices.
        base_model.enable_input_require_grads()

lora_config = LoraConfig(
    # LoRA adds small trainable low-rank matrices to the selected attention
    # projections instead of updating all 0.6B base-model parameters.
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.lora_target_modules,
    bias="none",
)

training_model = get_peft_model(
    base_model,
    lora_config
)

training_model.to(device)
training_model.train()
training_model.config.use_cache = False
training_model.print_trainable_parameters()

# This is a safety check for the experiment: only parameters introduced by
# LoRA should be trainable; the original reranker weights must remain frozen.
unexpected_trainable = [
    name
    for name, parameter in training_model.named_parameters()
    if parameter.requires_grad and "lora_" not in name
]

if unexpected_trainable:
    raise RuntimeError(
        f"Unexpected trainable base parameters: {unexpected_trainable}"
    )

# PyTorch operations require model parameters and input tensors on the same
# device. In Colab this moves the token IDs and attention mask onto the GPU.
inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

# Convert the cached text label to the binary class index expected by
# cross_entropy: non-match -> 0 and match -> 1.
normalized_target = (
    row["target_text"]
    .strip()
    .lower()
    .replace("_", "-")
)
if normalized_target not in config.label_to_id:
    raise ValueError(
        f"Unsupported target: {row['target_text']!r}"
    )
label_id = config.label_to_id[normalized_target]

labels = torch.tensor(
    [label_id],
    dtype=torch.long,
    device=device,
)

training_model.zero_grad(set_to_none=True)

# This is the single gradient-enabled forward pass. Calling the model builds
# the computation graph that loss.backward() will traverse.
outputs = training_model.forward(
    **inputs,
    use_cache=False,
    logits_to_keep=1,
)
# logits_to_keep=1 asks Qwen for only the final answer position. The model still
# emits a score for every vocabulary token at that position.
final_vocabulary_logits = outputs.logits[:, -1, :]

# Reduce the full vocabulary to the two allowed answers. Their order matches
# label_to_id: [no/non-match, yes/match].
no_logit = final_vocabulary_logits[:, no_token_id]
yes_logit = final_vocabulary_logits[:, yes_token_id]


binary_logits = torch.stack(
    [no_logit, yes_logit],
    dim=-1,
)

binary_probabilities = torch.softmax(
    # Normalize only across "no" and "yes". This conditional two-class
    # probability is what maps the generative reranker to entity matching.
    binary_logits.float(),
    dim=-1,
)

non_match_probability = binary_probabilities[0, 0].item()
match_probability = binary_probabilities[0, 1].item()

print("\n--- RAW SCORES ---")
print("no logit:", no_logit.item())
print("yes logit:", yes_logit.item())

print("\n--- NORMALIZED BINARY SCORES ---")
print("non-match:", non_match_probability)
print("match:", match_probability)
print("sum:", non_match_probability + match_probability)

import torch.nn.functional as F
# Cross-entropy increases the correct answer token's logit relative to the
# incorrect answer token. It consumes raw logits, not softmax probabilities.
loss = F.cross_entropy(
    binary_logits.float(),
    labels,
)

print("\n--- SCORES ---")
print("no logit:", no_logit.item())
print("yes logit:", yes_logit.item())
print("non-match score:", non_match_probability)
print("match score:", match_probability)
print("target:", normalized_target)
print("target class:", label_id)
print("loss:", loss.item())

# Backpropagate the classification error through Qwen into the LoRA matrices.
# No optimizer.step() is used here, so parameters are checked but not updated.
loss.backward()

# A 0.5 threshold is suitable for this demonstration because the two selected
# probabilities sum to one. A trained experiment should use its persisted
# validation-selected threshold instead.
threshold = 0.5

prediction = (
    "match"
    if match_probability >= threshold
    else "non-match"
)

print("\n--- DECISION ---")
print("Prediction:", prediction)
print("Gold target:", row["target_text"])
print("Correct:", prediction == row["target_text"])

# Verify that backward() really reached the adapter: every observed LoRA
# gradient must be finite, and at least one must carry a nonzero signal.
gradient_norms = {}

for name, parameter in training_model.named_parameters():
    if "lora_" not in name or parameter.grad is None:
        continue

    gradient = parameter.grad.detach()

    if not torch.isfinite(gradient).all():
        raise RuntimeError(
            f"Non-finite gradient: {name}"
        )

    gradient_norms[name] = (
        gradient.float().norm().item()
    )

if not gradient_norms:
    raise RuntimeError(
        "No LoRA parameter received a gradient"
    )

if not any(norm > 0 for norm in gradient_norms.values()):
    raise RuntimeError(
        "All LoRA gradients are zero"
    )

print("\n--- LORA GRADIENTS ---")

for name, norm in gradient_norms.items():
    print(f"{norm:.8e}  {name}")

print("\nLoRA backward-path verification passed")
