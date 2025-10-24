# -*- coding: utf-8 -*-
"""
Relation extraction (drug ↔ disease) using an instruction-tuned causal LM.

Pipeline summary:
1) Build instruction prompts from dataset examples.
2) Fine-tune a causal LM with supervised next-token learning on full prompt+label.
3) Evaluate by *generating* a JSON answer and mapping it back to a class label.
4) Report F1 (both macro and a custom "positive-only" F1).

Notes:
- Label id 0 is treated as "other" and is excluded from the F1's P/R counts.
- We disable W&B logging via WANDB_DISABLED to avoid accidental uploads.
- compute_metrics re-runs generation on the validation split each eval epoch (slow).
"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import argparse
import os
from tqdm import tqdm
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
# from huggingface_hub import login  # Imported but not used; keep if you log in elsewhere.
from sklearn.metrics import f1_score

# Disable Weights & Biases by default (avoid accidental logging in shared servers).
os.environ["WANDB_DISABLED"] = "true"

# Track best validation F1 across epochs for logging
best_f1 = -1.0
best_f1_targets, best_f1_preds = [], []  # (Not used; kept for potential future logging)


def compute_f1(preds, labels):
    """
    Compute:
      - macro_f1: sklearn macro-F1 across both classes (0/1).
      - custom 'positive-only' P/R/F1: treats any non-zero as positive and requires exact class
        match for correctness. With 2 classes, this mirrors standard binary P/R/F1 where 1 is positive.

    Args:
      preds (List[int]): predicted class ids
      labels (List[int]): gold class ids

    Returns:
      dict with precision, recall, f1, macro_f1, counts
    """
    n_gold = n_pred = n_correct = 0
    macro_f1 = f1_score(labels, preds, average='macro')

    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1

    if n_correct == 0:
        # No true positives → everything zeroed
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'macro_f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        f1 = 2.0 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        print('performance: ', {
            'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
            'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold
        })
        return {
            'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
            'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold
        }


# System instruction: role + required output format (JSON with answer_choice)
system_prompt = (
    'You are a helpful medical expert, and your task is to extract the relation '
    'between the given entity pairs. Please choose the relation label from the '
    'provided options. Organize your output in a json formatted as '
    'Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research '
    'purposes only, so please have a definite answer.\n\n'
)

# The choices shown to the model in binary form
relation_options = '''
"A": "positive treatment",
"B": "negative/no treatment",
'''

# Collapse 4-way labels → 2-way labels for evaluation
label_convertion = {
    "treats": "positive treatment",
    "conditional treatment": "positive treatment",
    "negative treatment": "negative/no treatment",
    "other": "negative/no treatment",
}

# Gold label → multiple choice letter for supervision
label_mapping = {
    "treats": "A",
    "conditional treatment": "A",
    "negative treatment": "B",
    "other": "B",
}

# Model answer letter → 2-way label
answer_mapping = {
    "A": "positive treatment",
    "B": "negative/no treatment",
}

# Numeric ids (used by metrics)
rel2id = {
    "positive treatment": 1,
    "negative/no treatment": 0,
}

# A small demonstration header to stabilize JSON formatting
instruction = '''
### User:
Here is the input text for relation extraction:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please generate your output in json.

### Assistant:
{"answer_choice": "X"}

'''

# Example-specific prompt template
prompt = '''
### User:
Here is the input text for relation extraction:
{text}

What is the potential relationship between the drug {subj} and the disease {obj}?

Here are the potential choices:
{relation_options}

Please generate your output in json.

### Assistant:
'''


def input_fun(examples, relation_options, system_prompt, instruction, prompt):
    """
    Build the full input strings for each example:
      [system_prompt] + [instruction] + [prompt(text, subj, obj, choices)]
    """
    texts = examples['context']
    subjs = examples['subject']
    objs = examples['object']
    prompt_text = []

    for text, subj, obj in zip(texts, subjs, objs):
        component = prompt.format(text=text, subj=subj, obj=obj, relation_options=relation_options)
        prompt_text.append(system_prompt + instruction + component)

    return prompt_text


def output_fun(examples, label_mapping):
    """
    Convert gold labels into the exact JSON string the model should output.
    We train the LM to reproduce this JSON verbatim as a sequence.
    """
    labels = [label_mapping[l] for l in examples['label']]
    return ['{' + f'"answer_choice": "{label}"' + '}' for label in labels]


def extract_label(dct):
    """
    Parse model JSON → human-readable 2-way label.
    """
    return answer_mapping[dct['answer_choice']]


def main(args):
    # -------------------------------------------------------------------------
    # Dataset & model setup
    # -------------------------------------------------------------------------
    access_token = args.access_token

    dataset_name = args.dataset_name
    # trust_remote_code in case the dataset defines a custom builder
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    print('model/tokenizer preparing...')
    # NOTE: model_id is hard-coded here; args.model_type is defined but not used.
    # If you want configurability, pass via CLI and read args.model_id.
    model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    tokenizer_kwargs = {
        "additional_special_tokens": ['<obj>', '<subj>', '<subj/>', '<obj/>'],
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side='left', token=access_token, **tokenizer_kwargs
    )
    # Many decoder-only LMs don't have a pad token; re-use eos to avoid warnings.
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=access_token)
    # Expand embeddings to include the new special tokens.
    model.resize_token_embeddings(len(tokenizer))
    print('model/tokenizer preparing...done')

    # -------------------------------------------------------------------------
    # Prompt construction & tokenization
    # -------------------------------------------------------------------------
    print('map to tokenized dataset...')

    # Train set (stored as a list of dicts compatible with Trainer)
    tokenized_train_dataset = []
    train_input = input_fun(dataset['train'], relation_options, system_prompt, instruction, prompt)
    train_output = output_fun(dataset['train'], label_mapping)
    for input_str, output_str in zip(train_input, train_output):
        full_sequence = input_str + '\n' + output_str
        # Supervise LM to reproduce the entire full_sequence
        tokenized_example = tokenizer(
            full_sequence,
            text_target=full_sequence,
            max_length=1024,
            padding="max_length",
            truncation=True
        )
        tokenized_train_dataset.append(tokenized_example)

    # Quick sanity check: reconstruct one training example
    untokenized_example = tokenizer.decode(tokenized_train_dataset[0]["input_ids"])
    print('untokenizing first training example: ', untokenized_example)

    # Validation set
    tokenized_val_dataset = []
    validation_input = input_fun(dataset['validation'], relation_options, system_prompt, instruction, prompt)
    validation_output = output_fun(dataset['validation'], label_mapping)
    for input_str, output_str in zip(validation_input, validation_output):
        full_sequence = input_str + '\n' + output_str
        tokenized_example = tokenizer(
            full_sequence,
            text_target=full_sequence,
            max_length=1024,
            padding="max_length",
            truncation=True
        )
        tokenized_val_dataset.append(tokenized_example)

    # Keep test split *un-tokenized* for generation-time evaluation
    test_input = input_fun(dataset['test'], relation_options, system_prompt, instruction, prompt)
    test_output = output_fun(dataset['test'], label_mapping)
    print('map to tokenized dataset...done')

    # -------------------------------------------------------------------------
    # Helpers for Trainer / Evaluation
    # -------------------------------------------------------------------------
    def preprocess_logits_for_metrics(logits, labels):
        """
        Reduce logits to token-level argmax for Trainer.
        (Not actually used in our generation-based metric, but required by API.)
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def evaluate(pipe, input_data, labels):
        """
        Run generation on `input_data`, parse JSON to labels, and compute F1.
        Fallback: if JSON parsing fails, default to 'negative/no treatment'.
        """
        pred_labels = []
        for input_str in tqdm(input_data, desc='Generating answers'):
            # Extract the user section and construct chat messages
            split = input_str.find('### User:')
            user_prompt = input_str[split:]
            # NOTE: there's a duplicated "messages =" in original; harmless but fixed here.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=args.max_new_tokens,  # CLI-configurable (default 256)
            )
            # Hugging Face chat pipeline returns a turn list; take last assistant content
            output = outputs[0]["generated_text"][-1]['content']
            try:
                pred_labels.append(extract_label(json.loads(output)))
            except Exception:
                pred_labels.append("negative/no treatment")

        # Map 4-way gold labels → binary ids, predictions already 2-way
        gold_relations = [rel2id[label_convertion[r]] for r in labels]
        predictions = [rel2id[r] for r in pred_labels]

        return compute_f1(predictions, gold_relations)

    def compute_metrics(eval_preds):
        """
        Trainer hook: re-run generation over validation inputs to compute metrics.
        This is slow; consider reducing eval frequency or caching if needed.
        """
        global best_f1

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        val_labels = dataset['validation']['label']
        performance = evaluate(pipe, validation_input, val_labels)
        val_f1 = performance['f1']
        if val_f1 > best_f1:
            best_f1 = val_f1
            print('new best eval_f1: ', performance['f1'])

        return performance

    def load_trained_model(args):
        """
        Reload a checkpoint from output_dir and return a generation pipeline.

        WARNING:
        - Uses the first entry from os.listdir(output_dir). Prefer
          `trainer.state.best_model_checkpoint` for reliability.
        - Uses args.model_id here, but the training model_id is hard-coded above.
          Keep them consistent if you make model_id configurable.
        """
        ckpt_path = os.path.join(args.output_dir, os.listdir(args.output_dir)[0])
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left', token=access_token, **tokenizer_kwargs)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe

    # -------------------------------------------------------------------------
    # Trainer setup
    # -------------------------------------------------------------------------
    total_train_steps = (
        (len(tokenized_train_dataset) // args.per_device_train_batch_size)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    warmup_steps = int(0.1 * max(total_train_steps, 1))
    print("warmup_steps", warmup_steps)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=TrainingArguments(
            output_dir=args.output_dir,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,   # memory saver; may slow throughput
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True,                     # ensure GPU supports bf16
            optim=args.optim,              # e.g., "adamw_torch"
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",    # must match key from compute_metrics
            greater_is_better=True,
            save_total_limit=1,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=20),
        ],
    )

    # Train
    if args.do_train:
        trainer.train()

    # Final test evaluation via generation
    if args.do_eval_on_test:
        test_labels = dataset['test']['label']
        model_pipe = load_trained_model(args)
        evaluate(model_pipe, test_input, test_labels)


if __name__ == "__main__":
    # CLI for reproducible runs (e.g., SLURM)
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)  # NOTE: not used in code
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)  # used by evaluate()
    parser.add_argument("--isLora", action='store_true')            # NOTE: not used
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--access_token", type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval_on_test", action='store_true')

    args = parser.parse_args()
    main(args)
