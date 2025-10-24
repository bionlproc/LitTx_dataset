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

import torch
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
from huggingface_hub import login  # used only if you call login elsewhere
from sklearn.metrics import f1_score

# Avoid accidental wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Track best validation F1 over training for logging
best_f1 = -1.0


def compute_f1(preds, labels):
    """
    Compute two F1 variants:
      - macro_f1: standard sklearn macro F1 over all classes (including 'other')
      - custom P/R/F1: only counts *non-zero* labels as "positive" (class-agnostic),
        effectively evaluating "any-positive vs other" while requiring exact class
        match for a 'correct' prediction.

    Expected:
      preds, labels: integer class ids where 0 == 'other'
    """
    n_gold = n_pred = n_correct = 0
    macro_f1 = f1_score(labels, preds, average='macro')

    # Custom P/R that ignores 'other' (id 0) when counting predicted/gold positives
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1

    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'macro_f1': 0.0}
    else:
        prec = n_correct * 1.0 / max(n_pred, 1)
        recall = n_correct * 1.0 / max(n_gold, 1)
        f1 = 2.0 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        print('performance: ', {
            'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
            'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold
        })
        return {
            'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
            'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold
        }


# System prompt sets the role and required output format
system_prompt = (
    'You are a helpful medical expert, and your task is to extract the relation '
    'between the given entity pairs. Please choose the relation label from the '
    'provided options. Organize your output in a json formatted as '
    'Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research '
    'purposes only, so please have a definite answer.\n\n'
)

# Options shown to the model; keep in sync with mappings below
relation_options = '''
"A": "treats",
"B": "conditional treatment",
"C": "negative treatment",
"D": "other",
'''

# Mapping between human-readable labels and multiple-choice letters
label_mapping = {
    "treats": "A",
    "conditional treatment": "B",
    "negative treatment": "C",
    "other": "D",
}

answer_mapping = {
    "A": "treats",
    "B": "conditional treatment",
    "C": "negative treatment",
    "D": "other",
}

# Numeric ids for computing metrics (0 reserved for 'other')
rel2id = {
    "treats": 1,
    "conditional treatment": 2,
    "negative treatment": 3,
    "other": 0,
}

# A neutral demo instruction block (few-shot style header). Kept in the prompt
# to stabilize formatting of the expected JSON response.
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

# Template that combines the text + entity mentions + choice list
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
    Build full instruction prompts for each example by concatenating:
    [system_prompt] + [instruction demo] + [example-specific prompt]
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
    Convert gold string labels into the expected JSON (string) the model should output.
    We supervise the LM to reproduce this JSON exactly during training.
    """
    labels = [label_mapping[l] for l in examples['label']]
    return ['{' + f'"answer_choice": "{label}"' + '}' for label in labels]


def extract_label(dct):
    """
    Map generated JSON back to human-readable label.
    Raises KeyError if answer_choice is missing; caller handles fallbacks.
    """
    return answer_mapping[dct['answer_choice']]


def main(args):

    def preprocess_logits_for_metrics(logits, labels):
        """
        For Trainer.compute_metrics: reduce logits to predicted token ids.
        Note: For CausalLM, this is not a pure classification head; we still return
        argmax tokens so Trainer can pass something consistent to compute_metrics,
        but compute_metrics below ignores these and performs *generation*-based eval.
        """
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def evaluate(pipe, input_data, labels):
        """
        Generate answers with the instruction-tuned model and compute F1.

        - `pipe` is a text-generation pipeline (chat-style messages).
        - We parse the final assistant turn as JSON; if parsing fails, default to 'other'.
        """
        pred_labels = []
        for input in tqdm(input_data, desc='Generating answers'):
            # Split off the "### User:" section and feed it + system prompt as messages
            split = input.find('### User:')
            user_prompt = input[split:]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=256,
            )
            # Hugging Face chat pipeline returns a list with a conversation trace;
            # we take the last assistant message content.
            output = outputs[0]["generated_text"][-1]['content']
            try:
                pred_labels.append(extract_label(json.loads(output)))
            except Exception:
                # Be robust to non-JSON generations or missing keys
                pred_labels.append("other")

        gold_relations = [rel2id[r] for r in labels]
        predictions = [rel2id[r] for r in pred_labels]

        return compute_f1(predictions, gold_relations)

    def compute_metrics(eval_preds):
        """
        Hook used by Trainer during evaluation.
        IMPORTANT: This spins up a generation pipeline and *re-generates* the entire
        validation set each time (epoch-level eval). This is slow but faithful to the
        task. Consider caching or reducing eval frequency if throughput is an issue.
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
        Reload the most recent checkpoint from output_dir and return a generation pipeline.
        WARNING: `os.listdir(args.output_dir)[0]` assumes exactly one subdir/checkpoint;
        you may want to sort by mtime or use Trainer's `trainer.state.best_model_checkpoint`.
        """
        ckpt_path = os.path.join(args.output_dir, os.listdir(args.output_dir)[0])
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left', token=access_token, **tokenizer_kwargs)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe

    # -------------------------------------------------------------------------
    # Data & model setup
    # -------------------------------------------------------------------------
    access_token = args.access_token

    dataset_name = args.dataset_name
    # trust_remote_code in case the dataset repo defines a custom loader
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    print('model/tokenizer preparing...')
    model_id = args.model_id  # e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    tokenizer_kwargs = {
        "additional_special_tokens": ['<obj>', '<subj>', '<subj/>', '<obj/>'],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', token=access_token, **tokenizer_kwargs)
    # Common pattern for decoder-only LMs when no pad_token exists
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=access_token)
    # Resize embeddings to accommodate newly added special tokens
    model.resize_token_embeddings(len(tokenizer))
    print('model/tokenizer preparing...done')

    # -------------------------------------------------------------------------
    # Prompt construction and tokenization
    # Trainer can take a list of dicts; alternatively wrap with datasets.Dataset.from_list
    # -------------------------------------------------------------------------
    print('map to tokenized dataset...')

    # Train
    tokenized_train_dataset = []
    train_input = input_fun(dataset['train'], relation_options, system_prompt, instruction, prompt)
    train_output = output_fun(dataset['train'], label_mapping)
    for input_text, output_text in zip(train_input, train_output):
        full_sequence = input_text + '\n' + output_text
        tokenized_example = tokenizer(
            full_sequence,
            text_target=full_sequence,          # supervise LM to reproduce the same string
            max_length=512,
            padding="max_length",
            truncation=True
        )
        tokenized_train_dataset.append(tokenized_example)

    # Debug: show one reconstructed training example
    untokenized_example = tokenizer.decode(tokenized_train_dataset[0]["input_ids"])
    print('untokenizing first training example: ', untokenized_example)

    # Validation
    tokenized_val_dataset = []
    validation_input = input_fun(dataset['validation'], relation_options, system_prompt, instruction, prompt)
    validation_output = output_fun(dataset['validation'], label_mapping)
    for input_text, output_text in zip(validation_input, validation_output):
        full_sequence = input_text + '\n' + output_text
        tokenized_example = tokenizer(
            full_sequence,
            text_target=full_sequence,
            max_length=512,
            padding="max_length",
            truncation=True
        )
        tokenized_val_dataset.append(tokenized_example)

    # Test (kept untokenized for generation-time evaluation)
    test_input = input_fun(dataset['test'], relation_options, system_prompt, instruction, prompt)
    test_output = output_fun(dataset['test'], label_mapping)
    print('map to tokenized dataset...done')

    # -------------------------------------------------------------------------
    # Training setup
    # -------------------------------------------------------------------------
    total_train_steps = (
        (len(tokenized_train_dataset) // args.per_device_train_batch_size)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    warmup_steps = int(0.1 * max(total_train_steps, 1))  # guard against tiny datasets
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
            gradient_checkpointing=True,   # saves memory; may reduce throughput
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True,                     # set to True only if your GPUs support bf16
            optim=args.optim,              # e.g., "adamw_torch" or "adamw_8bit"
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch",   # triggers compute_metrics each epoch
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",    # must match key returned by compute_metrics
            greater_is_better=True,
            save_total_limit=1,            # keep disk usage low
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=20),  # early stop on flat metrics
            # You could add a custom callback to save tokenizer alongside best model.
        ],
    )

    # Train
    if args.do_train:
        trainer.train()
        print("Model training completed.")

    # Final test-time evaluation via generation
    if args.do_eval_on_test:
        test_labels = dataset['test']['label']
        model_pipe = load_trained_model(args)
        evaluate(model_pipe, test_input, test_labels)


if __name__ == "__main__":
    # CLI arguments to keep experiments reproducible from the shell / SLURM scripts
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--optim", type=str, required=True)  # e.g., adamw_torch
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval_on_test", action='store_true')
    parser.add_argument("--model_id", type=str, required=True)      # HF model id
    parser.add_argument("--dataset_name", type=str, required=True)  # HF dataset id
    parser.add_argument("--access_token", type=str, required=True)  # HF token for gated models

    args = parser.parse_args()
    main(args)
