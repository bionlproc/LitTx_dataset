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
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import f1_score

os.environ["WANDB_DISABLED"] = "true"

best_f1 = -1.0

def compute_f1(preds, labels):
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
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'macro_f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        print('performance: ', {'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
                'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold})
        return {'precision': prec, 'recall': recall, 'f1': f1, 'macro_f1': macro_f1,
                'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': n_gold}
    

system_prompt = '''You are a helpful medical expert, and your task is to extract the relation between the given entity pairs. Please choose the relation label from the provided options. Organize your output in a json formatted as Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.\n\n'''

relation_options = '''
"A": "treats",
"B": "conditional treatment",
"C": "negative treatment",
"D": "other",
'''

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

rel2id = {
        "treats": 1,
        "conditional treatment": 2,
        "negative treatment": 3,
        "other": 0,
    }

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
    texts = examples['context']
    subjs = examples['subject']
    objs = examples['object']
    prompt_text = []

    for text, subj, obj in zip(texts, subjs, objs):
        component = prompt.format(text=text, subj=subj, obj=obj, relation_options=relation_options)
        prompt_text.append(system_prompt + instruction + component)

    return prompt_text

def output_fun(examples, label_mapping):
    labels = [label_mapping[l] for l in examples['label']]
    return ['{' + f'"answer_choice": "{label}"' + '}' for label in labels]

def extract_label(dict):
    return answer_mapping[dict['answer_choice']]



def main(args):

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def evaluate(pipe, input_data, labels):
        pred_labels = []
        for input in tqdm(input_data, desc='Generating answers'):
            split = input.find('### User:')
            user_prompt = input[split:]
            messages = messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=256,
            )
            output = outputs[0]["generated_text"][-1]['content']
            try:
                pred_labels.append(extract_label(json.loads(output)))
            except:
                pred_labels.append("other")

        gold_relations = [rel2id[r] for r in labels]
        predictions = [rel2id[r] for r in pred_labels]

        return compute_f1(predictions, gold_relations)

    def compute_metrics(eval_preds):
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
        model = AutoModelForCausalLM.from_pretrained(os.path.join(args.output_dir, os.listdir(args.output_dir)[0]), device_map="auto", token=access_token)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left', token=access_token, **tokenizer_kwargs)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe
    
        
    access_token = args.access_token

    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    print('model/tokenizer preparing...')
    model_id = args.model_id # 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    tokenizer_kwargs = {
        "additional_special_tokens": ['<obj>', '<subj>', '<subj/>', '<obj/>'],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left', token=access_token, **tokenizer_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", token=access_token)
    model.resize_token_embeddings(len(tokenizer))
    print('model/tokenizer preparing...done')

    print('map to tokenized dataset...')

    # tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = []
    train_input = input_fun(dataset['train'], relation_options, system_prompt, instruction, prompt)
    train_output = output_fun(dataset['train'], label_mapping)
    for input, output in zip(train_input, train_output):
        full_sequence = input + '\n' + output
        tokenized_example = tokenizer(full_sequence, text_target=full_sequence, max_length=512, padding="max_length", truncation=True)
        tokenized_train_dataset.append(tokenized_example)
    untokenized_example = tokenizer.decode(tokenized_train_dataset[0]["input_ids"])
    print('untokenizing first training example: ', untokenized_example)
    tokenized_val_dataset = []
    validation_input = input_fun(dataset['validation'], relation_options, system_prompt, instruction, prompt)
    validation_output = output_fun(dataset['validation'], label_mapping)
    for input, output in zip(validation_input, validation_output):
        full_sequence = input + '\n' + output
        tokenized_example = tokenizer(full_sequence, text_target=full_sequence, max_length=512, padding="max_length", truncation=True)
        tokenized_val_dataset.append(tokenized_example)
    test_input = input_fun(dataset['test'], relation_options, system_prompt, instruction, prompt)
    test_output = output_fun(dataset['test'], label_mapping)
    print('map to tokenized dataset...done')


    total_train_steps = (
            (len(tokenized_train_dataset) // args.per_device_train_batch_size)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
    )
    warmup_steps = int(0.1 * total_train_steps)
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
            gradient_checkpointing=True,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            optim=args.optim,
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            do_eval=True,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=20),
            # SaveModelCallback(tokenizer),
        ],
    )
    if args.do_train:
        trainer.train()
        print("Model training completed.")

    
    if args.do_eval_on_test:
        test_labels = dataset['test']['label']
        model = load_trained_model(args)
        evaluate(model, test_input, test_labels)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval_on_test", action='store_true')
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--access_token", type=str, required=True)

    args = parser.parse_args()
    main(args)