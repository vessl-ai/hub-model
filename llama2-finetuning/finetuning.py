import torch
import vessl
from datasets import load_from_disk
from omegaconf import OmegaConf
from peft import get_peft_config, get_peft_model, prepare_model_for_int8_training
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

train_config = OmegaConf.load("./train_config.yaml")

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name,
    load_in_8bit=True if train_config.quantization else None,
    device_map="auto" if train_config.quantization else None,
)

tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


if train_config.quantization:
    model = prepare_model_for_int8_training(model)

if train_config.peft.enable:
    peft_config = get_peft_config(OmegaConf.to_container(train_config.peft.config))
    model = get_peft_model(model, peft_config)


# dataset
def tokenize(sample, add_eos_token=True):
    prompt = sample["prompt"]

    result = tokenizer(
        prompt,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 256
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


ds = load_from_disk(train_config.dataset_path)
processed_ds = ds.map(tokenize)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)


training_arguments = TrainingArguments(
    per_device_train_batch_size=train_config.batch_size,
    gradient_accumulation_steps=train_config.grad_acc_steps,
    warmup_steps=train_config.warmup_steps,
    num_train_epochs=train_config.num_train_epochs,
    # max_steps=train_config.max_steps,
    learning_rate=train_config.learning_rate,
    fp16=True,
    logging_steps=train_config.logging_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=train_config.eval_steps,
    save_steps=train_config.save_steps,
    output_dir=train_config.intermediate_dir,
    save_total_limit=5,
    load_best_model_at_end=True,
)


class VesslLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "eval_loss" in logs.keys():
            payload = {
                "eval_loss": logs["eval_loss"],
            }
            vessl.log(step=state.global_step, payload=payload)
        elif "loss" in logs.keys():
            payload = {
                "train_loss": logs["loss"],
                "learning_rate": logs["learning_rate"],
            }
            vessl.log(step=state.global_step, payload=payload)


trainer = Trainer(
    model=model,
    train_dataset=processed_ds["train"],
    eval_dataset=processed_ds["validation"],
    args=training_arguments,
    data_collator=data_collator,
    callbacks=[VesslLogCallback],
)
model.config.use_cache = False

# old_state_dict = model.state_dict
# model.state_dict = (
#     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
# ).__get__(model, type(model))

model = torch.compile(model)
trainer.train()

model.save_pretrained(train_config.output_dir)
