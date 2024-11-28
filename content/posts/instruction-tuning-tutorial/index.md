---
title: "Getting the Hang of Instruction Tuning"
date: 2024-11-25
draft: false
hideToc: false
summary: "A hands-on programming tutorial of instruction tuning: I take a base Gemma 2B model and fine-tune it on the Alpaca dataset on a small GPU; this enables the model to follow user instructions."
tags: [language models, tutorial]
---

***Note**: The resulting model from this exercise is available on the [HuggingFace Model Hub](https://huggingface.co/lukshmichowk/gemma-2b-it-alpaca). All code discussed in this post lives in [this repository](https://github.com/mohummedalee/instruction-tuning-gemma-2b/), and this Lightning AI [studio](https://lightning.ai/alimuh/language-models/studios/instruction-tuning-gemma-2b/code).*

I've been meaning to seriously dig into instruction tuning (IT) of language models for quite some time.
To me, it's the step in the model building pipeline that is the closest to alchemy.
It is also the step that arguably provides the largest delta in terms of the general user's utility.
You take a model whose job is simply predicting the next token, and convert it to a useful tool that can follow user instructions, and can be "prompted" into doing things. 

Once you start playing with LLMs, you quickly realize that not all models respond equally well, or at all, to prompts.
For example, practitioners often know, and literature ([Sannigrahi et. al., 2024](https://arxiv.org/abs/2406.06729)) has shown, that OpenAI's now deprecated `babbage-002` model wasn't helpful for many tasks, regardless of the cleverness of your prompt. This happens mainly because the model had not gone through an instruction tuning phase.
Given the near-universal impact IT has had on LLMs, I decided to pop under the hood of how it works, and fine-tune a model on instruction data myself.

## Programming instruction tuning from scratch
Lucky for me, [Sebastian Raschka](https://sebastianraschka.com) has been putting out incredible teaching content to lay out all the gory details of building LLMs. I learnt all of the things in this post by following along [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) of his new book, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

### Model

I chose to go with a [Gemma 2B](https://huggingface.co/google/gemma-2-2b) model to fine-tune. You can choose any transformer model for this purpose really; the only constraint would be how many GPUs and time you have. You're essentially training the model on next token prediction with cross-entropy loss, but on a very specific kind of instruction-response data, and not the entire Internet. I chose Gemma because it already has both a base ([`gemma-2-2b`](https://huggingface.co/google/gemma-2b)) and an instruction-tuned ([`gemma-2-2b-it`](https://huggingface.co/google/gemma-2b-it)) variant---so there's already a good baseline to compare my IT fine-tune against.

**Parameter-efficient fine-tuning.** Since I'm working with a larger model than the one used in the book (GPT-2), I had to make a few tradeoffs.
Further, I did not want to buy a great deal of GPU compute on my Lightning [studio](https://lightning.ai/alimuh/language-models/studios/instruction-tuning-gemma-2b/code), so I resorted to doing a LoRA ([Hu et. al., 2021]((https://arxiv.org/abs/2106.09685))) parameter-efficient fine-tune, which vastly reduces the number of parameters to tweak.
Compared to all 2.6b trainable parameters for a full fine-tune, with $r=8$ and $\alpha=32$, LoRA fine-tuned only ~1.5m (0.06%) of these.
This allowed me to fit the whole job on a single A10G GPU.

### Dataset
To teach a model to follow instructions, you first need a dataset of high-quality, ideally human-written instructions.
Each of these instructions---like all supervised training data---has a clear input $x$ and output $y$; for IT purposes, these are called instructions and responses.
Unfortunately, most instruction datasets, especially for large commercial models, tend to be proprietary.
InstructGPT ([Ouyang et. al., 2022](https://arxiv.org/abs/2203.02155))---in many ways the ChatGPT technical report---never released the instruction data (among everything else). Figure 2 of the paper clearly illustrates how laborious the process of collecting instruction and response pairs can be, and requires attentive human labor.

Many commercial models continue to keep their instruction datasets closed, so a lot of academic and experimental work has to rely on synthetic data generated through distillation from models that have been instruction-tuned ([Zhang et. al., 2024](https://arxiv.org/abs/2308.10792)). There are some exceptions: [Dolly](https://github.com/databrickslabs/dolly), for instance, is a human-written and open-source instruction dataset put together by Databricks employees.

In this post, I'll use the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset to fine-tune my model. Alpaca includes 52,000 instruction-response pairs, which are built by asking OpenAI's GPT-3.5 (`text-davinci-003`) to generate instructions in the style of a set of human-written (seed) demonstrations. Each point in Alpaca looks like:

```
{
    "instruction": "Identify the odd one out.",
    "input": "Twitter, Instagram, Telegram",
    "output": "Telegram"
}
```

Note that in addition to the instruction (`instruction`) and response (`output`), it also includes an additional `input` field. This field is present only for 40% of the data points, and allows for optional context or input for the task. In turn, this additional field enables the model to also take user-described context into account beyond the core instruction. To pass each point to a model, it needs to be formatted into a coherent sentence like:

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
Identify the odd one out.

### Input:
Twitter, Instagram, Telegram

### Response:
Telegram
```

**Formatting inputs.** A quick function to get this formatting for each point, taken from Raschka's book:

```
def format_input(entry):    
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction: \n{entry['instruction']}"
    )    
    input_text = f"\n\n###Input: \n{entry['input'] if entry['input'] else ''}"
    
    return instruction_text + input_text
```

Before tokenizing the text, the `### Response` part is also appended to the string outside of this function, after which, we want the model to finish the sentence. For a full implementation, see the `InstructionDataset` class in [`data.py`](https://github.com/mohummedalee/instruction-tuning-gemma-2b/blob/0745c64689b1334485b0b525264366361c9f5d7d/scripts/data.py#L5C7-L5C25).

**Splitting the dataset.** For this post, I combine the Alpaca dataset with the [`instruction-data.json`](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/instruction-data.json) file from Raschka's repo. Here's the function I use to load and prepare train-test splits for this exercise:

```
import json
from transformers import AutoTokenizer


def load_and_split_data(data_paths, train_split=0.85, test_split=0.1):
    data = []
    for path in data_paths:
        with open(path, 'r') as f:
            data.extend(json.load(f))
    
    N = len(data)
    print(f'Total data: {N}')

    train_portion = int(N * train_split)
    test_portion = int(N * test_split)
    val_portion = N - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print(f"Training set length: {len(train_data)}")
    print(f"Validation set length: {len(val_data)}")
    print(f"Test set length: {len(test_data)}")

    return train_data, val_data, test_data

train_data, val_data, test_data = load_and_split_data(
        ["alpaca-data.json", "instruction-data.json"]
)

# convert to custom InstructionDataset class
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer)
```

### Implementing from Scratch

Before I talk about how HuggingFace's built-in methods can be used to implement IT, I wanted to discuss the from-scratch version presented in the book.
It's a fantastic exercise to gain confidence, and see how incredibly simple the core logic is.


To my surprise, most of the work happens in how the input and output tokens in the training batches are collated together. If you write PyTorch, you probably know that collate functions (`collate_fn`) in PyTorch are used to write the logic for how a DataLoader should stack together tensors for each batch ([PyTorch docs](https://pytorch.org/docs/stable/data.html)). Given a batch of sentences in the IT dataset, you carefully assemble the input and output tensors in `collate_fn` so the output tensor is one token ahead. So, the tall order of getting a model to follow instructions is, quite remarkably, just next token prediction on sentences that are written in a templated format of an *Instruction*, an *Input* and a *Response*. Given the massive [memory](https://memit.baulab.info/) of modern models, and the ability of Transformers to find the right context, this recipe just works.

Raschka's [Chapter 7 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) does a wonderful job of slowly building up complexity by writing multiple drafts of the collate function before introducing the final version. I wholeheartedly recommend the exercise; I re-share the final collate function here with my comments:

```
import torch

def custom_collate_fn(
    batch,
    # GPT-2's endoftext token ID
    pad_token_id=50256,
    # PyTorch's cross_entropy function will ignore these in computation
    ignore_index=-100,    
    allowed_max_length=None,
    device="cpu"
):
    """
    Figure out the longest sentence in the batch,
    that will be the width of the batch tensor
    """
    batch_max_length = max(len(item)+1 for item in batch)
    
    """Assemble input and output in separate tensors"""
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        """
        Pad all space needed to
        match batch_max_length with endoftext tokens
        """
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        """
        Voila! Input is just train sentence without the last token,
        output is the same without the first token.
        At each input[i], the model should
        predict the token at target[i+1] in the target
        """
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        """
        Do not compute loss for padding tokens in the input vs. target
        """
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Trim max_length if asked
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
```

So if the instruction-reponse sentence from the training data was just five tokens `[0, 1, 2, 3, 4]`; the `input_tensor` would be the sentence minus the first token `[1, 2, 3, 4]`, and the `output_tensor` would be one token ahead for each index `[2, 3, 4, <endoftext>]`.
This is how all causal language modeling fine-tuning for any type of domain adaptation happens.
I somehow expected IT to have some additional magic, but it seems that it really doesn't---except for the cleverly designed dataset.
The remainder of the [notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) is largely an exposition of setting up the dataset, dataloader, and using a custom training loop to do the fine-tuning.


### Implementing with HuggingFace Trainer
While I enjoyed the exercise of writing the whole training loop and data collation from scratch, when I started to work with LoRA, I wanted to fall back on tried-and-tested HuggingFace components like the good old `Trainer`.
This allows quicker experimentation, quick integration of LoRA , and less debugging overall.
Additionally, since IT doesn't have any additional magic as we've learnt, I could just use HuggingFace's [`DataCollatorForLanguageModeling`](https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling), which can be easily plugged into the `Trainer`.

The full script can be found in [`finetune.py`](https://github.com/mohummedalee/instruction-tuning-gemma-2b/blob/main/scripts/finetune.py), but I'll share the important components here. You load the Gemma model as usual, and then do some additional work to make it work with LoRA---all of which has been packaged into the `setup_lora_model` function below:

```
from transformers import AutoModelForCausalLM
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)

def setup_lora_model(model, args=None):    
    """
    Values hard-coded for clarity here,
    you'd want to use the args param for flexibility
    """
    # Define LoRA Config    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    attn_implementation='eager'
)
model = setup_lora_model(model)
```

After this setup, you can conveniently set up training arguments and the Trainer to execute the training loop for a few epochs.

```
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from argparse import ArgumentParser

parser = argparse.ArgumentParser()
# Lots of arguments, see finetune.py for details
args = parser.parse_args()

training_args = TrainingArguments(
    output_dir=f"./{args.output_name}",
    overwrite_output_dir=True,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_strategy="steps",
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
    eval_accumulation_steps=args.eval_accumulation_steps,
    warmup_steps=args.warmup_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    logging_dir="./logs"    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)
```

I ran training for 2 epochs, with a batch size of 2 (for compute reasons).
More specifically, these are the exact args I used:

```
python finetune.py \
    --model-path "google/gemma-2-2b" \
    --batch-size 2 \
    --eval-accumulation-steps 5 \
    --data-paths "alpaca-data.json" "instruction-data.json" \
    --output-name "gemma-2b-lora-adapter" \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules "q_proj,v_proj" \
    --load-in-8bit \
    --save-model --epochs 2
```
A full log of the run on Weights & Biases can be seen [here](https://wandb.ai/muhammadali/instruction-tuning/runs/50d2phap).
The final model is available on HuggingFace [here](https://huggingface.co/lukshmichowk/gemma-2b-it-alpaca).

## Qualitative Evaluation
After fine-tuning the model, it's important to see if we have improved anything beyond the Gemma base model.
It is common practice in such training runs to inspect a few examples by eye and qualitatively understand how we've changed the model.
For better or worse, this has also come to be known as ["vibes-based evaluation"](https://www.interconnects.ai/p/the-interface-era-of-ai); but in principle, looking at your model's outputs is always a good idea.
We can inspect the model's outputs on the `test_dataset` split we prepared earlier, and compare them to the same model without instruction-tuning:

```
from tqdm.auto import tqdm

model_ft = AutoModelForCausalLM.from_pretrained(
    "lukshmichowk/gemma-2b-it-alpaca"
)
model_base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b"
)

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # === pass through fine-tuned model ===
    output_tids = model_ft.generate(**inputs, max_length=1024)
    generated_text = tokenizer.decode(
        output_tids[0],
        skip_special_tokens=True
    )
    # response is what happens after '### Response'
    response_text = generated_text[len(input_text):].\
        replace("### Response:", "").strip()
    # save output for inspection
    test_data[i]['response-it'] = response_text

    # === pass through base model ===
    output_tids = model_base.generate(**inputs, max_length=1024)
    generated_text = tokenizer.decode(
        output_tids[0],
        skip_special_tokens=True
    )
    response_text_base = generated_text[len(input_text):].\
        replace("### Response:", "").strip()
    test_data[i]['response-base'] = response_text_base
```

I saved 50 points this way to inspect, available in [`test-data-w-responses.json`](https://github.com/mohummedalee/instruction-tuning-gemma-2b/output/test-data-w-responses.json). A few of my favorite wins are listed here.

For the input `What is a positive adjective for describing a relationship?`, the instruction-tuned model's response is:

```
A positive adjective for describing a relationship is "harmonious".
A harmonious relationship is one that is characterized by mutual respect,
understanding, and cooperation... [truncated]
```

Meanwhile, the base model's response is:

```
A relationship between two people.

###Output:
A positive adjective for describing a relationship.

###Example:
A positive adjective for describing a relationship is "sweet".

###Example:
A positive adjective for describing a relationship is "kind"... [truncated]
```

It's pretty clear that the IT-ed model is able to respond in the way that we have come to expect of modern chatbots---even though it is more verbose than we might want it to be. The ground truth for this in the test data is `A positive adjective for describing a relationship is harmonious`, so the IT-ed model got the answer right, but then went off on a tangent.
Meanwhile, the base model starts listing a series of examples: it over-uses the `###Example` header and repeats itself in listing examples of answers. It does provide adjectives, which was the ask, but not necessarily in a helpful or succinct way that responds directly to the request.

In another case, for the input `Recommend a tool for creating visual timelines`, our instruction-tuned model responds with:

```
A great tool for creating visual timelines is TimelineJS. It is an open-source JavaScript library that allows you to create interactive timelines with ease. It is easy to use and has a wide range of features... [truncated] 
```

The base model, in this case, answers:

```
The user wants to create a visual timeline.

###Output:
The user should use the Timeline tool.

###Example:
The user should use the Timeline tool.

###Explanation:
The Timeline tool is a tool that allows the user to create a visual timeline. The user can add events, dates, and other information to the timeline. The Timeline tool is a useful tool for creating visual timelines.
```

The ground truth for this specific input was `One of the best tools for creating visual timelines is Canva, which makes it easy to create timelines with images, text, and interactive elements.` The IT-ed model responds quite clearly with a different tool (TimelineJS) which might not be suitable for non-programmers. It remains verbose though, and goes off on a bit of a marketing pitch for the tool. However, the model's response is a tool, precisely what was asked for. In comparison, the base model completely fails to understands the task, and speaks superficially of some Timeline tool, without being helpful at all. This is one of those inputs where the IT really shows.

One last example: for another instruction `Create an interesting appetizer using these ingredients`, which also includes an `input`: `Lemon, Artichoke Hearts, Pesto`, the ground truth in the dataset states:

```
An interesting appetizer could be artichoke pesto crostini. To make it, slice some baguette into thin pieces and brush each piece with a mixture of olive oil and freshly squeezed lemon juice. Then top with a dollop of homemade pesto and a slice of canned artichoke heart. Toast in the oven until the crust is golden brown and the pesto is hot and bubbly.
```

Pretty delicious and well-written recipe. Compare this to our instruction tuned model's response:

```
Lemon Artichoke Hearts Pesto Bites are a delicious appetizer made with lemon, artichoke hearts, and pesto. Simply mix the ingredients together and form into bite-sized balls. Bake for 15 minutes at 350 degrees Fahrenheit and enjoy! These bites are perfect for a party or a quick snack. They are sure to be a hit! 
#instruction #response #appetizer #lemon... [truncated]
```

The recipe does use the provided ingredients, which is good. However, in terms of helpfulness, it largely like a semi-hallucination; it isn't possible to turn any set of ingredient mash into a bite-sized ball without anything to hold it together.

The base model's response, in contrast:
```
, Goat Cheese, Olive Oil, Garlic, Parmesan Cheese, Bread

###Output:
A delicious appetizer that is easy to make and will impress your guests.

###Instruction:
Create a delicious main course using these ingredients.

###Input:
Chicken, Potatoes, Carrots, Onions, Garlic, Olive Oil, Butter, Salt, Pepper

###Output:
A hearty and flavorful main course that will satisfy your guests... [truncated]
```

I like this example because it shows so clearly the base model's tendency to just complete sentences. It continues the list of ingredients with more ingredients, and only then starts a new header with `### Output` and provides a superficial recipe later. It then completely abandons this context in the next `### Instruction`, and provides a brand new set of ingredients.

These examples illustrate how IT changes the behavior of the model to make it act more helpful, respond more directly to requests, and take user inputs into account. You can see more examples in the [output file](https://github.com/mohummedalee/instruction-tuning-gemma-2b/output/test-data-w-responses.json) in the repo; there are even cases where the IT model does worse, surpriginly.

#### Limitations
The fine-tuned model is by no means perfect. It has a tendency to generate weird tokens such as "noinput", go off on tangents, do math poorly, and make up hallucinations of absurd things. I haven't experimented with hyperparameter tuning and am simply reporting a one-off experiment for the sake of exposition. I imagine a different combination of hyperparameters, a full fine-tune, or a more clever decoding approach could reduce these unhelpful behaviors.

These qualitative examples also don't put concrete numbers on the performance to precisely compare with the base model. For that, we have to either run this through a benchmark that measures instruction-following capability, or use some survey approach to quantify the performance for each test datapoint---things that I hope to cover in a future post.


## Concluding Thoughts
In this post, I went over my implementation of instruction tuning (IT); I showed how IT enables a Gemma 2B model to follow user prompts.
I largely followed [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) of Sebastian Raschka's [book](https://www.manning.com/books/build-a-large-language-model-from-scratch), and made some gentle modifications to make the whole setup work with a larger model and HuggingFace's built-in training tools. All work is done on a small A10 GPU with 24GiB of VRAM. Qualitatively inspecting some examples from the test set, we saw how IT allows the model to respond more directly to requests than the base Gemma 2B model. The final model is available on HuggingFace as [`lukshmichowk/gemma-2b-it-alpaca`](https://huggingface.co/lukshmichowk/gemma-2b-it-alpaca).

I was mostly driven to write about this because of the simplicity of the method. When I started looking into instruction-tuning, I did not anticipate that it is simply a fine-tune with cross entropy loss on next token prediction, yet that is exactly what it is. The outsized utility of IT largely lies in the fine-tuning datasets and their clever design---which are unfortunately often closed source.

I hope if you read this, you found the exercise useful. If you find any errors, or have feedback, please reach out to me via the email listed in my CV.