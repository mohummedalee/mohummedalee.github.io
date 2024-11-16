---
title: "Getting the Hang of Instruction Tuning"
date: 2024-11-13
draft: true
hideToc: false
summary: "A hands-on programming tutorial of instruction tuning"
tags: [machine learning, tutorial]
---

**Note**: All code discussed in this post lives in [this repository](https://github.com/mohummedalee/instruction-tuning-gemma-2b/).

I've been meaning to seriously dig into instruction tuning (IT) of language models for quite some time.
To me, it's the step in the model building pipeline that is the closest to alchemy.
It is also the step that arguably provides the largest delta in terms of the general user's utility.
You take a model whose job is simply predicting the next token, and convert it to a useful tool that can follow user instructions, and can be "prompted" into doing things. 

Once you start playing with LLMs, you quickly realize that not all models respond equally well, or at all, to prompts.
For example, practitioners often know, and literature ([Sannigrahi et. al., 2024](https://arxiv.org/abs/2406.06729)) has shown, that OpenAI's now deprecated `babbage-002` model wasn't helpful for many tasks, regardless of the cleverness of your prompt. This happens mainly because the model had not gone through an instruction tuning phase.
Given the near-universal impact IT has had on LLMs, I decided to pop under the hood of how it works, and fine-tune a model on instruction data myself.

## Programming instruction tuning from scratch
Lucky for me, [Sebastian Raschka](https://sebastianraschka.com) has been putting out incredible teaching content to lay out all the gory details of building LLMs. I learnt all of the things in this post by following along [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) of his new book, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

### Dataset
To teach a model to follow instructions, you first need a dataset of high-quality, ideally human-written instructions.
Each of these instructions---like all supervised training data---has a clear input $x$ and output $y$; for IT purposes, these are called instructions and responses.
Unfortunately, most instruction datasets, especially for large commercial models, tend to be proprietary.
InstructGPT ([Ouyang et. al., 2022](https://arxiv.org/abs/2203.02155))---in many ways the ChatGPT technical report---never released the instruction data (among everything else). Figure 2 of the paper clearly illustrates how laborious the process of collecting instruction and response pairs can be, and requires attentive human labor (Step 1).

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

A quick function to get this formatting for each point, taken from Raschka's book:

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

Before tokenizing the text, the `### Response` part is also appended to the string outside of this function, after which, we want the model to finish the sentence. For a full implementation, see the `InstructionDataset` class in [data.py](https://github.com/mohummedalee/instruction-tuning-gemma-2b/blob/0745c64689b1334485b0b525264366361c9f5d7d/scripts/data.py#L5C7-L5C25).

### Model

I chose to go with a [Gemma 2B](https://huggingface.co/google/gemma-2b) model to fine-tune. You can choose any transformer model for this purpose really; the only constraint would be how many GPUs and time you have. You're essentially training the model on next token prediction with cross-entropy loss, but on a very specific kind of instruction data, and not the entire Internet. I chose Gemma because it already has both a base ([`gemma-2b`](https://huggingface.co/google/gemma-2b)) and an instruction-tuned ([`gemma-2b-it`](https://huggingface.co/google/gemma-2b-it)) variant---so there's already a good IT model to evaluate mine against.

### Implementing from Scratch

Before I talk about how HuggingFace's built-in methods can be used to implement IT, I wanted to discuss the from-scratch version presented in the book.
It's a fantastic exercise to gain confidence, and see how incredibly simple the core logic is.


To my surprise, most of the work happens in the how you the training batches are collated together. If you write PyTorch, you probably know that collate functions (`collate_fn`) in PyTorch are used to write the logic for how a DataLoader should stack together tensors for each batch ([PyTorch docs](https://pytorch.org/docs/stable/data.html)). Given a batch of sentences in the IT dataset, you carefully assemble the input and output tensors in `collate_fn` so the output tensor is one token ahead. So, the tall order of getting a model to follow instructions is, quite remarkably, just next token prediction on sentences that are written in a templated format of an *Instruction*, an *Input* and a *Response*. Given the massive [memory](https://memit.baulab.info/) of modern models, and the ability of Transformers to find the right context, this recipe just works.

Raschka's [code](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb) does a wonderful job of slowly building up complexity by writing multiple drafts of the collate function before introducing the final version. I wholeheartedly recommend the exercise; I re-share the final collate function here with my comments:

```
def custom_collate_fn(
    batch,
    # GPT-2's endoftext token ID
    pad_token_id=50256,
    # PyTorch's cross_entropy function will ignore these in computation
    ignore_index=-100,    
    allowed_max_length=None,
    device="cpu"
):
    """Figure out the longest sentence in the batch,
    that will be the width of the batch"""
    batch_max_length = max(len(item)+1 for item in batch)
    """Assemble input and output in separate lists"""
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        """Pad all space needed to
        match batch_max_length with endoftext tokens"""
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        """
        Voila! Input is just train sentence without the last token;
        output is the same without the first token
        At each index in the input, the model should
        predict the token at index+1 in the target
        """
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        """Do not compute loss on padding tokens in the target sequence;
        do not backprop this additional loss"""
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

### Implementing with off-the-shelf Tools
While I enjoyed the exercise of writing the whole thing from scratch, when I started to actually train a large model,
I resorted to using HuggingFace's [`DataCollatorForLanguageModeling`](https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling), just so I could easily plug it into a `Trainer`.