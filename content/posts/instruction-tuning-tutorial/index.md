---
title: "Getting the Hang of Instruction Tuning"
date: 2024-11-11
draft: true
hideToc: false
# summary: "A hands-on programming tutorial of instruction tuning"
---

I've been meaning to seriously dig into instruction tuning of language models for quite some time.
To me, it's the step in the model building pipeline that is the closest to alchemy.
It is also the step that arguably provides the largest delta in terms of the general user's utility.
You take a model whose job is simply predicting the next token, and convert it to a useful tool that can follow user instructions, and can be "prompted" into doing things. 

Once you start playing with LLMs via API access, you quickly notice that not all models are able to respond to prompts.
For example, practitioners often know, and literature ([Sannigrahi et. al., 2024](https://arxiv.org/abs/2406.06729)) has shown, that OpenAI's now deprecated `babbage-002` model wasn't helpful for many tasks, regardless of the cleverness of your prompt.
Given the near-universal impact prompting has had on machine learning, I decided to pop under the hood of how it works, and fine-tune a model on instruction data myself.

### Programming instruction tuning from scratch
Lucky for me, [Sebastian Raschka](https://sebastianraschka.com) has been putting out incredible teaching content to lay out all the gory details of building LLMs. I learnt all of the things in this post by following along [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07) of his new book, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch).

**Dataset.** To teach a model to follow instructions, you first need a dataset of high-quality, ideally human-written instructions.
These instructions---like all supervised training data---have a very clear input and output for each data point; for instruction-tuning (IT) purposes, these are called instructions and responses.
Unfortunately, most instruction datasets, especially for large commercial models, tend to be proprietary.
InstructGPT ([Ouyang et. al., 2022](https://arxiv.org/abs/2203.02155))---in many ways the ChatGPT technical report---never released the instruction data (among everything else). Figure 2 of the paper clearly illustrates how laborious the process of collecting instruction and response pairs can be, and requires attentive human labor (Step 1).

Many commercial models continue to keep their instruction datasets closed, so a lot of experimental work has to rely on synthetic data generated through distillation from models that have been instruction-tuned ([Zhang et. al., 2024](https://arxiv.org/abs/2308.10792)). There are some exceptions: [Dolly](https://github.com/databrickslabs/dolly), for instance, is a human-written and open-source instruction dataset put together by Databricks employees. In this post, I'll use the [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset to fine-tune my model, which is distilled from OpenAI's GPT-3.5 (`text-davinci-003`). Each point in Alpaca looks like:

```
{
    "instruction": "Identify the odd one out.",
    "input": "Twitter, Instagram, Telegram",
    "output": "Telegram"
}
```

You'll notice that in addition to the `instruction` and response (`output`), it also includes an additional `input` field. This isn't present for all points, but allows for more expressive data points which can receive input arguments from the users at test time as well.


```
f"Below is an instruction that describes a task. "
f"Write a response that appropriately completes the request."
f"\n\n### Instruction: \n{entry['instruction']}"
```

The `input` part, which can be empty, comes next:
```
input_text = f"\n\n###Input: \n{entry['input'] if entry['input'] else ''}"
```
Finally the `output` is what we want the model to generate, and is put under a `### Response`, so at inference time, the model follows that exact format.