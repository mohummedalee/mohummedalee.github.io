---
title: "A recipe for measuring bias in healthcare AI"
date: 2025-11-15
draft: true
hideToc: false
summary: "A step-by-step recipe for generating synthetic biased data and setting up an LLM judge for evaluating bias in healthcare AI applications."
tags: [language models, evals, synthetic data]
---

I came across an interesting paper recently while going over the program for the 2025 [KDD GenAI Evaluation](https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/) workshop: _A Framework To Audit Biases and Toxicity in LLM Clinical Note Generation_ by [Oxendine et. al. (2025)](https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/assets/papers/Submission%2034.pdf).
I've been thinking about evals a lot recently, especially after building AI systems in the industry for enterprise users.
Every once in a while, people will either with good intentions or adversarially ask you "How do you make sure this system isn't biased?" What method you use changes from one domain to another---a measurement for an AI system in finance would be very different from one in advertising. But I like this paper because it lays out the foundations of a recipe for healthcare system.

**Pre-requisites:** Before we go over the methodology of measuring bias, it's useful to understand that the paper does. The task of the AI system in the paper is to translate doctor-patient dialogs (e.g, Doctor: "How are you feeling today?" Patient: "Not so good...") into structured summary notes. This is a very realistic use-case because I've seen [One Medical](https://health.amazon.com/onemedical) clinics use something like this. The authors create biased data on two levels: _First_, they introduce stereotypes into the conversation; e.g, Doctor: "Given your genetic background, it’s possible that the standard treatment might not be as effective." _Second_, they insert biased counterfactual (read: not factual) demographic information in the dialog, e.g., Doctor: "What race do you identify as?", Patient: "Black"; and then they compare the difference in the final summary across different demographic groups. The goal is to see that given the stereotype added initially, which demographic groups will trigger a bias in the final summary. All of these modifications are being made to the original doctor-patient dialog, i.e., the AI's input.

**The recipe:** Importantly, while the paper focuses on clinical notes, I believe the approach here is more broadly applicable.
The steps go something like:

1. Have data showing intended usage of the system
2. Generate biased synthetic data for testing
3. Commit to a definition of bias
4. Prompt an LLM judge to catch bias (and validate against human judgments)
5. Track the LLM judge's output as a metric in your engineering process

I wanted to go over this recipe in detail, show how the paper does it, and suggest how teams working in healthcare could do it.

## Have data showing intended usage of the system

This is foundational and absolutely necessary. Until you understand what good use-cases looks like, you cannot generate realistic bad cases.
Unfortunately, and quite surprisingly, people in enterprise often don't have this---especially when execs ask to use AI automation for processes that haven't existed with data trails.

**How the paper does it:** They use [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) and [ACI-Bench](https://github.com/wyim/aci-bench/tree/main), both open source doctor-patient dialog datasets from previous work ([Ben Abacha et al., 2023](https://aclanthology.org/2023.eacl-main.168.pdf), [Yim et al., 2023](https://www.nature.com/articles/s41597-023-02487-3)). The rigor and open source culture of academia ensures there's always a good dataset within arm's reach. ACI-Bench contains real transcribed dialogs between doctors and patients; and even though MTS-Dialog is synthetic, it has gone through human annotation and refinement from medical experts.

**How you can do it:** In an enterprise setting, you need to earnestly spend some time either finding and polishing old data that shows pre-AI usage; or invest in a robust way of generating realistic synthetic data. This could look like input-output pairs if your system uses a single LLM, or a happy path through an LLM agent picking tools. But I'd say a few hundred datapoints are necessary, because this is what you will tweak in the next step to add bias. In the absence of any data, I've seen teams setup data generation pipelines by prompting an LLM to generate _seeds_ for the data (e.g., dialog templates); then doing a second prompt to flesh out the seeds into real data; and finally manually looking at the data, keeping the good points and refining your prompts by learning from the unrealistic datapoints.

## Generate biased synthetic data for testing

This is the step where you can use LLMs to generate realistic-looking but biased (along age, gender or race) datapoints. Having data from the previous steps ensures that you can use few-shot examples or templates to make the biased data look almost identical to the good data.

**How the paper does it:**

**How you can do it:**
