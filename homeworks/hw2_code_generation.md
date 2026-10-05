# Homework 2: Code Generation

**Deadline for submitting the solution:**   
**30.10.2026, 23:59**

**Deadline for submitting the report:**   
**31.10.2026, 23:59**

## About the Task

This task is organized as a competition, where you'll compete against each other for the best quality. It asks you to train your Transformer from scratch to generate prompted code on [our dataset](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m). Working with code is no different from working with text (at least in this task), so all methods applicable to text also apply to code.

In this task, you are allowed to use agents for both research and coding. However, in your report, you will be required to describe how exactly you used agents and how this influenced your solution.

## Restrictions

The most important limitation is the speed of answer generation. All predictions for the test set must be generated in no more than 180 seconds on a single NVIDIA T4 GPU (available for free on Google Colab and Kaggle). This means you can optimize both the model architecture and its inference code.

To narrow your choice of architecture, only Transformer-based models are allowed. This means your model must have an encoder for text encoding and a decoder for code generation.

Using any methods to collect and create additional training data or additional targets (for example, distilling other models) is strictly prohibited.

If you have any doubts about the permissibility of a particular method, please contact @amshabalin.

## Submission format

To measure the quality of your model, you will need to send a file with your answers to the Telegram bot. A link to it will be posted later.

## Grading
You will have access to the current ranking on the public portion of the test dataset until the deadline. After the deadline, the private leaderboard results will be available. You can earn a total of 14 points for the task. These include:
1. 4 points for achieving the pass@1 > 0.33 metric on the __public__ leaderboard.
2. Up to 8 points for competition, calculated using the formula.    
  $$points = 8 \cdot \bigg(1 - \frac{\text{participant place} - 1}{\text{\\# participants}}\bigg),$$   
Where a participant's place is calculated based on a **private** leaderboard among participants who have completed a 4-point baseline. Thus, if 10 people have completed the baseline, 1st place receives 8 points, and 10th place receives 0.5 points.
3. Up to 2 points for report. If no report is submitted, the homework grade is reset to zero.

## Submitting Solutions

When submitting your solution, you will need to attach:
1. All code used.
2. A Jupiter notebook for generating answers for the test set, with explanations, that runs on Kaggle or Google Colab and generates answers in no more than 180 seconds on a single T4.
3. A report on the work done in PDF format. In the report, describe not only the final solution but also all the experiments that preceded it. What experiments did you conduct and what conclusions did you reach? How and for what tasks did you use agents, and how did this use influence your solution? In general, share everything you find interesting and relevant to your work.

## Hints

1. **Don't overcomplicate the solution.** Write a plan, decompose the problem into small chunks, and think through the solution thoroughly before jumping into action. There are countless ways to speed up your model and improve its quality. Start with the simplest and most reliable ones.
1. Write down the goal and setup of your experiments before running them, log all results, and **don't make more than one change at a time.** Firstly, this will help you more consciously choose experiments and avoid unnecessary runs. Secondly, this will be useful when writing your report.
1. **Choose the easiest hypotheses to test.** If testing a hypothesis takes 10+ hours, consider whether you really need it. Ideally, your final model should train in 24 hours maximum. Don't let the model train for a long time unless it's clearly necessary. 
1. **Start by optimizing the training cycle.** First, the faster the model learns, the faster you can test hypotheses. Second, the training speed usually correlates with the inference speed. And you need to speed up the inference speed.
