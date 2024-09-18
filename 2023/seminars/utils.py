import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm


class GPTLoss:
    def __init__(self, device="cpu"):
        self.name = "ai-forever/rugpt3large_based_on_gpt2"
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(self.name).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)

    @torch.no_grad()
    def __call__(self, text, reduce="mean"):
        inputs = self.tokenizer(text, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        loss = self.model(**inputs, labels=inputs["input_ids"]).loss.detach().cpu()
        num_tokens = torch.sum(inputs["attention_mask"]).item()

        if reduce == "sum":
            return loss.item() * num_tokens, num_tokens

        return loss.item(), num_tokens
    
    def compute(self, texts):
        num_tokens = 0.0
        metric = 0.0
        for text in tqdm(texts):
            if len(text) == 0:
                continue
            t_metric, t_num = self(text, reduce="sum")
            if t_metric is None or np.isnan(t_metric):
                continue
            metric += t_metric
            num_tokens += t_num
        return metric / num_tokens

    
def tokenize(text):
    reg = re.compile(r'\w+')
    return reg.findall(text.lower())


def unique_words_rate(texts):
    words = set()
    n_words = 0
    for text in texts:
        for word in tokenize(text):
            n_words += 1
            words.add(word)

    return len(words) / n_words


def avg_token_entropy(probs):
    entropy = -(torch.log(probs) * probs).nansum(-1)
    return entropy.mean()


@torch.no_grad()
def perplexity(model, texts, tokenizer):
    """
    :param min_prob: if P(w | ...) is smaller than min_prop, set it to min_prob.
    :returns: mean perplexity over the whole corpus
    """
    
    device = next(model.parameters()).device
    
    tokenized = tokenizer(texts, padding=True, return_tensors='pt')
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    logits, _ = model(input_ids)
    logits = logits[:, :-1]
    targets = input_ids[:, 1:].to(device)

    losses = F.cross_entropy(
        input=logits.reshape(-1, logits.shape[-1]),
        target=targets.reshape(-1),
        reduce=False,
    )
    mask = attention_mask[:, 1:]
    losses = losses * mask.reshape(-1)
    loss = torch.sum(losses) / torch.sum(mask)
    ppls = torch.exp(loss)
    return ppls
    

def clear_text(texts):
    clear_texts = []
    for text in texts:
        text = text.replace('[PAD]', '')
        cls_pos = text.find('[CLS]')
        if cls_pos > -1:
            text = text[cls_pos + len('[CLS]'):]

        sep_pos = text.find('[SEP]')
        if sep_pos > -1:
            text = text[:sep_pos]

        clear_texts.append(text.strip())

    return clear_texts