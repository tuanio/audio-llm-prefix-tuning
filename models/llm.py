import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM(nn.Module):
    def __init__(self, model_repo: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model = AutoModelForCausalLM.from_pretrained(model_repo)

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, inputs_embeds, labels):
        """
        to calculate the loss
        """
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
        return outputs.loss

    def generate(self, inputs_embeds):
        """
        to generate the text
        """
        outputs = self.model(inputs_embeds=inputs_embeds)
        return self.tokenizer.batch_decode(outputs)
