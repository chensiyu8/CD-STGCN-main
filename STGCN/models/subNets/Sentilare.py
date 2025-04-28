import torch
import torch.nn as nn
from .sentilare.modeling_sentilr_roberta import RobertaForSequenceClassification, RobertaModel
from pytorch_transformers import RobertaTokenizer as RT, RobertaConfig

__all__ = ['Sentilare_Model']

TRANSFORMERS_MAP = {
    'sentilare': (RobertaModel, RT)
}

class Sentilare_Model(nn.Module):
    def __init__(self, use_finetune=False, transformers='sentilare', pretrained='sentilare'):
        super().__init__()
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained)
        config = RobertaConfig.from_pretrained(pretrained, num_labels= 1 , finetuning_task='sst')
        self.model = model_class.from_pretrained(pretrained, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text, pos_tag_ids, senti_word_ids, polarity_ids):
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()

        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            pos_ids = pos_tag_ids,
                                            senti_word_ids = senti_word_ids,
                                            polarity_ids = polarity_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                pos_tag_ids=pos_tag_ids,
                                                senti_word_ids=senti_word_ids,
                                                polarity_ids=polarity_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)
        return last_hidden_states[0], last_hidden_states[1]
