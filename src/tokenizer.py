import json

import pandas as pd


class MedTokenizer():
    def __init__(self, code_to_ix, ix_to_code, max_num_codes_in_visit, PADDING, MASK, CLS):
        self.code_to_ix = code_to_ix
        self.ix_to_code = ix_to_code
        self.max_length = max_num_codes_in_visit +1 # add CLS in every event
        self.PADDING = PADDING
        self.MASK = MASK
        self.CLS = CLS
        self.missed_codes_during_finetune = list()
        self.vocab_size = len(self.code_to_ix) + 3

    def tokenize(self, events):
        """
        @Input : list[event0, event1, event2..], where each event is a list of codes
        event_x = [codes]
        @output = list[encoded_event0, encoded_event1, encoded_event2..]
        encoded_event_x = [CLS_token, code_ids]
        """
        input_ids = []
        attention_masks = []
        for event in events:
            num_codes = len(event)
            coded_event = [self.CLS]
            for code in event:
                if(str(code) in self.code_to_ix):
                    coded_event.append(int(self.code_to_ix[str(code)]))
                else:
                    self.missed_codes_during_finetune.append(str(code))
                    #print("Could not find a integer mapping for code in\
                    #     pretraining data", code)
            attention_mask = [1] * len(coded_event)

            if len(coded_event) < self.max_length:
                n_pad = self.max_length - len(coded_event)
                for i in range(n_pad):
                    coded_event.append(self.PADDING)
                    attention_mask.append(0)
            input_ids.append(coded_event)
            attention_masks.append(attention_mask)
        return pd.Series([input_ids, attention_masks], index=['input_ids', 'attention_masks'])

    def save(self, outpath):
        tokenizer = {'code_to_ix' : self.code_to_ix,
                     'ix_to_code' : self.ix_to_code,
                     'max_codes_in_visit': int(self.max_length),
                     'PADDING': int(self.PADDING),
                     'MASK' : int(self.MASK),
                     'CLS' : int(self.CLS)}
        json.dump(tokenizer, open(outpath, "w+"))
        return

    def set_max_codes_in_visit(self, max_num_codes_in_visit):
        self.max_length = max_num_codes_in_visit + 1

    def get_missed_codes(self):
        return self.missed_codes_during_finetune


def load_tokenizer(outpath):
    with open(outpath, 'r') as f:
        tokenizer = json.load(f)
    code_to_ix = tokenizer['code_to_ix']
    ix_to_code = tokenizer['ix_to_code']
    PADDING = tokenizer['PADDING']
    MASK = tokenizer['MASK']
    CLS = tokenizer['CLS']
    max_length = tokenizer['max_codes_in_visit']

    formatted_code_to_ix = dict()
    for k,v in code_to_ix.items():
        formatted_code_to_ix[str(k)] = int(v)

    formatted_ix_to_code = dict()
    for k,v in ix_to_code.items():
        formatted_ix_to_code[int(k)] = str(v)

    med_tokenizer = MedTokenizer(formatted_code_to_ix, formatted_ix_to_code, max_length, PADDING,
                                 MASK, CLS)

    return med_tokenizer
