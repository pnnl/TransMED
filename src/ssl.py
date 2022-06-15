import math

import torch
import torch.nn
from transformers import BertConfig, BertModel


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PatientSSLModel(torch.nn.Module):
    def __init__(self, pretrained_bert_model, d_model, vocab_size):
        super(PatientSSLModel, self).__init__()
        self.bert_embedding_model = pretrained_bert_model
        self.vocab_size = vocab_size

        self.norm = torch.nn.LayerNorm(d_model)#.cuda()
        self.linear = torch.nn.Linear(d_model, d_model)#.cuda()
        self.decoder = torch.nn.Linear(d_model, vocab_size)#.cuda()
        self.is_finetune = False

    def set_finetune(self, finetune):
        self.is_finetune = finetune


    def forward(self, input_ids, attention_mask, position_ids, masked_pos):

        bert_outputs = \
        self.bert_embedding_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids)
        sequence_output = bert_outputs[0]
        if(self.is_finetune == False):
            #print("Patient model sequence output size in forward function", sequence_output.size())

            masked_pos = masked_pos[:, :, None].expand(-1, -1, sequence_output.size(-1)) #[batch_size, maxlen, d_model]
            h_masked = torch.gather(sequence_output, 1, masked_pos) #masking position [batch_size, len, d_model]
            h_masked = self.norm(gelu(self.linear(h_masked)))
            #print("Patient model,masked embedding size after torch.gather", h_masked.size())
            pred_score = self.decoder(h_masked)
            #print("Patient model, decoder output size in forward function", pred_score.size())
            return pred_score
        else:
            #print("In finetuning mode, returning just visit.cls.embedding",
            #      sequence_output.size(), sequence_output[:, 0, :].size())
            return sequence_output[:, 0, :].squeeze(1)


class PatientFineTuneModel(torch.nn.Module):
    def __init__(self, d_model, num_static_and_temporal_steps, d_ffn, n_class,
                 dropout_rate=0.3, return_embedding=False):
        super().__init__()
        self.decoder1 = torch.nn.Linear(d_model*num_static_and_temporal_steps, d_ffn)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.decoder2 = torch.nn.Linear(d_ffn, n_class)
        self.return_embedding=return_embedding
        torch.nn.init.xavier_normal_(self.decoder1.weight)
        torch.nn.init.zeros_(self.decoder1.bias)
        torch.nn.init.xavier_normal_(self.decoder2.weight)
        torch.nn.init.zeros_(self.decoder2.bias)

    def set_return_embedding(self, val):
        self.return_embedding = val

    def forward(self, patient_embeddings):
        #given sequence_embeddings [batch_size, seq_len, d_model) and target
        #[batch_size, class_target], train the model to predict right class
        seq_embedding = torch.relu(self.dropout(self.decoder1(patient_embeddings)))
        if(self.return_embedding):
            return seq_embedding
        scores = torch.sigmoid(self.decoder2(seq_embedding))
        return scores

class PatientFineTuneGRUModel(torch.nn.Module):
    def __init__(self, d_model, num_static_and_temporal_steps, d_ffn, n_class,\
                 batch_size, dropout_rate=0.25, return_embedding=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_static_and_temporal_steps = num_static_and_temporal_steps
        self.return_embedding=return_embedding
        self.d_model = d_model
        n_gru_layers = 4
        d_gru= d_ffn*2
        self.d_gru = d_gru
        self.gru = torch.nn.GRU(d_model, d_gru, n_gru_layers, batch_first=True, dropout=0.3, bidirectional=False)
        self.decoder1 = torch.nn.Linear(d_gru*num_static_and_temporal_steps, d_ffn)
        self.dropout = torch.nn.Dropout(dropout_rate)
        print("Drop out in finetuning dense layer ", dropout_rate)
        self.decoder2 = torch.nn.Linear(d_ffn, n_class)
        torch.nn.init.xavier_normal_(self.decoder1.weight)
        torch.nn.init.zeros_(self.decoder1.bias)
        torch.nn.init.xavier_normal_(self.decoder2.weight)
        torch.nn.init.zeros_(self.decoder2.bias)

    def set_return_embedding(self, val):
        self.return_embedding = val

    def forward(self, patient_embeddings):
        #given sequence_embeddings [batch_size, seq_len, d_model) and target
        #[batch_size, class_target], train the model to predict right class
        patient_embeddings = torch.reshape(patient_embeddings, (self.batch_size,
                                                                self.num_static_and_temporal_steps,
                                                                self.d_model))
        #print("finetune bert, patient_embedding after reshape",
        #      patient_embeddings.size())
        patient_embeddings , hidden = self.gru(patient_embeddings) #batch*seq_length*d_ffn
        #print("finetune GRU,  patient_embedding after GRU output",\
        #      patient_embeddings.size())
        # take hidden state from last time step
        patient_embeddings = torch.reshape(patient_embeddings, (self.batch_size,
                                                                self.num_static_and_temporal_steps*self.d_gru))
        #patient_embeddings = patient_embeddings[:, -1, :].squeeze(1)
        #print("finetune GRU,  patient_embedding from GRU last hidden layer",\
        #      patient_embeddings.size())

        # pass through dense layers
        seq_embedding = torch.relu(self.dropout(self.decoder1(patient_embeddings)))
        if(self.return_embedding):
            return seq_embedding
        scores = torch.sigmoid(self.decoder2(seq_embedding))
        return scores



class PatientFineTuneBERTModel(torch.nn.Module):
    def __init__(self, d_model, num_static_and_temporal_steps, d_ffn, n_class, \
                 batch_size, dropout_rate=0.2, return_embedding=False):
        config = BertConfig(
                         vocab_size=num_static_and_temporal_steps+1,  # FOR cls, masking
                         hidden_size=d_model,
                         num_hidden_layers=2,
                         num_attention_heads=2,
                         max_position_embeddings=num_static_and_temporal_steps+1,
                         pad_token_id=num_static_and_temporal_steps,
                         intermediate_size=4*d_model,
                     )

        self.bert_patient_stay_encoder = BertModel(config)
        self.decoder1 = torch.nn.Linear(d_model, d_ffn)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.decoder2 = torch.nn.Linear(d_ffn, n_class)
        self.num_static_and_temporal_steps = num_static_and_temporal_steps
        self.d_model = d_model
        self.batch_size = batch_size
        self.return_embedding=return_embedding

        #Init layer weights
        torch.nn.init.xavier_normal_(self.decoder1.weight)
        torch.nn.init.zeros_(self.decoder1.bias)
        torch.nn.init.xavier_normal_(self.decoder2.weight)
        torch.nn.init.zeros_(self.decoder2.bias)

    def set_return_embedding(self, val):
        self.return_embedding = val


    def forward(self, patient_embeddings):
        #given sequence_embeddings [batch_size, seq_len, d_model) and target
        #[batch_size, class_target], train the model to predict right class
        patient_embeddings = torch.reshape(patient_embeddings, (self.batch_size,
                                                                self.num_static_and_temporal_steps,
                                                                self.d_model))
        #print("finetune bert, patient_embedding after reshape",
        #      patient_embeddings.size())

        patient_embeddings = self.bert_patient_stay_encoder(inputs_embeds=patient_embeddings)[0]
        #print("finetune bert, patient_embeddings after BERT encoder",
        #      patient_embeddings.size())

        patient_embeddings = patient_embeddings[:, 0, :].squeeze(1)

        #print("finetune bert, first timestep embedding size", patient_embeddings.size())
        seq_embedding = torch.relu(self.dropout(self.decoder1(patient_embeddings)))
        if(self.return_embedding):
            return seq_embedding
        scores = torch.sigmoid(self.decoder2(seq_embedding))
        return scores
