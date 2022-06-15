import time
from pathlib import Path

import sklearn.metrics
import torch
import torch.nn
from torch.nn import LogSoftmax, NLLLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class SSLFineTuneRunner:

    def __init__(
        self,
        pretrain_model,
        finetune_model,
        num_epochs,
        batch_size,
        patience,
        patience_threshold,
        finetune_dir,
        args,
        device,
        class_weights,
    ):
        self.pretrain_model = pretrain_model.to(device)
        self.finetune_model = finetune_model.to(device)
        self.num_time_steps = args.num_time_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.patience_threshold = patience_threshold
        self.lr = args.ft_lr
        self.scheduler_patience = args.ft_scheduler_patience

        #self.optimizer = SGD(self.finetune_model.parameters(), lr=self.lr, momentum=0.9)
        print(f"Using Adam optimizer")
        self.optimizer = Adam(self.finetune_model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.scheduler_patience, factor=0.1, verbose=True)
        self.nlog_prob = LogSoftmax(dim=1)
        self.criterion = NLLLoss(weight=class_weights)
        self.finetune_dir = finetune_dir
        self.device = device
        self.topk = args.topk
        self.args = args

    def train_epoch(self, training_data, val_data):
        # enable training model
        self.finetune_model.train()
        total_loss = 0.0
        start_time = time.time()
        data_size = training_data.size()
        total_loss = 0.0
        log_interval = self.args.log_interval

        num_samples_seen = 0
        #pbar = tqdm(enumerate(range(0, data_size - 1, self.batch_size)))
        #for batch_id, batch_start_index in pbar:
        for batch_id, batch_start_index in enumerate(range(0, data_size - 1, self.batch_size)):
            (
                batch_input,
                batch_target,
                batch_attention_mask,
                batch_masked_position,
                batch_position_ids,
                batch_static_embeddings,
                batch_labvalues,
            ) = self.get_batch(training_data, batch_start_index)

            # get event embeddings as pretraining model output
            with torch.no_grad():
                model_batch_output = self.pretrain_model(
                    input_ids=batch_input,
                    attention_mask=batch_attention_mask,
                    position_ids=batch_position_ids,
                    masked_pos=batch_masked_position,
                )
                # print("pretrain model output.shape", model_batch_output.size())

            num_patients_batch = int(self.batch_size / self.num_time_steps)
            all_patient_embeddings, all_patient_outputs = self.get_patient_embedding(
                batch_static_embeddings,
                batch_labvalues,
                model_batch_output,
                batch_target,
                self.num_time_steps,
            )

            # print("finetuning model batch input size", all_patient_embeddings.size(), all_patient_outputs.size())

            model_batch_output = self.finetune_model(all_patient_embeddings)
            # print("In finetunning training, predicted output, target output size",\
            #      model_batch_output.size(), all_patient_outputs.size())
            scores = self.nlog_prob(model_batch_output)
            loss = self.criterion(scores, all_patient_outputs.squeeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.finetune_model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

            # avg_loss = total_loss/(num_samples_seen)
            # pbar.set_description('total_loss: %f' % avg_loss)

            # log progress every 200 seconds

            if batch_id % log_interval == 0 and batch_id > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                lr = [group["lr"] for group in self.optimizer.param_groups][0]
                print(
                    "| {:5d}/{:5d} batches | "
                    "lr {:02.4f} | ms/batch {:5.2f} | "  #'loss {:5.4f} | ppl {:8.2f}'.format( \
                    "loss {:5.4f}".format(
                        batch_id,
                        data_size // num_patients_batch,
                        lr,
                        elapsed * 1000 / log_interval,
                        cur_loss,
                    ),
                    flush=True,
                )
                # cur_loss, math.exp(cur_loss)))
                #pbar.set_description("cur_loss %f" % cur_loss)
                # total_loss = 0.0
                # start_time = time.time()
                # self.evaluate(val_data)
                # self.finetune_model.train()

    def run_train(self, training_data, val_data):
        # runs all training epochs
        # computing validation loss after very epoch and
        # reports testset accuracy
        best_epoch = -1
        best_val_loss = float("inf")

        num_batches = int(training_data.size() / self.batch_size)
        data_size = num_batches * self.batch_size
        training_data.resize(data_size)

        num_val_batches = int(val_data.size() / self.batch_size)
        val_data_size = num_val_batches * self.batch_size
        val_data.resize(val_data_size)

        best_model = None
        wait = self.patience

        print(
            f"using config {self.pretrain_model.bert_embedding_model.config}",
            flush=True,
        )

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.train_epoch(training_data, val_data)
            # validate now
            val_loss = self.evaluate(val_data)
            print("-" * 89, flush=True)
            # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' \
            #      'valid ppl {:8.2f}'.format(epoch, (time.time() - \
            #                                          epoch_start_time),
            #                                 val_loss, math.exp(val_loss)))
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | ".format(
                    epoch, (time.time() - epoch_start_time), val_loss
                ),
                flush=True,
            )
            # Using rel patience threshold
            # example: https://pytorch.org/docs/1.9.0/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            best_val_loss_to_beat = best_val_loss * (1 - self.patience_threshold)
            if val_loss < best_val_loss_to_beat:
                best_epoch = epoch
                best_model = self.finetune_model
                self.save_model(best_model, best_epoch)
                best_val_loss = val_loss
                wait = self.patience
                print("Found better val_loss.", flush=True)
            else:
                wait -= 1
                print(f"Wait {wait} more epochs to check.", flush=True)
            if wait == 0:
                break
            self.scheduler.step(val_loss)
        print(f"Best Epoch {best_epoch}", flush=True)
        return best_model, best_epoch

    def get_patientid_with_all_timesteps(self, batch_patient_id_with_timestep):
        """
        take as input a list of __len__ = self.batch_size
        returns a list concatenating each successive 'num_time_steps' elements
        of size (self.batch_size/self.num_time-steps)
        """
        batch_patient_id_with_all_timesteps = []
        for i in range(0, len(batch_patient_id_with_timestep),
                       self.num_time_steps):
            x = ";".join(batch_patient_id_with_timestep[i:i+self.num_time_steps]) + ";"
            #print(x)
            batch_patient_id_with_all_timesteps.append(x)
        return batch_patient_id_with_all_timesteps


    def run_test(self, best_model, test_data, return_score_dict_only = False):
        self.finetune_model.eval()
        #print("Running #samples with batch size ",  test_data.size(), self.batch_size)
        num_batches = int(test_data.size() / self.batch_size)
        data_size = num_batches * self.batch_size
        test_data.resize(data_size)
        y_pred = []
        y_true = []
        y_scores = []
        patient_id_with_timesteps = []

        with torch.no_grad():
            #for batch_id, batch_start_index in tqdm(
            #    enumerate(range(0, data_size - 1, self.batch_size))
            #):
            for batch_id, batch_start_index in enumerate(range(0, data_size - 1, self.batch_size)):
                (
                    batch_input,
                    batch_target,
                    batch_attention_mask,
                    batch_masked_position,
                    batch_position_ids,
                    batch_static_embeddings,
                    batch_labvalues,
                ) = self.get_batch(test_data, batch_start_index)

                # get event embeddings as pretraining model output
                model_batch_output = self.pretrain_model(
                    input_ids=batch_input,
                    attention_mask=batch_attention_mask,
                    position_ids=batch_position_ids,
                    masked_pos=batch_masked_position,
                )
                (
                    all_patient_embeddings,
                    all_patient_outputs,
                ) = self.get_patient_embedding(
                    batch_static_embeddings,
                    batch_labvalues,
                    model_batch_output,
                    batch_target,
                    self.num_time_steps,
                )

                #print(all_patient_embeddings, all_patient_outputs)

                batch_patient_id_with_all_time_steps = self.get_patientid_with_all_timesteps(test_data.patient_id_with_timestep[batch_start_index:batch_start_index+self.batch_size])

                patient_scores = best_model(all_patient_embeddings).data.cpu()

                #print("size of batch_patient_ids_with_all_timesteps, batch_patient_scores, SHOULD BE EQUAL",
                #len(batch_patient_id_with_all_time_steps), len(list(patient_scores)))

                y_pred += [
                    list(pt_score).index(max(list(pt_score)))
                    for pt_score in list(patient_scores)
                ]
                y_true += all_patient_outputs.squeeze(1).data.cpu().numpy().tolist()
                y_scores += patient_scores.numpy()[:, 1].tolist()
                patient_id_with_timesteps += \
                batch_patient_id_with_all_time_steps

                print(f"{len(y_true)}, {patient_scores.shape}", flush=True)
        patient_score_dict = {}
        for i in range(len(patient_id_with_timesteps)):
            patient_score_dict[patient_id_with_timesteps[i]] = [y_pred[i],
                                                                y_true[i],
                                                                y_scores[i]]
        if(return_score_dict_only == True):
            return patient_score_dict
        return self.get_accuracy(y_true, y_pred, y_scores), patient_score_dict

    def get_accuracy(self, y_true, y_pred, y_scores):
        print(
            f"Confusion matrix: {sklearn.metrics.confusion_matrix(y_true, y_pred)}",
            flush=True,
        )
        (
            precision,
            recall,
            avg_fscore,
            support,
        ) = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
        auroc = sklearn.metrics.roc_auc_score(y_true, y_scores)
        precision_, recall_, thresholds = sklearn.metrics.precision_recall_curve(
            y_true, y_scores
        )
        auprc = sklearn.metrics.auc(recall_, precision_)
        f1_binary = sklearn.metrics.f1_score(y_true, y_pred, average="binary")
        f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
        f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
        f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
        print(f"f1_binary: {f1_binary}", flush=True)
        print(f"f1_micro: {f1_micro}", flush=True)
        print(f"f1_macro: {f1_macro}", flush=True)
        print(f"f1_weighted: {f1_weighted}", flush=True)
        precision_score = sklearn.metrics.precision_score(y_true, y_pred)
        recall_score = sklearn.metrics.recall_score(y_true, y_pred)
        print(f"precision_score: {precision_score}", flush=True)
        print(f"recall_score: {recall_score}", flush=True)
        return (
            precision,
            recall,
            avg_fscore,
            support,
            f1_micro,
            auroc,
            auprc,
        )

    def evaluate(self, val_data):
        # enable evaluation model
        self.finetune_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            data_size = int(val_data.size())
            print(f"Running validation data size {data_size}", flush=True)

            for batch_id, batch_start_index in enumerate(
                range(0, data_size - 1, self.batch_size)
            ):
                (
                    batch_input,
                    batch_target,
                    batch_attention_mask,
                    batch_masked_position,
                    batch_position_ids,
                    batch_static_embeddings,
                    batch_labvalues,
                ) = self.get_batch(val_data, batch_start_index)

                # get event embeddings as pretraining model output
                model_batch_output = self.pretrain_model(
                    input_ids=batch_input,
                    attention_mask=batch_attention_mask,
                    position_ids=batch_position_ids,
                    masked_pos=batch_masked_position,
                )

                num_patients_batch = int(self.batch_size / self.num_time_steps)
                (
                    all_patient_embeddings,
                    all_patient_outputs,
                ) = self.get_patient_embedding(
                    batch_static_embeddings,
                    batch_labvalues,
                    model_batch_output,
                    batch_target,
                    self.num_time_steps,
                )

                pred_prob = self.finetune_model(all_patient_embeddings)
                loss = self.criterion(
                    self.nlog_prob(pred_prob), all_patient_outputs.squeeze(1)
                )
                batch_loss = num_patients_batch * loss
                total_loss += batch_loss
        return total_loss / (data_size - 1)

    def get_patient_embedding(
        self, batch_static_embeddings, batch_labvalues, model_batch_output, batch_target, num_time_steps
    ):
        num_patients_batch = int(self.batch_size / self.num_time_steps)
        demo_or_nlp = ("demo" in self.args.features) or ("nlp" in self.args.features)
        add_time_step = demo_or_nlp
        embed_size = (
            (self.num_time_steps + 1)
            if add_time_step
            else self.num_time_steps
        )

        all_patient_embeddings = torch.empty(
            num_patients_batch, embed_size * self.args.hidden_dim
        ).to(self.device)
        all_patient_outputs = (
            torch.empty(num_patients_batch, 1).to(self.device).to(dtype=int)
        )
        # get final patient embedding by combining static attribute
        # embeddings + all temporal embeddings from PRETRAIN-BERT
        for p_idx, i in enumerate(range(0, self.batch_size - 1, self.num_time_steps)):
            # note that the static_embeddings and patient outcome tensors
            # are constructed to be same length as other temporal patient
            # tensors by repeating num_time_steps time for each patient
            # i.e. static_embeddings[i: i+num_time_steps] = \
            # patient_static_embeddings * num_time_teps
            #if "demo" in self.args.features:
            if demo_or_nlp:
                patient_embedding = batch_static_embeddings[i].to(dtype=float)
                for j in list(range(i, i + self.num_time_steps)):
                    if 'val' in self.args.features:
                        patient_embedding = torch.cat(
                            (patient_embedding, model_batch_output[j].to(dtype=float), batch_labvalues[j])
                        )
                    else:
                        patient_embedding = torch.cat(
                            (patient_embedding, model_batch_output[j].to(dtype=float))
                        )
            else:
                patient_embedding = model_batch_output[i].to(dtype=float)
                for j in list(range(i + 1, i + self.num_time_steps)):
                    if 'val' in self.args.features:
                        patient_embedding = torch.cat(
                            (patient_embedding, model_batch_output[j].to(dtype=float), batch_labvalues[j])
                        )
                    else:
                        patient_embedding = torch.cat(
                            (patient_embedding, model_batch_output[j].to(dtype=float))
                        )

            all_patient_embeddings[p_idx] = patient_embedding
            all_patient_outputs[p_idx] = batch_target[i].to(dtype=int)
            #all_patient_outputs[p_idx] = int(bool(batch_target[i].to(dtype=int)))
        return all_patient_embeddings, all_patient_outputs

    def get_batch(self, finetune_data, batch_start_index):
        batch_input = finetune_data.x[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_output = finetune_data.y[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_attention_mask = finetune_data.mask[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_masked_positions = finetune_data.masked_posn[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_position_ids = finetune_data.position_ids[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_static_embeddings = finetune_data.static_embeddings[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_labvalues = finetune_data.labvalues[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        return (
            batch_input,
            batch_output,
            batch_attention_mask,
            batch_masked_positions,
            batch_position_ids,
            batch_static_embeddings,
            batch_labvalues
        )

    def save_model(self, model, best_epoch):
        # make sure checkpoints directory exists
        checkpoints_dir = f"{self.finetune_dir}/checkpoints"
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        # save checkpoint to checkpoints dir
        torch.save(
            model.state_dict(),
            f"{checkpoints_dir}/finetune_model.{best_epoch}.ckpt",
        )
