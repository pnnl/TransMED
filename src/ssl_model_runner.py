import time
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SSLRunner:
    def __init__(
        self,
        model,
        num_epochs,
        batch_size,
        patience,
        patience_threshold,
        pretrain_dir,
        args,
        device,
    ):
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.patience_threshold = patience_threshold
        self.lr = args.pt_lr
        self.scheduler_patience = args.pt_scheduler_patience
        #self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=self.scheduler_patience, factor=0.1, verbose=True)
        self.criterion = CrossEntropyLoss()
        self.pretrain_dir = pretrain_dir
        self.device = device
        self.topk = args.topk
        self.args = args

    def train_epoch(self, inputs, outputs, src_mask, masked_position):
        # enable training model
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        data_size = inputs.size(0)
        total_loss = 0.0
        log_interval = self.args.log_interval

        for batch_id, batch_start_index in enumerate(
            range(0, data_size - 1, self.batch_size)
        ):
            (
                batch_input,
                batch_target,
                attention_mask_batch,
                batch_masked_position,
            ) = self.get_batch(
                inputs, outputs, src_mask, masked_position, batch_start_index
            )
            seq_length = batch_input.shape[1]
            position_ids = [1] * seq_length
            position_ids = torch.tensor([position_ids] * self.batch_size).to(
                self.device
            )
            # position_ids = torch.ones(attention_mask_batch.size(),
            #                          dtype=torch.int32, device=self.device)

            # print("in batch: tensor size (input, output/masked_node_ids, position tensor, masked positions)", batch_input.size(), \
            #      batch_target.size(), position_ids.size(),
            #      batch_masked_position.size())

            model_batch_output = self.model(
                input_ids=batch_input,
                attention_mask=attention_mask_batch,
                position_ids=position_ids,
                masked_pos=batch_masked_position,
            )
            model_batch_output = model_batch_output.transpose(1, 2)
            # masked_nodes = torch.LongTensor(batch_target)
            masked_nodes = batch_target
            # print("ypred and ytrue size", model_batch_output.size(),\
            #                        masked_nodes.size())
            loss = self.criterion(model_batch_output, masked_nodes)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
            # log progress every 200 seconds

            if batch_id % log_interval == 0 and batch_id > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                lr = [group["lr"] for group in self.optimizer.param_groups][0]
                print(
                    "| {:5d}/{:5d} batches | "
                    "lr {:02.8f} | "
                    "ms/batch {:5.2f} | "  #'loss {:5.4f} | ppl {:8.2f}'.format( \
                    "loss {:5.4f}".format(
                        batch_id,
                        data_size // self.batch_size,
                        lr,
                        elapsed * 1000 / log_interval,
                        cur_loss,
                    ),
                    flush=True,
                )
                total_loss = 0.0
                start_time = time.time()
                self.evaluate()
                self.model.train()

    def run_train(
        self,
        x_train,
        y_train,
        mask_train,
        masked_positions_train,
        x_val,
        y_val,
        mask_val,
        masked_positions_val,
    ):
        # runs all training epochs
        # computing validation loss after very epoch and
        # reports testset accuracy
        best_epoch = -1
        best_val_loss = float("inf")
        num_batches = int(x_train.size(0) / self.batch_size)
        data_size = num_batches * self.batch_size
        inputs = x_train[0:data_size]
        outputs = y_train[0:data_size]
        mask_train = mask_train[0:data_size]
        masked_positions_train = masked_positions_train[0:data_size]

        num_val_batches = int(x_val.size(0) / self.batch_size)
        val_data_size = num_val_batches * self.batch_size
        self.x_val = x_val[0:val_data_size]
        self.y_val = y_val[0:val_data_size]
        self.mask_val = mask_val[0:val_data_size]
        self.masked_positions_val = masked_positions_val[0:val_data_size]
        # self.validation_inputs = (x_val, y_val, mask_val, masked_positions_val)

        best_model = None
        wait = self.patience

        print(f"using config {self.model.bert_embedding_model.config}", flush=True)
        print(
            f"using config return_dict {self.model.bert_embedding_model.config.use_return_dict}",
            flush=True,
        )

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.train_epoch(inputs, outputs, mask_train, masked_positions_train)
            # validate now
            val_loss = self.evaluate()
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
                best_model = self.model
                self.save_model(best_model, best_epoch)
                best_val_loss = val_loss
                wait = self.patience
                print(f"Found better val_loss.", flush=True)
            else:
                wait -= 1
                print(f"Wait {wait} more epochs to check.", flush=True)
            if wait == 0:
                break
            self.scheduler.step(val_loss)
        print(f"Best Epoch {best_epoch}", flush=True)
        return best_model, best_epoch

    def run_test(self, best_model, x_test, y_test, src_mask, masked_positions):
        self.model.eval()
        num_batches = int(x_test.size(0) / self.batch_size)
        data_size = num_batches * self.batch_size
        y_pred = []
        y_true = []
        x_test = x_test[0:data_size]
        y_test = y_test[0:data_size]
        src_mask = src_mask[0:data_size]
        masked_positions = masked_positions[0:data_size]

        with torch.no_grad():
            for batch_id, batch_start_index in enumerate(
                range(0, data_size - 1, self.batch_size)
            ):
                (
                    batch_input,
                    batch_target,
                    batch_attention_mask,
                    batch_masked_position,
                ) = self.get_batch(
                    x_test, y_test, src_mask, masked_positions, batch_start_index
                )
                seq_length = batch_input.shape[1]
                position_ids = [1] * seq_length
                position_ids = torch.tensor([position_ids] * self.batch_size).to(
                    self.device
                )
                masked_nodes = batch_target
                batch_pred_prob = best_model(
                    batch_input,
                    batch_attention_mask,
                    position_ids,
                    batch_masked_position,
                )

                prob = torch.softmax(batch_pred_prob, dim=2)
                score, index = batch_pred_prob.topk(self.topk, dim=2)
                # print("Predicting class and score", index, score)
                # print(index.size(), score.size())
                y_pred += list(index.squeeze(2).data.cpu().numpy())
                y_true += list(masked_nodes.data.cpu().numpy())
        return self.get_accuracy(y_true, y_pred)

    def get_accuracy(self, y_true, y_pred):
        print(f"Sample y_true {y_true[0:5]}", flush=True)
        print(f"Sample y_pred {y_pred[0:5]}", flush=True)
        num_pos = 0
        # TODO for multi node masking, define accuracy
        # This only calculates for n_mask =1
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                # number of masked node ids
                if y_true[i][j] in y_pred[i]:
                    num_pos += 1
        accuracy = float(num_pos) / float(len(y_true) * len(y_true[0]))
        print(f"accuracy of masked node prediction {accuracy}", flush=True)
        return accuracy

        # print(sklearn.metrics.precision_recall_fscore_support(y_true, y_pred))

    def evaluate(self):
        # enable evaluation model
        inputs = self.x_val
        outputs = self.y_val
        src_mask = self.mask_val
        masked_positions = self.masked_positions_val
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            data_size = int(inputs.size(0))

            for batch_id, batch_start_index in enumerate(
                range(0, data_size - 1, self.batch_size)
            ):
                (
                    batch_input,
                    batch_target,
                    batch_attention_mask,
                    batch_masked_position,
                ) = self.get_batch(
                    inputs, outputs, src_mask, masked_positions, batch_start_index
                )
                seq_length = batch_input.shape[1]
                position_ids = [1] * seq_length
                position_ids = torch.tensor([position_ids] * self.batch_size).to(
                    self.device
                )
                batch_output = self.model(
                    batch_input,
                    batch_attention_mask,
                    position_ids,
                    batch_masked_position,
                )
                masked_nodes = batch_target
                loss = self.criterion(batch_output.transpose(1, 2), masked_nodes)
                batch_loss = len(batch_input) * loss
                total_loss += batch_loss
        return total_loss / (data_size - 1)

    def get_batch(
        self, inputs, outputs, attention_mask, masked_positions, batch_start_index
    ):
        batch_input = inputs[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_output = outputs[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_attention_mask = attention_mask[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        batch_masked_positions = masked_positions[
            batch_start_index : batch_start_index + self.batch_size
        ].to(self.device)
        return batch_input, batch_output, batch_attention_mask, batch_masked_positions

    def save_model(self, model, epoch):
        # make sure checkpoints directory exists
        checkpoints_dir = f"{self.pretrain_dir}/checkpoints"
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
        # save checkpoint to checkpoints dir
        torch.save(
            model.state_dict(), f"{checkpoints_dir}/pretrain_model.{epoch}.ckpt"
        )
