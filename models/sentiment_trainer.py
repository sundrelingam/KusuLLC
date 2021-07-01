import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup


class SentimentTrainer:
    def __init__(self, data=os.path.join(os.getcwd(), 'data', 'stock_data.csv')):
        df = pd.read_csv(data)
        self._sentences = df.Text.values
        labels = df.Sentiment.values
        self._labels = np.where(labels==-1, 0, labels)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._batch_size = 32
        self._epochs = 2

        SentimentTrainer.set_seed()

        print('Loading BERT tokenizer...')
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        print('Loading BERT model...')
        self._model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        self._optimizer = AdamW(self._model.parameters(), lr=2e-5, eps=1e-8)

    def _tokenize(self):
        input_ids = []
        attention_masks = []

        for sentence in self._sentences:
            encoded_dict = self._tokenizer.encode_plus(
                sentence,  # Sentence to encode.
                padding='max_length',
                truncation=True,
                max_length=64,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self._labels)

        return input_ids, attention_masks, labels

    @staticmethod
    def create_datasets(input_ids, attention_masks, labels):
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # perform train/test split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def _create_dataloader(self, dataset):

        dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=self._batch_size
        )

        return dataloader

    @staticmethod
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        elapsed_rounded = int(round(elapsed))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _training_epoch(self, dataloader, scheduler):
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        self._model.train()

        for step, batch in enumerate(dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = SentimentTrainer.format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

            b_input_ids = batch[0].to(self._device)
            b_input_mask = batch[1].to(self._device)
            b_labels = batch[2].to(self._device)

            self._model.zero_grad()

            result = self._model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True
            )

            loss = result.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

            self._optimizer.step()

            scheduler.step()

        avg_train_loss = total_train_loss / len(dataloader)
        training_time = SentimentTrainer.format_time(time.time() - t0)

        return avg_train_loss, training_time

    def _validation_epoch(self, dataloader):
        print("")
        print("Running Validation...")

        t0 = time.time()

        self._model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in dataloader:

            b_input_ids = batch[0].to(self._device)
            b_input_mask = batch[1].to(self._device)
            b_labels = batch[2].to(self._device)

            with torch.no_grad():
                result = self._model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True
                )

            loss = result.loss
            logits = result.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += SentimentTrainer.flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(dataloader)
        validation_time = SentimentTrainer.format_time(time.time() - t0)

        return avg_val_loss, validation_time

    def train(self):
        input_ids, attention_masks, labels = self._tokenize()
        train_dataset, val_dataset = SentimentTrainer.create_datasets(input_ids, attention_masks, labels)
        train_dataloader = self._create_dataloader(train_dataset)
        val_dataloader = self._create_dataloader(val_dataset)

        total_steps = len(train_dataloader) * self._epochs
        scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(0, self._epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self._epochs))

            avg_train_loss, training_time = self._training_epoch(train_dataloader, scheduler)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            avg_val_loss, validation_time = self._validation_epoch(val_dataloader)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            print("")
            print("Training complete!")

    def save(self, dir=os.path.join(os.getcwd(), 'models', 'pretrained_SA_model')):
        model_to_save = self._model.module if hasattr(self._model, 'module') else self._model
        model_to_save.save_pretrained(dir)
        self._tokenizer.save_pretrained(dir)
