import requests
import praw
from GoogleNews import GoogleNews
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
import numpy as np
from getpass import getpass


class Sentiment:
    def __init__(self, ticker: str = None, model: str = './pretrained_SA_model/'):
        self._ticker = ticker
        self._name = Sentiment.get_symbol(ticker)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32

        self._model = BertForSequenceClassification.from_pretrained(model)
        self._tokenizer = BertTokenizer.from_pretrained(model)

    @staticmethod
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

        result = requests.get(url).json()

        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']

    def reddit(self):
        self._client_id = getpass(prompt='Client ID:')
        self._client_secret = getpass(prompt='Client Secret:')
        self._user_agent = getpass(prompt='User Agent:')
        self._reddit = True

        reddit_api = praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent
        )

        subreddits = 'stocks+options+wallstreetbets'

        posts = reddit_api.subreddit(subreddits).search(query=self._ticker + ' OR ' + self._name, sort='hot')

        self._posts = [post.title for post in posts]

    def google(self):
        news = GoogleNews(period='1d')
        news.get_news(self._ticker) # @TODO how to also accomodate full name (case insensitive)
        self._news = news.get_texts()

    def bert_preprocessing(self, sentences):
        input_ids = []
        attention_masks = []

        for sentence in sentences:
            encoded_dict = self._tokenizer.encode_plus(
                sentence,
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

        return input_ids, attention_masks

    def create_datasets(self, input_ids, attention_masks, batch_size):
        data = TensorDataset(input_ids, attention_masks)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def eval(self, dataloader):
        self._model.eval()
        predictions = []

        for batch in dataloader:
            batch = tuple(t.to(self._device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                result = self._model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    return_dict=True
                )

            logits = result.logits
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        predictions = np.concatenate(predictions)
        res = np.argmax(predictions, axis=1)

        print(f"{np.round(np.mean(res), 4) * 100}% positive on {len(res)} posts")

    def analyze(self):
        self.reddit()
        input_ids, attention_masks = self.bert_preprocessing(self._posts)
        dataloader = self.create_datasets(input_ids, attention_masks, self.batch_size)
        print("### ANALYZING REDDIT POSTS ###")
        self.eval(dataloader)

        self.google()
        input_ids, attention_masks = self.bert_preprocessing(self._news)
        dataloader = self.create_datasets(input_ids, attention_masks, self.batch_size)
        print("### ANALYZING GOOGLE NEWS ###")
        self.eval(dataloader)


if __name__ == '__main__':
    sentiment = Sentiment('MSFT')
    sentiment.analyze()
