import os
from datetime import date, timedelta
from os.path import join, dirname

import numpy as np
import praw
import requests
import torch
import tweepy as tw
from GoogleNews import GoogleNews
from dotenv import load_dotenv
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


class Sentiment:
    def __init__(
            self,
            model: str = os.path.join(os.getcwd(), 'models', 'pretrained_SA_model')
    ):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 32
        self._model = BertForSequenceClassification.from_pretrained(model)
        self._tokenizer = BertTokenizer.from_pretrained(model)

        dotenv_path = join(dirname(__file__), '.env')
        load_dotenv(dotenv_path)

    @staticmethod
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

        result = requests.get(url).json()

        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']

    def _reddit(self):

        self._client_id = os.environ.get("client_id")
        self._client_secret = os.environ.get("client_secret")
        self._user_agent = os.environ.get("user_agent")

        reddit_api = praw.Reddit(
            client_id=self._client_id,
            client_secret=self._client_secret,
            user_agent=self._user_agent
        )

        subreddits = 'stocks+options+wallstreetbets'

        posts = reddit_api.subreddit(subreddits).search(query=self._ticker + ' OR ' + self._name, sort='hot')

        self._posts = [post.title for post in posts]

    def _google(self):
        news = GoogleNews(period='1d')
        news.get_news(self._ticker) # @TODO how to also accomodate full name (case insensitive)
        self._news = news.get_texts()

    def _twitter(self):

        self._consumer_key = os.environ.get("consumer_key")
        self._consumer_secret = os.environ.get("consumer_secret")
        self._access_token = os.environ.get("access_token")
        self._access_token_secret = os.environ.get("access_token_secret")

        auth = tw.OAuthHandler(self._consumer_key, self._consumer_secret)
        auth.set_access_token(self._access_token, self._access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)

        tweets = tw.Cursor(
            api.search,
            q=self._ticker + ' OR ' + self._name,
            lang="en",
            since=date.today() - timedelta(days=1)
        ).items()

        self._tweets = [tweet.text for tweet in tweets]


    def _bert_preprocessing(self, sentences):
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

    @staticmethod
    def create_datasets(input_ids, attention_masks, batch_size):
        data = TensorDataset(input_ids, attention_masks)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader

    def _eval(self, dataloader):
        self._model.eval()
        predictions = []

        for batch in tqdm(dataloader):
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

    def _run(self, on):
        input_ids, attention_masks = self._bert_preprocessing(on)
        dataloader = Sentiment.create_datasets(input_ids, attention_masks, self.batch_size)
        self._eval(dataloader)

    def analyze(self, ticker):
        self._ticker = ticker
        self._name = Sentiment.get_symbol(ticker)

        print("### ANALYZING REDDIT POSTS ###")
        self._reddit()
        self._run(self._posts)

        print("### ANALYZING GOOGLE NEWS ###")
        self._google()
        self._run(self._news)

        print("### ANALYZING TWEETS ###")
        self._twitter()
        self._run(self._tweets)


if __name__ == '__main__':
    Sentiment().analyze('PINS')
