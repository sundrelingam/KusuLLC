from fundamental_analysis import Fundamentals
from sentiment_analysis import Sentiment
from technical_analysis import Technicals


def analyze(ticker):
    Sentiment().analyze(ticker)
    Fundamentals().analyze(ticker)
    Technicals().analyze(ticker)


if __name__ == '__main__':
    analyze("PINS")