# Automated-trading-system
This project creates an automated trading system in Python and tests its performance against the performance of the FTSE100 index.


The aim is to build an algorithmic trading system for the equity market. The AT system I am designing is conceptually made up of two parts: return prediction and decision making. For predicting the returns of the stocks, I am using a Long Short Term Memory (LSTM) network (an advanced neural network) for price and return prediction. The system uses these predictions to make entry and exit decisions. The data used for this is financial time-series data from the UK acquired using the Bloomberg terminal. I compare the performance of this algorithm on 2 datasets of UK traded stocks with the performance of the market itself, for which the FTSE100 index was used as proxy. The hypothesis is that insofar as the market is efficient, the average returns over one year for the algorithm should be lower than for the index fund.

