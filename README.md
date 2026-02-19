This branch contains the code and datasets with which I try to replicate the research of Christensen, Siggaard and Veliyev (2021), called "A machine learning approach to volatility forecasting".
As this project is mainly for familiarization purposes with the econometric and ML techniques adopted and how to code them, as well as the specific data handling, (and the data they use was inaccessible for me :( ),
I use EURUSD exchange rate tick data from November 2025-January 2026, so with intraday trading data (bid and ask). 

- Data_handling.py: Used to import the data, calculate midprice, log-returns, daily realized variance
- HAR_regressions.py: Used to construct the regressor spaces for the respective variants of the HAR model used in the paper. After that, calculated the rolling forecasts for the models and MSE.
