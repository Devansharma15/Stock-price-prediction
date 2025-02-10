Stock Price Prediction using LSTM (PyTorch)

📌 Overview: 
This project predicts stock prices using Long Short-Term Memory (LSTM) networks built with PyTorch. The model learns from historical stock prices and forecasts future trends.



⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/devanshsharma15/stock-price-prediction.git
cd stock-price-prediction

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run Training

python main.py

4️⃣ Make Predictions

python predict.py

📊 How It Works:

1.Fetches stock price data from Yahoo Finance (yfinance).

2.Preprocesses data using Min-Max Scaling.

3.Creates sequences where past 50 days predict the next day's price.

4.Trains an LSTM model with PyTorch.

5.Evaluates performance using Mean Squared Error (MSE).

6.Plots actual vs. predicted prices.

🚀 Results:

Blue Line: Actual stock prices.

Red Line: Predicted stock prices.



📌 Key Features:

✅ Uses PyTorch for deep learning.
✅ Supports custom stock ticker selection.
✅ Implements MinMaxScaler for data normalization.
✅ Utilizes DataLoader for batch training.
✅ Saves and evaluates model performance.

🔧 Possible Improvements:

Use GRU or Transformers for better predictions.

Hyperparameter tuning (learning rate, sequence length, layers).

Add external data sources (news sentiment, trading volume).

Deploy as a Flask API or Streamlit app.

🛠️ Dependencies

torch

numpy

pandas

matplotlib

sklearn

yfinance

💡 Contributing

Feel free to fork this repository and submit a PR with enhancements! 🚀

📜 License

MIT License

