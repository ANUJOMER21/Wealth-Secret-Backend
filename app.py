import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
from flask import Flask, jsonify, request, send_file, send_from_directory
import requests
import pandas as pd
import matplotlib as plt

plt.use('agg')
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

app = Flask(__name__)


# Function to fetch all NSE tickers


# Endpoint to get Nifty and Sensex data
@app.route('/indices', methods=['GET'])
def get_indices():
    nifty = yf.Ticker("^NSEI")  # Nifty 50 index ticker
    sensex = yf.Ticker("^BSESN")  # Sensex index ticker

    nifty_data = nifty.history(period="1d")
    sensex_data = sensex.history(period="1d")

    # Convert numpy.int64 to Python int
    nifty_data = nifty_data.astype(float)
    sensex_data = sensex_data.astype(float)

    response = {
        'Nifty': {
            'current_price': float(nifty_data['Close'].iloc[-1]),
            'open': float(nifty_data['Open'].iloc[-1]),
            'high': float(nifty_data['High'].iloc[-1]),
            'low': float(nifty_data['Low'].iloc[-1]),
            'volume': float(nifty_data['Volume'].iloc[-1])
        },
        'Sensex': {
            'current_price': float(sensex_data['Close'].iloc[-1]),
            'open': float(sensex_data['Open'].iloc[-1]),
            'high': float(sensex_data['High'].iloc[-1]),
            'low': float(sensex_data['Low'].iloc[-1]),
            'volume': float(sensex_data['Volume'].iloc[-1])
        }
    }

    return jsonify(response)


def create_pdf_with_chart(symbol, current_price, option_chain_file, output_path, calls_df, puts_df):
    # Create a PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(f"<b>Option Chain Report for {symbol}</b>", styles["Title"])
    current_price_text = Paragraph(f"<b>Current Price: {current_price}</b>", styles["Normal"])

    # Add chart image
    chart_image = Image(option_chain_file, width=400, height=300)

    # Option chain data
    call_data = [Paragraph(f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", styles["Normal"]) for
                 index, row in calls_df.iterrows()]
    put_data = [Paragraph(f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", styles["Normal"]) for
                index, row in puts_df.iterrows()]

    # Build PDF content
    content = [title, current_price_text, Spacer(1, 12), chart_image, Spacer(1, 12)]
    content.append(Paragraph("<b>Call Option Data:</b>", styles["Normal"]))
    content.extend(call_data)
    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Put Option Data:</b>", styles["Normal"]))
    content.extend(put_data)

    # Add content to the PDF document
    doc.build(content)


# Endpoint to get top 15 stocks
@app.route('/stock', methods=['GET'])
def get_top_stocks():
    top_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                  "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS",
                  "ITC.NS", "BAJFINANCE.NS", "ADANIGREEN.NS", "ASIANPAINT.NS", "DMART.NS"]

    stock_data = []
    for ticker in top_stocks:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        stock_info = stock.info if not data.empty else {}
        stock_entry = {
            'ticker': ticker,
            'current_price': data['Close'].iloc[-1].item() if not data.empty else None,
            'open': data['Open'].iloc[-1].item() if not data.empty else None,
            'high': data['High'].iloc[-1].item() if not data.empty else None,
            'low': data['Low'].iloc[-1].item() if not data.empty else None,
            'volume': data['Volume'].iloc[-1].item() if not data.empty else None,
            'name': stock_info.get('longName', None),
            'sector': stock_info.get('sector', None),
            'industry': stock_info.get('industry', None),
            'website': stock_info.get('website', None),
            'market_cap': stock_info.get('marketCap', None),
            'dividend_yield': stock_info.get('dividendYield', None),
            'forward_PE': stock_info.get('forwardPE', None),
            'previous_close': stock_info.get('previousClose', None),
            '52_week_high': stock_info.get('fiftyTwoWeekHigh', None),
            '52_week_low': stock_info.get('fiftyTwoWeekLow', None),
            'beta': stock_info.get('beta', None),
            'eps': stock_info.get('eps', None),
            'pe_ratio': stock_info.get('peRatio', None),
            'short_name': stock_info.get('shortName', None),
            'long_business_summary': stock_info.get('longBusinessSummary', None),
            'logo_url': stock_info.get('logo_url', None)  # Adding the company's logo URL
        }

        stock_data.append(stock_entry)

    return jsonify(stock_data)


# @app.route('/option-chain', methods=['GET'])
# def option_chain():
#     symbol = request.args.get('symbol')
#     current_price = float(request.args.get('current_price'))
#
#     try:
#         calls_df, puts_df = get_nse_option_chain(symbol)
#         atm_strike, call_price, put_price = find_atm_straddle(calls_df, puts_df, current_price)
#
#         option_chain_img = plot_option_chain(calls_df, puts_df, current_price)
#         straddle_payoff_img = plot_straddle_payoff(atm_strike, call_price, put_price)
#
#         response = {
#             'ATM Strike Price': atm_strike,
#             'Call Option Price': call_price,
#             'Put Option Price': put_price,
#             'Long Straddle Cost': call_price + put_price,
#             'Short Straddle Premium': call_price + put_price,
#             'Option Chain Image': option_chain_img,
#             'Straddle Payoff Image': straddle_payoff_img
#         }
#         return jsonify(response)
#
#     except Exception as e:
#         return jsonify({'error': str(e)})

# Endpoint to get data for a specific stock with a given time span
@app.route('/stock/<ticker>/<time_span>', methods=['GET'])
def get_stock(ticker, time_span):
    # Get the time span from the query parameter
    # time_span = request.args.get('time_span', default='1d', type=str)

    # Validate the time span input
    valid_time_spans = ['1d', '1mo', '6mo', '1y']
    if time_span not in valid_time_spans:
        return jsonify({"error": "Invalid time span. Valid options are: " + ", ".join(valid_time_spans)}), 400

    stock = yf.Ticker(ticker)
    data = stock.history(period=time_span)

    stock_data = data.reset_index().to_dict(orient='records')

    return jsonify(stock_data)


# Endpoint to search for stocks by name
# Endpoint to search for stocks by name
# Function to calculate returns
# Function to calculate returns
# Function to calculate returns
def calculate_interest_rate(today_price, past_price, years):
    if today_price <= 0 or past_price <= 0:
        return None
    interest_rate = ((today_price / past_price) ** (1 / years)) - 1
    return interest_rate * 100


def get_gold_price(date):
    # Format the date
    date_str = date.strftime("%Y%m%d")
    # API endpoint
    url = f"https://www.goldapi.io/api/XAU/INR/{date_str}"
    headers = {
        "x-access-token": "goldapi-zirtvslwy9juge-io"
    }
    # Make the request
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('price')
    else:
        return None


@app.route('/gold-interest-rate', methods=['GET'])
def gold_interest_rate():
    # Get today's date
    today = datetime.now()
    # Get gold price for today
    today_price = get_gold_price(today)
    if today_price is None:
        return jsonify({'error': 'Failed to fetch today\'s gold price'}), 500
    # Get gold price for 5 years ago
    five_years_ago = today - datetime.timedelta(days=5 * 365)
    past_price = get_gold_price(five_years_ago)
    if past_price is None:
        return jsonify({'error': 'Failed to fetch gold price 5 years ago'}), 500
    # Calculate effective rate of interest
    interest_rate = calculate_interest_rate(today_price, past_price, 5)
    if interest_rate is None:
        return jsonify({'error': 'Failed to calculate interest rate'}), 500
    # Return the result
    return jsonify({'interest_rate': interest_rate})


def calculate_returns(data):
    nav_values = [float(d['nav']) for d in data]
    dates = [datetime.strptime(d['date'], '%d-%m-%Y') for d in data]

    # Calculate 1-year return
    one_year_ago = datetime.now() - timedelta(days=365)
    one_year_nav = None
    for date in sorted(filter(lambda x: x <= one_year_ago, dates), reverse=True):
        one_year_nav = nav_values[dates.index(date)]
        break

    if one_year_nav is None:
        one_year_return = "Data not available"
    else:
        one_year_return = ((nav_values[0] / one_year_nav) ** (1) - 1) * 100

    # Calculate 1-month return
    one_month_ago = datetime.now() - timedelta(days=30)
    one_month_nav = None
    for date in sorted(filter(lambda x: x <= one_month_ago, dates), reverse=True):
        one_month_nav = nav_values[dates.index(date)]
        break

    if one_month_nav is None:
        one_month_return = "Data not available"
    else:
        one_month_return = ((nav_values[0] / one_month_nav) ** (1 / 12) - 1) * 100

    # Calculate 3-year return
    three_years_ago = datetime.now() - timedelta(days=3 * 365)
    three_year_nav = None
    for date in sorted(filter(lambda x: x <= three_years_ago, dates), reverse=True):
        three_year_nav = nav_values[dates.index(date)]
        break

    if three_year_nav is None:
        three_year_return = "Data not available"
    else:
        three_year_return = ((nav_values[0] / three_year_nav) ** (1 / 3) - 1) * 100

    return {
        "1_year_return": one_year_return,
        "1_month_return": one_month_return,
        "3_year_return": three_year_return
    }


def get_nse_option_chain(symbol):
    base_url = "https://www.nseindia.com/"
    option_chain_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_url,
        "X-Requested-With": "XMLHttpRequest"
    }

    session = requests.Session()
    session.get(base_url, headers=headers)
    response = session.get(option_chain_url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        # Extract call and put data
        call_data = []
        put_data = []

        for record in data['records']['data']:
            if 'CE' in record:
                call_data.append(record['CE'])
            if 'PE' in record:
                put_data.append(record['PE'])

        # Convert to DataFrame
        calls_df = pd.DataFrame(call_data)
        puts_df = pd.DataFrame(put_data)

        return calls_df, puts_df
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")


def find_atm_straddle(calls_df, puts_df, current_price):
    # Identify the ATM strike price
    calls_df['strikePrice'] = calls_df['strikePrice'].astype(float)
    puts_df['strikePrice'] = puts_df['strikePrice'].astype(float)

    atm_call = calls_df.iloc[(calls_df['strikePrice'] - current_price).abs().argsort()[:1]]
    atm_put = puts_df.iloc[(puts_df['strikePrice'] - current_price).abs().argsort()[:1]]

    if not atm_call.empty and not atm_put.empty:
        atm_strike = atm_call.iloc[0]['strikePrice']
        call_price = atm_call.iloc[0]['lastPrice']
        put_price = atm_put.iloc[0]['lastPrice']

        return atm_strike, call_price, put_price
    else:
        raise Exception("ATM options not found")


def plot_option_chain(calls_df, puts_df, current_price):
    plt.figure(figsize=(14, 7))

    # Plot call prices
    plt.plot(calls_df['strikePrice'], calls_df['lastPrice'], 'b-', label='Call Prices')

    # Plot put prices
    plt.plot(puts_df['strikePrice'], puts_df['lastPrice'], 'r-', label='Put Prices')

    # Mark the current stock price
    plt.axvline(x=current_price, color='g', linestyle='--', label='Current Stock Price')

    # Highlight areas below and above the current price
    plt.fill_between(calls_df['strikePrice'], calls_df['lastPrice'], where=(calls_df['strikePrice'] < current_price),
                     interpolate=True, color='blue', alpha=0.3)
    plt.fill_between(calls_df['strikePrice'], calls_df['lastPrice'], where=(calls_df['strikePrice'] > current_price),
                     interpolate=True, color='lightblue', alpha=0.3)
    plt.fill_between(puts_df['strikePrice'], puts_df['lastPrice'], where=(puts_df['strikePrice'] < current_price),
                     interpolate=True, color='red', alpha=0.3)
    plt.fill_between(puts_df['strikePrice'], puts_df['lastPrice'], where=(puts_df['strikePrice'] > current_price),
                     interpolate=True, color='lightcoral', alpha=0.3)

    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Option Chain Data')
    plt.legend()
    plt.grid(True)

    # Save plot to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(temp_file.name)
    temp_file.seek(0)
    plt.close()
    return temp_file


@app.route('/stradle', methods=['GET'])
def option_chain_stradle():
    symbol = request.args.get('symbol')
    current_price = float(request.args.get('current_price'))
    straddle_type = request.args.get('straddle_type', 'long')

    try:
        calls_df, puts_df = get_nse_option_chain(symbol)
        atm_strike, call_price, put_price = find_atm_straddle(calls_df, puts_df, current_price)

        option_chain_file = plot_option_chain(calls_df, puts_df, current_price)
        output_path = os.path.join('pdf_reports', f"{symbol}_option_chain_report.pdf")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        create_pdf_with_chart(symbol, current_price, option_chain_file, output_path, calls_df, puts_df, atm_strike,
                              call_price, put_price, straddle_type)

        download_link = f"/pdf_reports/{symbol}_option_chain_report.pdf"
        return jsonify({'download_link': download_link})

    except Exception as e:
        return jsonify({'error': str(e)})


largecap = ["106235", "108466", "102273", "101349"]
midcap = ["101065", "104513", "127042", "100375", "104908"]
smallcap = ["113177", "108466", "145205", "146127"]


def get_nse_option_chain(symbol):
    base_url = "https://www.nseindia.com/"
    option_chain_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_url,
        "X-Requested-With": "XMLHttpRequest"
    }

    session = requests.Session()
    session.get(base_url, headers=headers)
    response = session.get(option_chain_url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        # Extract call and put data
        call_data = []
        put_data = []

        for record in data['records']['data']:
            if 'CE' in record:
                call_data.append(record['CE'])
            if 'PE' in record:
                put_data.append(record['PE'])

        # Convert to DataFrame
        calls_df = pd.DataFrame(call_data)
        puts_df = pd.DataFrame(put_data)

        return calls_df, puts_df
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")


@app.route('/option_chain_strangle', methods=['GET'])
def option_chain_strangle():
    symbol = request.args.get('symbol')
    current_price = float(request.args.get('current_price'))

    try:
        calls_df, puts_df = get_nse_option_chain(symbol)
        option_chain_file = plot_option_chain(calls_df, puts_df, current_price)
        output_path = os.path.join('pdf_reports', f"{symbol}_option_chain_report.pdf")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        create_pdf_with_chart(symbol, current_price, option_chain_file, output_path, calls_df, puts_df)

        download_link = f"/pdf_reports/{symbol}_option_chain_report.pdf"
        return jsonify({'download_link': download_link})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/option_chain_straddle', methods=['GET'])
def option_chain_straddle():
    symbol = request.args.get('symbol')
    current_price = float(request.args.get('current_price'))

    try:
        calls_df, puts_df = get_nse_option_chain(symbol)

        # Find the nearest at-the-money call and put options
        atm_call = calls_df.iloc[(calls_df['strike_price'] - current_price).abs().argsort()[:1]]
        atm_put = puts_df.iloc[(puts_df['strike_price'] - current_price).abs().argsort()[:1]]

        # Plot the straddle
        straddle_option_chain_file = plot_option_chain(atm_call, atm_put, current_price)
        output_path = os.path.join('pdf_reports', f"{symbol}_straddle_option_chain_report.pdf")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        create_pdf_with_chart(symbol, current_price, straddle_option_chain_file, output_path, calls_df, puts_df)

        download_link = f"/pdf_reports/{symbol}_straddle_option_chain_report.pdf"
        return jsonify({'download_link': download_link})

    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/pdf_reports/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(directory='pdf_reports', filename=filename, as_attachment=True)


def get_data(number):
    try:
        api_url = "https://api.mfapi.in/mf/" + number
        response = requests.get(api_url)
        response.raise_for_status()  # Raise exception for 4xx or 5xx status codes
        data = response.json()
        returns_data = calculate_returns(data['data'])

        api_url2 = f"https://api.mfapi.in/mf/{number}/latest"
        response2 = requests.get(api_url2)
        response2.raise_for_status()  # Raise exception for 4xx or 5xx status codes
        latest_data = response2.json()

        latest_data["returns"] = (returns_data)
        return latest_data
    except requests.exceptions.RequestException as e:
        return {"error": "Failed to fetch data from the API. Error: {}".format(str(e))}
    except ValueError as e:
        return {"error": "Failed to parse JSON response from the API. Error: {}".format(str(e))}


@app.route('/mf')
def get_allmf():
    datal = []
    datam = []
    datas = []
    for i in largecap:
        datal.append(get_data(i))
    for i in midcap:
        datam.append(get_data(i))
    for i in smallcap:
        datas.append(get_data(i))
    data = {"large": datal, "midcap": datam, "smallcap": datam}
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True,port=5001)
