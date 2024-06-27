import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from flask import Flask, request, jsonify, send_from_directory
import requests
import seaborn as sns
from fpdf import FPDF

app = Flask(__name__)

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

def plot_option_chain(calls_df, puts_df, current_price):
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=calls_df['strikePrice'], y=calls_df['lastPrice'], color='blue', label='Call Prices', marker='o')
    sns.lineplot(x=puts_df['strikePrice'], y=puts_df['lastPrice'], color='red', label='Put Prices', marker='o')
    plt.axvline(current_price, color='green', linestyle='--', label='Current Price')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title('Option Chain')
    plt.legend()

    # Save the plot to a file
    output_path = os.path.join('charts', 'option_chain_chart.png')
    plt.savefig(output_path)
    plt.close()

    return output_path

def create_pdf_with_chart(symbol, current_price, option_chain_file, output_path, calls_df, puts_df):
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set title
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Option Chain Report for {symbol}", ln=True, align='C')

    # Set subtitle
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Current Price: {current_price}", ln=True, align='C')

    # Add chart image
    pdf.image(option_chain_file, x=10, y=30, w=180)

    # Move to the next line
    pdf.ln(80)

    # Add option chain data
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Call Option Data:", ln=True, align='L')

    pdf.set_font("Arial", size=10)
    for index, row in calls_df.iterrows():
        pdf.cell(200, 10, txt=f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", ln=True, align='L')

    # Move to the next line
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Put Option Data:", ln=True, align='L')

    pdf.set_font("Arial", size=10)
    for index, row in puts_df.iterrows():
        pdf.cell(200, 10, txt=f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", ln=True, align='L')

    # Save PDF to specified path
    pdf.output(output_path)
from fpdf import FPDF
import pandas as pd
import os

def create_pdf_with_chart1(symbol, current_price, option_chain_file, output_path, calls_df, puts_df, atm_strike, call_price, put_price, straddle_type):
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set title
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(200, 10, txt=f"Option Chain Report for {symbol}", ln=True, align='C')

    # Set subtitle
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Current Price: {current_price}", ln=True, align='C')

    # Add chart image
    pdf.image(option_chain_file, x=10, y=30, w=180)

    # Move to the next line
    pdf.ln(95)  # Adjusted to move after the image

    # Add ATM Straddle Information
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt=f"ATM Strike Price: {atm_strike}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"ATM Call Price: {call_price}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"ATM Put Price: {put_price}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Straddle Type: {straddle_type}", ln=True, align='L')

    # Add Call Option Data
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Call Option Data:", ln=True, align='L')

    pdf.set_font("Arial", size=10)
    for index, row in calls_df.iterrows():
        pdf.cell(200, 10, txt=f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", ln=True, align='L')

    # Move to the next line
    pdf.ln(10)

    # Add Put Option Data
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Put Option Data:", ln=True, align='L')

    pdf.set_font("Arial", size=10)
    for index, row in puts_df.iterrows():
        pdf.cell(200, 10, txt=f"Strike: {row['strikePrice']}, Last Price: {row['lastPrice']}", ln=True, align='L')

    # Save PDF to specified path
    pdf.output(output_path)


@app.route('/long_straddle', methods=['GET'])
def option_chain():
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

        create_pdf_with_chart1(symbol, current_price, option_chain_file, output_path, calls_df, puts_df, atm_strike, call_price, put_price, straddle_type)

        download_link = f"/pdf_reports/{symbol}_option_chain_report.pdf"
        return jsonify({'download_link': download_link})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/pdf_reports/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(directory='pdf_reports', filename=filename, as_attachment=True)
@app.route('/long_stringle', methods=['GET'])
def option_chain_stringle():
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
if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('pdf_reports', exist_ok=True)
    os.makedirs('charts', exist_ok=True)

    app.run(debug=True)
