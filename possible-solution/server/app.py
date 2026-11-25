# Create a base Flask server

import pickle
import csv
import json
import os
import urllib.request
import urllib.error
from functools import lru_cache
from datetime import datetime, timezone
from flask import Flask, request, jsonify

app = Flask(__name__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model from pickle file
model = pickle.load(open('model.pkl', 'rb'))


@lru_cache(maxsize=1024)
def predict_delay(day_of_week, airport_id):
    prediction = model.predict_proba([[day_of_week, airport_id]])[0]
    prediction_text = str(prediction).split(' ')
    certainty = float(prediction_text[0][2:])
    delay = float(prediction_text[1][:-1])
    return {'certainty': certainty, 'delay': delay}


# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    # Store day_of_week as int
    day_of_week = int(request.args.get('day_of_week'))
    airport_id = int(request.args.get('airport_id'))
    return jsonify(predict_delay(day_of_week, airport_id))

@lru_cache(maxsize=1)
def load_airports():
    with open('airports.csv', 'r', newline='', encoding='utf-8') as airport_file:
        airports = [
            {'id': int(row['OriginAirportID']), 'name': row['OriginAirportName']}
            for row in csv.DictReader(airport_file)
        ]
    return sorted(airports, key=lambda airport: airport['name'])


# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    response = jsonify(load_airports())
    response.headers['Cache-Control'] = 'public, max-age=300'
    return response


def _chat_reply(message):
    text = message.strip()
    if not text:
        return "Ask me about flight delays, airport choices, or how to use this app."

    lowered = text.lower()
    if any(word in lowered for word in ("password", "secret", "api key", "token", "credential")):
        return "I can't help expose or transmit passwords or credentials. Use environment variables and keep secrets out of source control."
    if any(word in lowered for word in ("hello", "hi ", "hey")):
        return "Hello! I can help you understand flight-delay predictions or choose a day and airport."
    if "airport" in lowered:
        return "Choose an airport from the dropdown, then select a weekday. The prediction compares that day and airport with the trained flight-delay model."
    if any(word in lowered for word in ("delay", "flight", "predict", "chance")):
        return "Use the form to pick a destination airport and day of the week. The result reports the estimated chance of a delay and the model's confidence."
    if any(word in lowered for word in ("how", "help", "use", "work")):
        return "Select Monday through Sunday and one of the available airports, then select Find delay."

    return "I'm not sure about that. Try asking about flight delays, airports, weekdays, or how to use the prediction form."


def _anthropic_reply(message):
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        return None

    model = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-5')
    payload = json.dumps({
        'model': model,
        'max_tokens': 500,
        'system': (
            'You assist with this flight-delay workshop app and its clearly labeled '
            'paper-trading demo. Never execute real trades or provide brokerage '
            'credentials. Keep answers concise and avoid financial guarantees.'
        ),
        'messages': [{'role': 'user', 'content': message[:2000]}],
    }).encode('utf-8')

    request_object = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=payload,
        headers={
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(request_object, timeout=15) as response:
            result = json.loads(response.read().decode('utf-8'))
        return ''.join(
            block.get('text', '')
            for block in result.get('content', [])
            if block.get('type') == 'text'
        ).strip()
    except urllib.error.HTTPError as error:
        app.logger.error('Anthropic API request failed: HTTP %s', error.code)
        error.read()
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        app.logger.error('Anthropic API request failed')
        return None


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get('message', '')
    if not isinstance(message, str):
        return jsonify({'error': 'message must be a string'}), 400
    message = message[:2000]
    reply = _chat_reply(message)

    lowered = message.lower()
    contains_secret_request = any(
        word in lowered
        for word in ("password", "secret", "api key", "token", "credential")
    )
    if os.environ.get('ANTHROPIC_API_KEY', '').strip() and not contains_secret_request:
        anthropic_reply = _anthropic_reply(message)
        if anthropic_reply:
            reply = anthropic_reply

    return jsonify({'reply': reply})


PAPER_PORTFOLIO = {
    'cash': 10000.0,
    'holdings': {},
    'orders': [],
}


DEMO_PRICES = {
    'AAPL': 227.16,
    'MSFT': 505.27,
    'GOOG': 236.49,
    'AMZN': 223.30,
}


@app.route('/paper-trading/signal', methods=['GET'])
def paper_trading_signal():
    symbol = str(request.args.get('symbol', 'AAPL')).upper()
    if symbol not in DEMO_PRICES:
        return jsonify({'error': 'Unsupported demo symbol'}), 400

    price = DEMO_PRICES[symbol]
    return jsonify({
        'symbol': symbol,
        'price': price,
        'signal': 'HOLD',
        'confidence': 0.55,
        'mode': 'DEMO',
        'disclaimer': 'Educational simulation only. Not investment advice.',
    })


@app.route('/paper-trading/order', methods=['POST'])
def paper_trading_order():
    data = request.get_json(silent=True) or {}
    symbol = str(data.get('symbol', '')).upper()
    side = str(data.get('side', '')).lower()

    try:
        quantity = float(data.get('quantity', 0))
    except (TypeError, ValueError):
        quantity = 0

    if symbol not in DEMO_PRICES:
        return jsonify({'error': 'Unsupported demo symbol'}), 400
    if side not in ('buy', 'sell'):
        return jsonify({'error': 'Side must be buy or sell'}), 400
    if quantity <= 0:
        return jsonify({'error': 'Quantity must be greater than zero'}), 400

    price = DEMO_PRICES[symbol]
    total = round(price * quantity, 2)
    holdings = PAPER_PORTFOLIO['holdings'].get(symbol, 0)

    if side == 'buy':
        if total > PAPER_PORTFOLIO['cash']:
            return jsonify({'error': 'Insufficient virtual cash'}), 400
        PAPER_PORTFOLIO['cash'] = round(PAPER_PORTFOLIO['cash'] - total, 2)
        PAPER_PORTFOLIO['holdings'][symbol] = holdings + quantity
    else:
        if quantity > holdings:
            return jsonify({'error': 'Insufficient virtual shares'}), 400
        remaining = holdings - quantity
        if remaining:
            PAPER_PORTFOLIO['holdings'][symbol] = remaining
        else:
            PAPER_PORTFOLIO['holdings'].pop(symbol, None)
        PAPER_PORTFOLIO['cash'] = round(PAPER_PORTFOLIO['cash'] + total, 2)

    order = {
        'symbol': symbol,
        'side': side,
        'quantity': quantity,
        'price': price,
        'total': total,
        'mode': 'PAPER',
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    PAPER_PORTFOLIO['orders'].insert(0, order)
    return jsonify({'order': order, 'portfolio': PAPER_PORTFOLIO})


@app.route('/paper-trading/portfolio', methods=['GET'])
def paper_trading_portfolio():
    return jsonify(PAPER_PORTFOLIO)

if __name__ == '__main__':
    app.run(debug=True)