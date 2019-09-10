#!/usr/local/bin python3
from flask import Flask, render_template, request, jsonify
from app_function import calculate_health_score, add_response_identity

app = Flask(__name__)

@app.route('/health_score', methods = ['POST'])
def handle_request_healthscore():
    if request.method == 'POST':
        response = calculate_health_score(request)
        response = add_response_identity(response)
        # use function jsonify to ensure response is returned in compatible json format
        return jsonify(response)

@app.route('/predict_rul', methods = ['POST'])
def handle_request_rul():
    if request.method == 'POST':
        response = calculate_health_score(request)
        response = add_response_identity(response)
        # use function jsonify to ensure response is returned in compatible json format
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
    