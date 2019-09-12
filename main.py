#!/usr/local/bin python3
from flask import Flask, render_template, request, jsonify
from app_function import calculate_health_score, add_rul_prediction, add_response_identity

app = Flask(__name__)

@app.route('/health_score', methods = ['POST'])
def handle_request_healthscore():
    if request.method == 'POST':
        data = request.get_json(force=True)
        health_score = calculate_health_score(data)
        health_score_with_rul = add_rul_prediction(health_score)
        response = add_response_identity(health_score_with_rul)
        return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
    