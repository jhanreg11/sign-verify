from server import app
from flask import request, jsonify
from server.model.preprocessor import prepare, create_user, load_model, add_training_data
import os, torch

path = os.getcwd()


@app.route('/api/verify', methods=['GET'])
def get_verification():
    data = request.get_json(force=True)
    signature = data['signature']
    username = data['username']
    res = load_model().forward(prepare(signature, False).view(1, 1, 220, 155), prepare(username).view(1, 1, 220, 155), True)
    return jsonify({'success': res})

@app.route('/api/create', methods=['POST'])
def post_signatures():
    data = request.get_json(force=True)
    res = create_user(data['username'], data['signatures'])
    return jsonify({'success': res})

@app.route('/api/training-data', methods=['POST'])
def post_training_data():
    data = request.get_json()
    username = data['username']
    signature = data['signature']
    classification = data['class']
    add_training_data(username, signature, classification)

@app.route('/api/health-check', methods=['GET'])
def health_check():
    return "I'm alive"