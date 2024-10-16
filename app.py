from flask import Flask, render_template, request, jsonify
from etta_core import initialize_conversation_bot, update_index, chat_with_bot
from threading import Thread
import time
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'Data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

query_engine = initialize_conversation_bot()

@app.route('/')
def home():
    return render_template('index.html', timestamp=int(time.time()))

@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']
        response_stream = chat_with_bot(query_engine, question)

        def generate():
            for token in response_stream.response_gen:
                yield token

        return app.response_class(generate(), mimetype='text/html')
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        update_index()
        return jsonify({'message': f'{file.filename} uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
