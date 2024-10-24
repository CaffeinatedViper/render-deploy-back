from uuid import UUID
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from DataStore import DataStore
from DataPreprocessor.correlation_analyzer import CorrelationAnalyzer
from DataPreprocessor.benford_analyzer import BenfordAnalyzer
from DataPreprocessor.anomaly_detector import AnomalyDetector
import os
app = Flask("Benford Analyzer")
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # dla lokalnego developmentu
            "https://benford-analyzer-frontend.onrender.com"  # dla produkcji
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
datastore = DataStore()
correlation_analyzer = CorrelationAnalyzer()


# do testowania na szybko
@app.route("/", methods=['GET'])
def index():
    initialize_server()
    return "<p>Benford server up and running</p>"

# Przykladowy "porzadny" endpoint
@app.route("/data/upload", methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return {"reason": "No file selected"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"reason": "File not named"}, 400
    try:
        id = datastore.store_file(file)
    except Exception as e:
        print(e)
        return {"reason": "Could not upload data"}, 500
    return {"id": str(id)}, 200

@app.route('/data/benford/analyze', methods=['POST'])
def benford_analyze():
    id = request.json.get('id')
    column = request.json.get('column')
    try:
        df = datastore.get_dataset(id)
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404
        if column not in df.columns:
            return jsonify({'error': f"Column '{column}' not found in dataset."}), 400

        benford_analyzer = BenfordAnalyzer()
        results = benford_analyzer.analyze(df, column)

 
        image_base64 = benford_analyzer.plot_distribution(
            results['empirical_probs'],
            results['benford_probs'],
            title=f"Analiza Prawa Benforda dla kolumny '{column}'"
        )

        response = {
            'empirical_probs': results['empirical_probs'].to_dict(),
            'benford_probs': results['benford_probs'].to_dict(),
            'chi_stat': results['test_results']['chi_stat'],
            'p_value': results['test_results']['p_value'],
            'plot': image_base64 
        }
        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:

        print("Exception in benford_analyze:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
    
    
@app.route('/data/anomaly/detect', methods=['POST'])
def detect_anomalies():
    id = request.json.get('id')
    contamination = request.json.get('contamination', 0.05) 
    try:
        df = datastore.get_dataset(id)
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404

  
        identifier_columns = ['TransactionID', 'CustomerID']
        df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[col for col in identifier_columns if col in df.columns])


        if df_numeric.empty:
            return jsonify({'error': 'No numeric data available for anomaly detection after removing identifier columns.'}), 400

        anomaly_detector = AnomalyDetector(contamination=contamination)
        anomaly_detector.fit(df_numeric)

        predictions = anomaly_detector.predict(df_numeric)
        anomaly_scores = anomaly_detector.get_anomaly_scores(df_numeric)


        df_results = df.copy()
        df_results['Anomaly'] = predictions
        df_results['Anomaly_Score'] = anomaly_scores

        anomalies = df_results[df_results['Anomaly'] == -1]

        response = {
            'anomalies': anomalies.to_dict(orient='records'),
            'anomaly_count': len(anomalies),
            'total_count': len(df_numeric),
        }
        return jsonify(response), 200

    except Exception as e:
        print("Exception in detect_anomalies:", e)
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/data/correlation/preprocess', methods=['POST'])
def preprocess_correlation():
    id = request.json.get('id')
    threshold = float(request.json.get('threshold'))
    method = request.json.get('method')
    selection_method = request.json.get('selection_method')
    try:
        df = datastore.get_dataset(id)
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404
        if not correlation_analyzer.validate_settings(method,threshold,selection_method):
            return jsonify({'error': 'invalid settings'}), 400
        result_df = correlation_analyzer.process(df,threshold,method,selection_method)
        id = datastore.store_df(result_df)

        return {"id": str(id)}, 200
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/data/correlation/preview_preprocess', methods=['POST'])
def preview_features_to_remove():
    id = request.json.get('id')
    threshold = float(request.json.get('threshold'))
    method = request.json.get('method')
    selection_method = request.json.get('selection_method')
    print(threshold,method,selection_method)
    try:
        df = datastore.get_dataset(id)
        if df is None:
            return jsonify({'error': 'Dataset not found'}), 404
        if not correlation_analyzer.validate_settings(method,threshold,selection_method):
            return jsonify({'error': 'invalid settings'}), 400
        columns_to_remove = correlation_analyzer.columns_to_remove(df,threshold,method,selection_method)
        return jsonify({'columns_to_remove': columns_to_remove}), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route("/data/visualize", methods=['POST'])
def visualize_data():
    id = request.json.get('id')
    if id is None:
        return {"reason": "no ID"}, 401
    try:
        rowAmt, cols, head, tail = datastore.visualize(id)
    except Exception:
        return {"reason": "Could not fetch data"}, 500
    print(head[0])
    return jsonify(length=rowAmt, columns=cols, head=head, tail=tail), 200


# Jeżeli chcemy coś na szybko przetestować/podpiąć, tutaj to dobre miejsce.
# Funkcja odpali się w momencie w którym wejdziemy na http://127.0.0.1:5000
server_inititialized = False


def initialize_server():
    global server_inititialized
    if server_inititialized: return
    server_inititialized = True
    print("Inicjalizacja serwera")
    # Tu Twój kod ...
    print("Koniec inicjalizacji serwera")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)