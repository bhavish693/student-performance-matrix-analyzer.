"""
=============================================================================
Student Performance Matrix Analyzer - Flask Backend Application
=============================================================================
Course: MAT2303 - Linear Algebra & Vector Calculus
Architecture: Flask Web Application with REST Endpoints
=============================================================================
"""

import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# Import Mathematical Engine and PDF Generator
try:
    from math_engine import MatrixAnalyzer, generate_sample_data
except ImportError as err:
    logging.error("Failed to import math_engine: %s", err)
    raise

try:
    from pdf_generator import generate_report
except ImportError:
    generate_report = None
    logging.warning("pdf_generator not found or could not be imported; PDF export will be disabled until available.")

# Setup Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')


def make_json_serializable(obj):
    """
    Recursively convert NumPy data types to native Python types
    to guarantee flawless JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj


@app.route('/', methods=['GET'])
def index():
    """Render the primary single-page user interface."""
    return render_template('index.html')


@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """
    Return realistic, correlated student performance sample data
    generated via math_engine.generate_sample_data.
    """
    try:
        data, subjects, student_names = generate_sample_data(n_students=10, n_subjects=6, seed=42)
        return jsonify({
            'success': True,
            'matrix': data.tolist(),
            'subjects': subjects,
            'students': student_names,
            'shape': list(data.shape)
        }), 200
    except Exception as e:
        app.logger.exception("Error generating sample data: %s", e)
        return jsonify({
            'success': False,
            'error': f"Failed to generate sample data: {str(e)}"
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Accept student performance data matrix, execute all 18 linear
    algebra analysis stages via MatrixAnalyzer, and return full results.
    """
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({'success': False, 'error': "Invalid request: No JSON payload received"}), 400

        # Retrieve matrix data from various standard payload shapes
        raw_matrix = payload.get('matrix') or payload.get('data')
        subjects = payload.get('subjects')
        students = payload.get('students')

        if raw_matrix is None or len(raw_matrix) == 0:
            return jsonify({'success': False, 'error': "Data matrix cannot be empty"}), 400

        # Convert to numpy float array
        try:
            A = np.array(raw_matrix, dtype=float)
        except (ValueError, TypeError) as e:
            return jsonify({'success': False, 'error': f"Matrix must contain only valid numeric scores: {str(e)}"}), 400

        if A.ndim != 2:
            return jsonify({'success': False, 'error': "Input matrix must be 2-dimensional (Students x Subjects)"}), 400

        m, n = A.shape
        if m < 2:
            return jsonify({'success': False, 'error': "At least 2 student rows are required for meaningful linear algebra analysis"}), 400
        if n < 2:
            return jsonify({'success': False, 'error': "At least 2 subject columns are required for multidimensional analysis"}), 400

        # Validate score bounds [0, 100]
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            return jsonify({'success': False, 'error': "Scores must not contain NaN or infinite values"}), 400

        if np.any(A < 0) or np.any(A > 100):
            return jsonify({'success': False, 'error': "All subject scores must be within the valid academic range [0, 100]"}), 400

        # Defaults for metadata if not supplied
        if not subjects or len(subjects) != n:
            default_subjects = ['Mathematics', 'Physics', 'Chemistry', 'English', 'Computer Science', 'Statistics']
            subjects = [default_subjects[i] if i < len(default_subjects) else f"Subject {i+1}" for i in range(n)]

        if not students or len(students) != m:
            students = [f"Student {i+1}" for i in range(m)]

        # Run complete 18-stage analysis
        analyzer = MatrixAnalyzer(A)
        results = analyzer.run_full_analysis()

        # Sanitize results for JSON
        serializable_results = make_json_serializable(results)

        return jsonify({
            'success': True,
            'students': students,
            'subjects': subjects,
            'shape': [m, n],
            'results': serializable_results
        }), 200

    except Exception as e:
        app.logger.exception("Error in /analyze: %s", e)
        return jsonify({
            'success': False,
            'error': f"Mathematical computation error: {str(e)}"
        }), 500


@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    """
    Generate and serve a comprehensive academic PDF report
    summarizing all 18 linear algebra analysis stages.
    """
    if generate_report is None:
        return jsonify({
            'success': False,
            'error': "PDF generation engine is currently unavailable on the server"
        }), 503

    try:
        payload = request.get_json(silent=True) or {}
        raw_matrix = payload.get('matrix') or payload.get('data')
        subjects = payload.get('subjects')
        students = payload.get('students')
        cached_results = payload.get('results')

        if raw_matrix is None:
            # Fall back to default sample data if no matrix was submitted
            A, subjects, students = generate_sample_data(n_students=10, n_subjects=6, seed=42)
        else:
            A = np.array(raw_matrix, dtype=float)
            if subjects is None:
                default_subjects = ['Mathematics', 'Physics', 'Chemistry', 'English', 'Computer Science', 'Statistics']
                subjects = [default_subjects[i] if i < len(default_subjects) else f"Subject {i+1}" for i in range(A.shape[1])]
            if students is None:
                students = [f"Student {i+1}" for i in range(A.shape[0])]

        # If analysis results were not already provided, compute them on the fly
        if cached_results is None:
            analyzer = MatrixAnalyzer(A)
            cached_results = analyzer.run_full_analysis()

        # Generate the PDF file
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, 'Student_Performance_Matrix_Analysis_Report.pdf')

        generate_report(
            results=cached_results,
            data=A,
            subjects=subjects,
            student_names=students,
            output_path=pdf_path
        )

        if not os.path.exists(pdf_path):
            return jsonify({'success': False, 'error': "PDF file could not be generated on disk"}), 500

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="Student_Performance_Matrix_Analysis_Report.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        app.logger.exception("Error generating PDF: %s", e)
        return jsonify({
            'success': False,
            'error': f"PDF generation error: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Verify module imports upon startup
    print("[OK] Starting Student Performance Matrix Analyzer...")
    print("[OK] Routes registered: / [GET], /sample-data [GET], /analyze [POST], /generate-pdf [POST]")
    app.run(host='0.0.0.0', port=5000, debug=True)
