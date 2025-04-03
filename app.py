from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from werkzeug.utils import secure_filename
from utils.db import initialize_db, get_db_connection
from utils.auth import login_required
from utils.predict import predict_blood_group
from utils import initialize_db, get_db_connection, login_required, predict_blood_group
import io
import base64
from PIL import Image
import cv2
import tempfile
import pdfkit

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','bmp'}

# Load the model
model = tf.keras.models.load_model('models/fingerprint_blood_group_model.h5')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
initialize_db()

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and user['password'] == password:  # In production, use proper password hashing
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if username already exists
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            flash('Username already exists')
            return render_template('register.html')
        
        # Insert new user
        cursor.execute(
            'INSERT INTO users (username, password, email) VALUES (%s, %s, %s)',
            (username, password, email)  # In production, hash the password
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT id, name, age, gender, DATE_FORMAT(created_at, '%%Y-%%m-%%d') as date 
        FROM patients 
        ORDER BY created_at DESC
    ''')
    patients = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('dashboard.html', patients=patients)

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        weight = request.form['weight']
        gender = request.form['gender']
        address = request.form['address']
        phone = request.form['phone']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (name, age, weight, gender, address, phone, user_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (name, age, weight, gender, address, phone, session['user_id']))
        patient_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        
        return redirect(url_for('fingerprint', patient_id=patient_id))
    
    return render_template('add_patient.html')

@app.route('/patient/<int:patient_id>/fingerprint', methods=['GET', 'POST'])
@login_required
def fingerprint(patient_id):
    if request.method == 'POST':
        # Check if the user uploaded a file
        if 'fingerprint' in request.files:
            file = request.files['fingerprint']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Predict blood group
                blood_group = predict_blood_group(model, filepath)
                
                # Save prediction to database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE patients 
                    SET blood_group = %s, fingerprint_path = %s 
                    WHERE id = %s
                ''', (blood_group, filename, patient_id))
                conn.commit()
                cursor.close()
                conn.close()
                
                return redirect(url_for('results', patient_id=patient_id))
        
        # Check if image was captured from webcam
        elif 'imageData' in request.form:
            image_data = request.form['imageData'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save the captured image
            filename = secure_filename(f"{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            
            # Predict blood group
            blood_group = predict_blood_group(model, filepath)
            
            # Save prediction to database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE patients 
                SET blood_group = %s, fingerprint_path = %s 
                WHERE id = %s
            ''', (blood_group, filename, patient_id))
            conn.commit()
            cursor.close()
            conn.close()
            
            return redirect(url_for('results', patient_id=patient_id))
    
    return render_template('fingerprint.html', patient_id=patient_id)

@app.route('/patient/<int:patient_id>/results')
@login_required
def results(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM patients WHERE id = %s', (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not patient:
        flash('Patient not found')
        return redirect(url_for('dashboard'))
    
    return render_template('results.html', patient=patient)

@app.route('/patient/<int:patient_id>/report')
@login_required
def report(patient_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM patients WHERE id = %s', (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not patient:
        flash('Patient not found')
        return redirect(url_for('dashboard'))
    
    # Format the date correctly
    if 'created_at' in patient and patient['created_at']:
        patient['created_at'] = patient['created_at'].strftime('%Y-%m-%d')
    
    return render_template('report.html', patient=patient)

@app.route('/patient/<int:patient_id>/generate_pdf')
@login_required
def generate_pdf(patient_id):
    # Get patient data
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM patients WHERE id = %s', (patient_id,))
    patient = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not patient:
        flash('Patient not found')
        return redirect(url_for('dashboard'))
    
    # Format the date correctly
    if 'created_at' in patient and patient['created_at']:
        patient['created_at'] = patient['created_at'].strftime('%Y-%m-%d')
    
    # Generate HTML content
    html_content = render_template('report.html', patient=patient, pdf_mode=True)

    # Set up wkhtmltopdf configuration
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")


    # Create PDF
    pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    # pdfkit.from_string(html_content, pdf_file.name)
    # pdfkit.from_string(html_content, pdf_file.name, configuration=config)
    options = {
    'enable-local-file-access': None  # Enables access to local files
        }
    pdfkit.from_string(html_content, pdf_file.name, options=options, configuration=config)

    
    # Send the file
    return send_file(
        pdf_file.name,
        as_attachment=True,
        download_name=f"patient_{patient_id}_report.pdf"
    )

if __name__ == '__main__':
    app.run(debug=True)
