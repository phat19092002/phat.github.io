# Project: Nén file ảnh/PDF bằng SVD, lưu trữ file đã nén trên web, phân quyền theo user
# Framework: Flask + SQLite
# Tác giả: Trinh

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import secrets
import numpy as np
from PIL import Image
from numpy.linalg import svd
from fpdf import FPDF
import fitz  # PyMuPDF
import cv2
import zipfile
import sqlite3
from datetime import datetime
import smtplib
from email.message import EmailMessage

EMAIL_ADDRESS = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_app_password'

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

DB_PATH = os.path.join(BASE_DIR, 'files.db')
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff','webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def insert_file_metadata(filename, owner_email, file_type, k_value, size_before, size_after):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO files (filename, owner_email, file_type, k_value, size_before, size_after)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (filename, owner_email, file_type, k_value, size_before, size_after))
    file_id = c.lastrowid
    # Cấp quyền owner
    c.execute('INSERT INTO file_access (file_id, user_email) VALUES (?, ?)', (file_id, owner_email))
    conn.commit()
    conn.close()

def compress_svd_color(image, k):
    compressed_channels = []
    for i in range(3):
        U, S, V = np.linalg.svd(image[:, :, i], full_matrices=False)
        compressed = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
        compressed_channels.append(compressed)
    return np.stack(compressed_channels, axis=-1)

def compress_image_file_with_svd(input_path, output_path, k):
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("File ảnh không hợp lệ")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    compressed = compress_svd_color(image, k)
    compressed_uint8 = (np.clip(compressed, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(compressed_uint8, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70])

def compress_image_svd(image, k):
    img_gray = image.convert('L')
    A = np.array(img_gray, dtype=float)
    U, S, Vt = svd(A, full_matrices=False)
    S_k = np.diag(S[:k])
    A_k = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
    A_k = np.clip(A_k, 0, 255)
    return Image.fromarray(A_k.astype('uint8'))

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csGRAY)
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def extract_first_page_as_image(pdf_path, output_image_path):
    doc = fitz.open(pdf_path)
    if len(doc) > 0:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_image_path, format="JPEG", quality=70)

def images_to_pdf(images, output_path):
    pdf = FPDF(unit="mm", format="A4")
    for i, img in enumerate(images, start=1):
        img_path = f"temp_page_{i}.jpg"
        img.save(img_path, format="JPEG", quality=70)
        w_px, h_px = img.size
        w_mm, h_mm = w_px * 0.2646, h_px * 0.2646
        scale = min(210 / w_mm, 297 / h_mm)
        w_final = w_mm * scale
        h_final = h_mm * scale
        x = (210 - w_final) / 2
        y = (297 - h_final) / 2
        pdf.add_page()
        pdf.image(img_path, x=x, y=y, w=w_final, h=h_final)
        os.remove(img_path)
    pdf.output(output_path)

def compress_pdf_with_svd(input_pdf, output_pdf, k):
    images = pdf_to_images(input_pdf)
    compressed = [compress_image_svd(img, k) for img in images]
    images_to_pdf(compressed, output_pdf)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    if email and password:
        session['user'] = {'email': email}
        return redirect(url_for('upload'))
    return render_template('login.html', error='Vui lòng nhập email và mật khẩu')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        files = request.files.getlist('file')
        k_image = request.form.get('k_image')
        k_pdf = request.form.get('k_pdf')

        if not files or any(f.filename == '' for f in files):
            return render_template('upload.html', error='Chưa chọn file')

        if not k_image or not k_image.isdigit() or int(k_image) <= 0 or int(k_image) > 500:
            return render_template('upload.html', error='Giá trị K cho ảnh phải từ 1 đến 500')
        if not k_pdf or not k_pdf.isdigit() or int(k_pdf) <= 0 or int(k_pdf) > 500:
            return render_template('upload.html', error='Giá trị K cho PDF phải từ 1 đến 500')

        k_image = int(k_image)
        k_pdf = int(k_pdf)
        results = []
        compressed_files = []

        for file in files:
            try:
                ext = file.filename.rsplit('.', 1)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                filename = secrets.token_hex(8) + '_' + file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                compressed_name = f"compressed_{filename}"
                compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], compressed_name)
                size_before = os.path.getsize(filepath)

                if ext == 'pdf':
                    extract_first_page_as_image(filepath, os.path.join(app.config['UPLOAD_FOLDER'], f"preview_{filename}.jpg"))
                    compress_pdf_with_svd(filepath, compressed_path, k_pdf)
                else:
                    compress_image_file_with_svd(filepath, compressed_path, k_image)

                size_after = os.path.getsize(compressed_path)
                file_type = 'pdf' if ext == 'pdf' else 'image'
                k_value = k_pdf if ext == 'pdf' else k_image

                insert_file_metadata(compressed_name, session['user']['email'], file_type, k_value, size_before, size_after)

                results.append({
                    'filename': compressed_name,
                    'rate': round(100 * (1 - size_after / size_before), 2),
                    'warning': size_after > size_before,
                    'size_before': size_before,
                    'size_after': size_after
                })

                compressed_files.append(compressed_path)
                os.remove(filepath)

            except Exception as e:
                continue

        if not results:
            return render_template('upload.html', error='Không có file nào được xử lý thành công')

        return render_template('download.html', results=results)

    return render_template('upload.html')

@app.route('/files')
@app.route('/files')
def view_files():
    if 'user' not in session:
        return redirect(url_for('index'))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT filename, created_at, file_type, k_value, size_before, size_after FROM files WHERE owner_email = ? ORDER BY created_at DESC''', (session['user']['email'],))
    files_raw = c.fetchall()
    files = []
    from datetime import datetime, timedelta
    for row in files_raw:
        created_at_utc = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
        created_at_vn = created_at_utc + timedelta(hours=7)
        files.append((row[0], created_at_vn.strftime("%Y-%m-%d %H:%M:%S"), row[2], row[3], row[4], row[5]))
    conn.close()
    return render_template('files.html', files=files)

@app.route('/uploads/<filename>')
def download_file(filename):
    if 'user' not in session:
        return redirect(url_for('index'))
    
    user_email = session['user']['email']
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM files WHERE filename = ?', (filename,))
    row = c.fetchone()
    if row:
        file_id = row[0]
        c.execute('SELECT 1 FROM file_access WHERE file_id = ? AND user_email = ?', (file_id, user_email))
        access = c.fetchone()
        conn.close()
        if access:
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    conn.close()
    return render_template('error.html', message='Bạn không có quyền truy cập file này.')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    return render_template('upload.html', error='File quá lớn (tối đa 20MB)'), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', message='Trang không tồn tại'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message='Lỗi server'), 500

@app.route('/delete_file/<filename>', methods=['POST'])
def delete_file(filename):
    if 'user' not in session:
        return redirect(url_for('index'))

    user_email = session['user']['email']
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT filename FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
    file = c.fetchone()
    if file:
        # Xoá file vật lý
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        # Xoá record DB
        c.execute('DELETE FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
        conn.commit()
    conn.close()
    return redirect(url_for('view_files'))

@app.route('/bulk_delete_files', methods=['POST'])
def bulk_delete_files():
    if 'user' not in session:
        return redirect(url_for('index'))

    user_email = session['user']['email']
    filenames = request.form.getlist('filenames')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for filename in filenames:
        c.execute('SELECT filename FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
        file = c.fetchone()
        if file:
            # Xoá file vật lý
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            # Xoá record DB
            c.execute('DELETE FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
    conn.commit()
    conn.close()
    return redirect(url_for('view_files'))

@app.route('/grant_access', methods=['POST'])
def grant_access():
    if 'user' not in session:
        return redirect(url_for('index'))
    
    user_email = session['user']['email']
    filename = request.form.get('filename')
    target_email = request.form.get('target_email')

    if not target_email:
        return render_template('error.html', message='Vui lòng nhập email người được cấp quyền.')
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
    row = c.fetchone()
    if row:
        file_id = row[0]
        # Kiểm tra đã cấp quyền chưa
        c.execute('SELECT 1 FROM file_access WHERE file_id = ? AND user_email = ?', (file_id, target_email))
        exists = c.fetchone()
        if not exists:
            c.execute('INSERT INTO file_access (file_id, user_email) VALUES (?, ?)', (file_id, target_email))
            conn.commit()
            conn.close()
            return redirect(url_for('view_files'))
        else:
            conn.close()
            return render_template('error.html', message='Người này đã có quyền truy cập file.')
    else:
        conn.close()
        return render_template('error.html', message='Bạn không có quyền cấp quyền cho file này.')
    
@app.route('/bulk_grant_access', methods=['POST'])
def bulk_grant_access():
    if 'user' not in session:
        return redirect(url_for('index'))

    user_email = session['user']['email']
    target_email = request.form.get('bulk_target_email')
    filenames = request.form.getlist('filenames')

    if not target_email:
        return render_template('error.html', message='Vui lòng nhập email người được cấp quyền.')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for filename in filenames:
        c.execute('SELECT id FROM files WHERE filename = ? AND owner_email = ?', (filename, user_email))
        row = c.fetchone()
        if row:
            file_id = row[0]
            c.execute('SELECT 1 FROM file_access WHERE file_id = ? AND user_email = ?', (file_id, target_email))
            exists = c.fetchone()
            if not exists:
                c.execute('INSERT INTO file_access (file_id, user_email) VALUES (?, ?)', (file_id, target_email))
                conn.commit()
                # Gửi email thông báo
                try:
                    msg = EmailMessage()
                    msg['Subject'] = 'Bạn đã được cấp quyền truy cập file'
                    msg['From'] = EMAIL_ADDRESS
                    msg['To'] = target_email
                    msg.set_content(f'Bạn đã được cấp quyền truy cập file: {filename}. Hãy đăng nhập để xem và tải file.')

                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        smtp.send_message(msg)
                except Exception as e:
                    print(f'Gửi email thất bại: {e}')
    conn.close()
    return redirect(url_for('view_files'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
