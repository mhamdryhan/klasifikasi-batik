import pickle
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'models/batik_model_efficientnet.h5'
LE_PATH = 'models/label_encoder.pkl'

deskripsi_batik = {
    "Batik Poleng": "Berasal dari Bali, motif Poleng memiliki ciri khas kotak-kotak hitam dan putih seperti papan catur. Motif ini melambangkan konsep Rwa Bhineda, yaitu keseimbangan antara dua hal yang berlawanan di alam, seperti baik dan buruk, siang dan malam. Kain ini sering digunakan dalam upacara adat dan dianggap sakral.",
    "Batik Insang": "Merupakan motif khas dari Pontianak, Kalimantan Barat, yang terinspirasi dari bentuk insang ikan. Motif ini melambangkan kehidupan masyarakat yang erat kaitannya dengan Sungai Kapuas. Batik Insang sering dijadikan oleh-oleh dan menjadi simbol ketahanan serta adaptasi.",
    "Batik Kawung": "Salah satu motif batik tertua dari Jawa, berbentuk seperti irisan buah kawung (aren) atau kolang-kaling yang tersusun geometris. Motif ini melambangkan kesempurnaan, kemurnian, dan harapan agar pemiliknya menjadi orang yang berguna bagi sesama. Dahulu, motif ini hanya boleh dikenakan oleh kalangan kerajaan.",
    "Batik Ikat Celup": "Ini adalah sebuah teknik, bukan motif tunggal, yang juga dikenal sebagai 'tie-dye'. Prosesnya melibatkan pengikatan bagian-bagian kain sebelum dicelupkan ke dalam pewarna untuk menciptakan pola yang unik dan abstrak. Motif yang dihasilkan melambangkan kebebasan berekspresi, kreativitas, dan penuh warna.",
    "Batik Megamendung": "Ikon batik dari Cirebon, motif ini menggambarkan bentuk awan yang bergulung-gulung. Megamendung melambangkan kesabaran, ketenangan, dan sifat kepemimpinan yang menyejukkan. Meskipun langit mendung, ia membawa hujan yang memberi kehidupan. Motif ini mendapat pengaruh kuat dari kebudayaan Tionghoa."
}

model = None
le_object = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model dari {MODEL_PATH}: {e}")

try:
    with open(LE_PATH, 'rb') as f:
        le_object = pickle.load(f)
    print(f"LabelEncoder berhasil dimuat dari {LE_PATH}")
    print("Kelas yang dimuat dari LabelEncoder:", le_object.classes_)
except Exception as e:
    print(f"Error loading LabelEncoder dari {LE_PATH}: {e}")

def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_ready = preprocess_input(img_array)
    return img_ready

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/classifier', methods=['GET', 'POST'])
def index():
    predicted_class = None
    file_url = None
    error_msg = None
    description = None 
    confidence = None  

    if request.method == 'POST':
        if 'batik_image' not in request.files:
            error_msg = "File tidak ditemukan pada permintaan."
        else:
            file = request.files['batik_image']
            if file.filename == '':
                error_msg = "Tidak ada file yang dipilih."
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_url = url_for('uploaded_file', filename=filename)

                if model and le_object:
                    try:
                        img_ready = prepare_image(file_path)
                        preds = model.predict(img_ready)
                        class_idx = np.argmax(preds, axis=1)[0]
                        confidence = float(preds[0][class_idx])  # Confidence score

                        THRESHOLD = 0.9  

                        if confidence >= THRESHOLD:
                            predicted_class = le_object.inverse_transform([class_idx])[0]
                            description = deskripsi_batik.get(predicted_class, "Deskripsi untuk motif ini belum tersedia.")
                        else:
                            error_msg = f"Motif tidak dapat dideteksi. Silakan coba gambar lain yang lebih jelas."

                    except Exception as e:
                        error_msg = f"Terjadi kesalahan saat proses prediksi: {str(e)}"
                else:
                    error_msg = "Model atau LabelEncoder belum dimuat. Cek log server."

    return render_template('index.html',
                           predicted_class=predicted_class,
                           file_url=file_url,
                           error_msg=error_msg,
                           description=description,
                           confidence=None)  

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/img/<filename>')
def static_img(filename):
    return send_from_directory(os.path.join(app.config['STATIC_FOLDER'], 'img'), filename)