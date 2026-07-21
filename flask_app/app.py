from flask import Flask, render_template, request, redirect
import pandas as pd
from sqlalchemy import create_engine, text
import random
import string
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
engine = create_engine('postgresql://localhost/customer_intelligence')

# ---- Modelleri yükle ----
churn_model = joblib.load('models/churn_model_logistic_regression.pkl')
churn_scaler = joblib.load('models/churn_scaler.pkl')
revenue_model = joblib.load('models/revenue_model_rf.pkl')
kmeans_model = joblib.load('models/kmeans_segmentation_model.pkl')
segmentation_scaler = joblib.load('models/segmentation_scaler.pkl')

MODEL_DOSYALARI = {
    'Logistic Regression': 'models/churn_model_logistic_regression.pkl',
    'Random Forest': 'models/churn_model_random_forest.pkl',
    'XGBoost': 'models/churn_model_xgboost.pkl'
}


def aktif_model_bilgisi():
    df = pd.read_sql(
        "SELECT modelid, modelname FROM models WHERE aktif = TRUE AND modelname IN ('Logistic Regression', 'Random Forest', 'XGBoost') LIMIT 1",
        engine
    )
    if df.empty:
        return None, 'Logistic Regression', churn_model
    model_id = int(df['modelid'].iloc[0])
    model_adi = df['modelname'].iloc[0]
    return model_id, model_adi, joblib.load(MODEL_DOSYALARI[model_adi])


SEGMENT_ISIMLERI = {
    0: 'Yaşlı, Orta Riskli Müşteri',
    1: 'Yüksek Değerli, Sadık Müşteri',
    2: 'Yeni, Yüksek Riskli Müşteri',
    3: 'Genç, Düşük Harcamalı, Düşük Riskli Müşteri'
}

CHURN_FEATURE_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Age', 'SupportTicketCount',
    'Partner_Binary', 'Dependents_Binary', 'PhoneService_Binary', 'PaperlessBilling_Binary', 'Gender_Binary',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]


def rastgele_id_uret():
    sayilar = ''.join(random.choices(string.digits, k=4))
    harfler = ''.join(random.choices(string.ascii_uppercase, k=5))
    return f"{sayilar}-{harfler}"


def musteri_ozelliklerini_hazirla(form):
    tenure = float(form['tenure'])
    ucret = float(form['monthlycharge'])
    total_charges = tenure * ucret

    ozellikler = {
        'SeniorCitizen': 1 if form['seniorcitizen'] == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': ucret,
        'TotalCharges': total_charges,
        'Age': float(form['age']),
        'SupportTicketCount': float(form['supportticketcount']),
        'Partner_Binary': 1 if form['partner'] == 'Yes' else 0,
        'Dependents_Binary': 1 if form['dependents'] == 'Yes' else 0,
        'PhoneService_Binary': 1 if form['phoneservice'] == 'Yes' else 0,
        'PaperlessBilling_Binary': 1 if form['paperlessbilling'] == 'Yes' else 0,
        'Gender_Binary': 1 if form['gender'] == 'Male' else 0,
        'MultipleLines_No phone service': 1 if form['multiplelines'] == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if form['multiplelines'] == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if form['internetservice'] == 'Fiber optic' else 0,
        'InternetService_No': 1 if form['internetservice'] == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if form['onlinesecurity'] == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if form['onlinesecurity'] == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if form['onlinebackup'] == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if form['onlinebackup'] == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if form['deviceprotection'] == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if form['deviceprotection'] == 'Yes' else 0,
        'TechSupport_No internet service': 1 if form['techsupport'] == 'No internet service' else 0,
        'TechSupport_Yes': 1 if form['techsupport'] == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if form['streamingtv'] == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if form['streamingtv'] == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if form['streamingmovies'] == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if form['streamingmovies'] == 'Yes' else 0,
        'Contract_One year': 1 if form['contract'] == 'One year' else 0,
        'Contract_Two year': 1 if form['contract'] == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if form['paymentmethod'] == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if form['paymentmethod'] == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if form['paymentmethod'] == 'Mailed check' else 0,
    }
    return ozellikler
def urun_onerisi_uret(segment_adi, risk, tahmini_gelir):
    if risk == 'High':
        return "Sadakat Paketi + %20 İndirim Kampanyası"
    elif segment_adi == 'Yüksek Değerli, Sadık Müşteri':
        return "Premium Fiber + Ekstra Veri Paketi"
    elif segment_adi == 'Genç, Düşük Harcamalı, Düşük Riskli Müşteri':
        return "Öğrenci/Genç Kampanyası"
    elif tahmini_gelir > 3000:
        return "Kurumsal Paket Önerisi"
    else:
        return "Standart Paket"

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/musteriler')
def musteriler():
    sayfa = request.args.get('sayfa', 1, type=int)
    offset = (sayfa - 1) * 50

    df = pd.read_sql(f'SELECT * FROM customers LIMIT 50 OFFSET {offset}', engine)

    toplam = pd.read_sql('SELECT COUNT(*) as adet FROM customers', engine)['adet'].iloc[0]
    toplam_sayfa = (toplam // 50) + 1

    return render_template('musteriler.html',
                           musteriler=df.to_dict(orient='records'),
                           sayfa=sayfa,
                           toplam_sayfa=toplam_sayfa)


@app.route('/dashboard')
def dashboard():
    toplam_musteri = pd.read_sql('SELECT COUNT(*) as adet FROM customers', engine)['adet'].iloc[0]

    churn_data = pd.read_sql('SELECT AVG(churnprobability) as oran FROM predictions', engine)
    churn_orani = round(churn_data['oran'].iloc[0] * 100, 2)

    revenue_data = pd.read_sql('SELECT AVG(predictedrevenue) as ortalama FROM predictions', engine)
    ortalama_gelir = round(revenue_data['ortalama'].iloc[0], 2) if revenue_data['ortalama'].iloc[0] is not None else 0

    segment_data = pd.read_sql('''
        SELECT segmentlabel, COUNT(*) as adet
        FROM predictions
        WHERE segmentlabel IS NOT NULL
        GROUP BY segmentlabel
    ''', engine)
    segment_dagilimi = segment_data.to_dict(orient='records')

    return render_template('dashboard.html',
                           toplam_musteri=toplam_musteri,
                           churn_orani=churn_orani,
                           ortalama_gelir=ortalama_gelir,
                           segment_dagilimi=segment_dagilimi)


@app.route('/musteri-ekle', methods=['GET', 'POST'])
def musteri_ekle():
    if request.method == 'POST':
        yeni_id = rastgele_id_uret()
        yeni_isim = request.form['name']
        yeni_yas = request.form['age']
        yeni_cinsiyet = request.form['gender']
        yeni_sozlesme = request.form['contracttype']
        yeni_ucret = request.form['monthlycharge']
        yeni_tenure = request.form['tenure']
        yeni_internet = request.form['internetservice']
        yeni_odeme = request.form['paymentmethod']
        yeni_destek = request.form['supportticketcount']

        insert_sorgusu = text('''
            INSERT INTO customers (customerid, name, age, gender, contracttype,
                                    monthlycharge, tenure, internetservice,
                                    paymentmethod, supportticketcount)
            VALUES (:id, :isim, :yas, :cinsiyet, :sozlesme,
                    :ucret, :tenure, :internet,
                    :odeme, :destek)
        ''')
        with engine.connect() as conn:
            conn.execute(insert_sorgusu, {
                'id': yeni_id,
                'isim': yeni_isim,
                'yas': yeni_yas,
                'cinsiyet': yeni_cinsiyet,
                'sozlesme': yeni_sozlesme,
                'ucret': yeni_ucret,
                'tenure': yeni_tenure,
                'internet': yeni_internet,
                'odeme': yeni_odeme,
                'destek': yeni_destek
            })
            conn.commit()

        return redirect('/musteriler')

    return render_template('musteri_ekle.html')


@app.route('/musteri-sil/<musteri_id>')
def musteri_sil(musteri_id):
    sil_sorgusu = text('DELETE FROM customers WHERE customerid = :id')
    with engine.connect() as conn:
        conn.execute(sil_sorgusu, {'id': musteri_id})
        conn.commit()
    return redirect('/musteriler')


@app.route('/musteri-duzenle/<musteri_id>', methods=['GET', 'POST'])
def musteri_duzenle(musteri_id):
    if request.method == 'POST':
        yeni_isim = request.form['name']
        yeni_yas = request.form['age']
        yeni_cinsiyet = request.form['gender']
        yeni_sozlesme = request.form['contracttype']
        yeni_ucret = request.form['monthlycharge']
        yeni_tenure = request.form['tenure']
        yeni_internet = request.form['internetservice']
        yeni_odeme = request.form['paymentmethod']
        yeni_destek = request.form['supportticketcount']

        guncelle_sorgusu = text('''
            UPDATE customers 
            SET name = :isim, age = :yas, gender = :cinsiyet, 
                contracttype = :sozlesme, monthlycharge = :ucret,
                tenure = :tenure, internetservice = :internet,
                paymentmethod = :odeme, supportticketcount = :destek
            WHERE customerid = :id
        ''')
        with engine.connect() as conn:
            conn.execute(guncelle_sorgusu, {
                'isim': yeni_isim, 'yas': yeni_yas, 'cinsiyet': yeni_cinsiyet,
                'sozlesme': yeni_sozlesme, 'ucret': yeni_ucret, 'tenure': yeni_tenure,
                'internet': yeni_internet, 'odeme': yeni_odeme, 'destek': yeni_destek,
                'id': musteri_id
            })
            conn.commit()

        return redirect('/musteriler')

    df = pd.read_sql(text('SELECT * FROM customers WHERE customerid = :id'),
                     engine, params={'id': musteri_id})
    musteri = df.iloc[0].to_dict()

    return render_template('musteri_duzenle.html', musteri=musteri)


@app.route('/tahmin', methods=['GET', 'POST'])
def tahmin():
    if request.method == 'POST':
        ozellikler = musteri_ozelliklerini_hazirla(request.form)
        X = pd.DataFrame([ozellikler])[CHURN_FEATURE_COLUMNS]

        # ---- CHURN TAHMİNİ ----
        X_olceklenmis = churn_scaler.transform(X)
        model_id, model_adi, secili_model = aktif_model_bilgisi()
        churn_olasiligi = secili_model.predict_proba(X_olceklenmis)[:, 1][0]

        if churn_olasiligi >= 0.7:
            risk = 'High'
        elif churn_olasiligi >= 0.4:
            risk = 'Medium'
        else:
            risk = 'Low'

        # ---- GELİR TAHMİNİ ----
        revenue_kolonlari = [c for c in CHURN_FEATURE_COLUMNS if c not in ['tenure', 'TotalCharges']]
        X_revenue = X[revenue_kolonlari]
        tahmini_gelir = revenue_model.predict(X_revenue)[0]

        # ---- SEGMENT TAHMİNİ ----
        segment_kolonlari = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Age', 'SupportTicketCount']
        X_segment = X[segment_kolonlari]
        X_segment_olcekli = segmentation_scaler.transform(X_segment)
        kume_no = kmeans_model.predict(X_segment_olcekli)[0]
        segment_adi = SEGMENT_ISIMLERI.get(kume_no, 'Bilinmeyen Segment')
        # ---- SEGMENT TAHMİNİ ----
        segment_kolonlari = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Age', 'SupportTicketCount']
        X_segment = X[segment_kolonlari]
        X_segment_olcekli = segmentation_scaler.transform(X_segment)
        kume_no = kmeans_model.predict(X_segment_olcekli)[0]
        segment_adi = SEGMENT_ISIMLERI.get(kume_no, 'Bilinmeyen Segment')

        # ---- ÜRÜN ÖNERİSİ ----
        oneri = urun_onerisi_uret(segment_adi, risk, tahmini_gelir)


        musteri_id = request.form['customerid']

        kayit_sorgusu = text('''
            INSERT INTO predictions (customerid, predictiondate, churnprobability, risklevel, predictedrevenue, segmentlabel)
            VALUES (:id, CURRENT_DATE, :olasilik, :risk, :gelir, :segment)
        ''')
        with engine.connect() as conn:
            conn.execute(kayit_sorgusu, {
                'id': musteri_id,
                'olasilik': round(float(churn_olasiligi), 4),
                'risk': risk,
                'gelir': round(float(tahmini_gelir), 2),
                'segment': segment_adi
            })
            conn.commit()

        log_sorgusu = text('''
            INSERT INTO predictionlogs (customerid, modelid, predictiondate, result)
            VALUES (:id, :modelid, CURRENT_DATE, :sonuc)
        ''')
        with engine.connect() as conn:
            conn.execute(log_sorgusu, {
                'id': musteri_id,
                'modelid': model_id,
                'sonuc': f"Churn: %{round(churn_olasiligi * 100, 1)}, Risk: {risk}, Segment: {segment_adi}, Kampanya: {oneri}"
            })
            conn.commit()

        return render_template('tahmin_sonuc.html',
                               olasilik=round(churn_olasiligi * 100, 2),
                               risk=risk,
                               gelir=round(tahmini_gelir, 2),
                               segment=segment_adi,
                               oneri=oneri)
    musteri_id = request.args.get('customerid', '')
    musteri = None

    if musteri_id:
        df = pd.read_sql(text('SELECT * FROM customers WHERE customerid = :id'),
                         engine, params={'id': musteri_id})
        if not df.empty:
            musteri = df.iloc[0].to_dict()

    return render_template('tahmin.html', musteri=musteri, musteri_id=musteri_id)


@app.route('/analitik')
def analitik():
    segment_data = pd.read_sql('''
        SELECT segmentlabel, COUNT(*) as adet
        FROM predictions
        WHERE segmentlabel IS NOT NULL
        GROUP BY segmentlabel
    ''', engine)

    fig, ax = plt.subplots()
    ax.bar(segment_data['segmentlabel'], segment_data['adet'])
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    grafik_kodu = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return render_template('analitik.html', segment_grafik=grafik_kodu)


@app.route('/gecmis')
def gecmis():
    tarih_filtre = request.args.get('tarih', '')
    risk_filtre = request.args.get('risk', '')

    sorgu = '''
        SELECT pl.logid, pl.customerid, c.name, pl.predictiondate, pl.result, m.modelname,
               SPLIT_PART(pl.result, 'Kampanya: ', 2) AS oneri
        FROM predictionlogs pl
        LEFT JOIN customers c ON pl.customerid = c.customerid
        LEFT JOIN models m ON pl.modelid = m.modelid
        WHERE 1=1
    '''
    parametreler = {}

    if tarih_filtre:
        sorgu += ' AND pl.predictiondate = :tarih'
        parametreler['tarih'] = tarih_filtre

    if risk_filtre:
        sorgu += ' AND pl.result ILIKE :risk'
        parametreler['risk'] = f'%Risk: {risk_filtre}%'

    sorgu += ' ORDER BY pl.logid DESC'

    df = pd.read_sql(text(sorgu), engine, params=parametreler)

    return render_template('gecmis.html',
                           loglar=df.to_dict(orient='records'),
                           tarih_filtre=tarih_filtre,
                           risk_filtre=risk_filtre)


@app.route('/modeller')
def modeller():
    df = pd.read_sql('SELECT * FROM models ORDER BY modelid', engine)
    return render_template('modeller.html', modeller=df.to_dict(orient='records'))


@app.route('/model-aktif-yap/<int:model_id>')
def model_aktif_yap(model_id):
    with engine.connect() as conn:
        conn.execute(text('UPDATE models SET aktif = FALSE'))
        conn.execute(text('UPDATE models SET aktif = TRUE WHERE modelid = :id'), {'id': model_id})
        conn.commit()
    return redirect('/modeller')


if __name__ == "__main__":
    app.run(debug=True, port=5001)