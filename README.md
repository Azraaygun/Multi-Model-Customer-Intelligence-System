# Multi-Model Customer Intelligence System

Müşteri davranışlarını analiz eden, makine öğrenmesi modelleriyle churn (müşteri kaybı) tahmini, gelir tahmini ve müşteri segmentasyonu yapan; sonuçları bir PostgreSQL veritabanında yöneten ve Flask tabanlı bir web arayüzü üzerinden sunan uçtan uca bir müşteri istihbarat sistemi.

Bu proje, veri analizi, makine öğrenmesi, veritabanı tasarımı ve web geliştirme adımlarının tamamını içeren bir uygulama olarak geliştirilmiştir.

---

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Veri Seti](#veri-seti)
- [Veritabanı Tasarımı](#veritabanı-tasarımı)
- [Proje Aşamaları](#proje-aşamaları)
- [Makine Öğrenmesi Modelleri](#makine-öğrenmesi-modelleri)
- [Web Uygulaması](#web-uygulaması)
- [Kişiselleştirilmiş Kampanya Önerisi Sistemi](#kişiselleştirilmiş-kampanya-önerisi-sistemi)
- [Kurulum](#kurulum)
- [Klasör Yapısı](#klasör-yapısı)
- [Karşılaşılan Zorluklar](#karşılaşılan-zorluklar)
---

## Proje Amacı

Bu projenin amacı, müşteri verileri üzerinde yalnızca makine öğrenmesi modeli geliştirmek değil, aynı zamanda bu modeli gerçek bir veritabanı yapısı ve web arayüzü ile entegre ederek uçtan uca çalışan bir analiz sistemi oluşturmaktır.

Sistem üç temel problemi çözer:

- **Müşteri kaybı tahmini** (Classification) — bir müşterinin churn edip etmeyeceğini tahmin eder
- **Müşteri segmentasyonu** (Clustering) — müşterileri davranışsal olarak benzer gruplara ayırır
- **Gelir tahmini** (Regression) — bir müşterinin toplam getireceği geliri tahmin eder

Tüm süreç bir PostgreSQL veritabanı üzerinden yönetilir ve izlenir.

---
## Sistem Mimarisi

```
Veri Seti
   ↓
Python (EDA + Feature Engineering)
   ↓
ML Modelleri (Churn / Segmentation / Revenue)
   ↓
Model Kaydı (joblib)
   ↓
PostgreSQL
   ↓
Flask Backend
   ↓
CRUD + API
   ↓
Web Arayüzü
   ↓
Tahmin Logları + Analitik
```

---

## Kullanılan Teknolojiler

**Programlama Dili**
- Python 3

**Veri Analizi ve Makine Öğrenmesi**
- Pandas, NumPy
- Scikit-learn
- XGBoost

**Görselleştirme**
- Matplotlib, Seaborn

**Web Framework**
- Flask
- Jinja2 (template motoru)

**Veritabanı**
- PostgreSQL 16
- SQLAlchemy (ORM/sorgu katmanı)

**Diğer**
- joblib (model kaydetme/yükleme)
- Faker (sentetik veri üretimi)

---
## Veri Seti

Proje, **IBM Telco Customer Churn Dataset** üzerine kuruludur (7.043 müşteri kaydı, 21 orijinal değişken).

Veri seti aşağıdaki bilgi kategorilerini içerir:

- Demografik bilgiler (cinsiyet, yaşlılık durumu, medeni durum, bakmakla yükümlü olunan kişiler)
- Abonelik bilgileri (sözleşme türü, internet servisi, ek hizmetler)
- Aylık ve toplam ücret bilgileri
- Kullanım süresi (tenure)
- Ödeme yöntemi
- Churn durumu

### Veri Seti Seçim Süreci: Maven Telecom vs. IBM Telco

Veri seti belirlenirken, IBM'in orijinal veri setine ek olarak Maven Analytics'in genişletilmiş **Maven Telecom Churn Dataset**'i de değerlendirilmiş ve üzerinde tam bir EDA çalışması yapılmıştır. Nihai olarak IBM veri seti tercih edilmiştir.

| Kriter | Maven Telecom | IBM Telco (seçilen) |
|---|---|---|
| Kaynak güvenilirliği | Maven Analytics (ikincil kaynak) | IBM resmi GitHub deposu (birincil kaynak) |
| Kolon sayısı | 38 | 21 |
| Dosya yapısı | 3 ayrı dosya (birleştirme gerekli) | Tek dosya |
| Age bilgisi | Mevcut | Yok (sentetik üretildi) |
| Gelir hedefi | `Total Revenue` (hazır) | Yok (`TotalCharges` kullanıldı) |
| Veri anomalisi | `Monthly Charge`'da 120 satırda açıklanamayan negatif değer | Anomali tespit edilmedi |
| Churn değişkeni | 3 kategori (Stayed/Churned/Joined) | 2 kategori (Yes/No) |
| Literatür/kaynak desteği | Sınırlı | Çok yaygın |

**Seçim gerekçesi:** Maven veri setinde `Monthly Charge` değişkeninde istatistiksel olarak açıklanamayan bir anomali (120 satırda -1 ile -10 arası negatif değer) tespit edilmesi, veri setinin sentetik/simüle edilmiş doğasına işaret etmiştir. Buna karşın IBM veri seti, doğrudan IBM'in resmi deposundan temin edilen, literatürde en yaygın kullanılan referans veri setlerinden biridir. Kaynak güvenilirliği, veri kalitesi, tek dosyalık sade yapısı ve karşılaştırılabilirlik açısından üstün bulunduğu için **IBM Telco veri seti nihai karar olarak belirlenmiştir**. Maven veri seti üzerindeki EDA çalışmasının tam bulguları ayrı bir belgede raporlanmıştır.

### Veri Setinde Bulunmayan, Sentetik Olarak Üretilen Alanlar

Ödev şartnamesinin istediği bazı alanlar (`Name`, `Age`, `SupportTicketCount`) gerçek veri setinde bulunmadığından, gizlilik ve iş mantığı gerekçeleriyle sentetik olarak üretilmiştir:

| Alan | Üretim Yöntemi |
|---|---|
| `Name` | Faker kütüphanesi (`tr_TR` locale), cinsiyete uygun isim/soyisim üretimi |
| `Age` | 18–80 aralığında rastgele tam sayı |
| `SupportTicketCount` | Poisson dağılımı — `TechSupport` hizmeti olmayan müşterilerde ortalama 3, olanlarda ortalama 1 destek talebi |

---

## Veritabanı Tasarımı

PostgreSQL üzerinde 4 tablo bulunur:

### `customers`
Müşteri temel bilgileri (CustomerId, Name, Age, Gender, ContractType, MonthlyCharge, Tenure, InternetService, PaymentMethod, SupportTicketCount)

### `models`
Eğitilen modellerin performans kayıtları (ModelId, ModelName, Algorithm, Accuracy, Precision, Recall, F1Score, TrainDate, Aktif)

### `predictions`
Model tahmin sonuçları (PredictionId, CustomerId → customers, PredictionDate, ChurnProbability, RiskLevel, PredictedRevenue, SegmentLabel)

### `predictionlogs`
Tüm tahmin geçmişi (LogId, CustomerId → customers, ModelId → models, PredictionDate, Result)

**İlişkiler:** `predictions` ve `predictionlogs` tabloları `customers` tablosuna; `predictionlogs` ayrıca `models` tablosuna foreign key ile bağlıdır.

---
## Proje Aşamaları

### Aşama 1 — Keşifsel Veri Analizi (EDA)
- Eksik veri analizi
- Aykırı değer analizi (IQR yöntemi)
- Churn oranı analizi
- Korelasyon analizi
- Görselleştirmeler (boxplot, heatmap, dağılım grafikleri)

### Aşama 2 — Feature Engineering
- Kategorik değişken encoding (binary + One-Hot Encoding)
- Tenure gruplama
- TotalCharges eksik veri düzenleme
- Kural bazlı ön-segment oluşturma

### Aşama 3 — Makine Öğrenmesi
Üç ayrı problem için toplam 6 model eğitildi.

### Aşama 4 — Model Değerlendirme
Tüm modellerin performans metrikleri karşılaştırmalı olarak değerlendirildi, her problem için en iyi model belirlendi.

### Aşama 5 — Model Kaydetme

En iyi modeller `joblib` ile `.pkl` formatında kaydedildi; model bilgileri veritabanına yazıldı.

---

## Makine Öğrenmesi Modelleri

### Problem 1 — Churn Prediction (Classification)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.819 | 0.681 | 0.595 | 0.635 |
| Random Forest | 0.789 | 0.639 | 0.469 | 0.541 |
| XGBoost | 0.795 | 0.638 | 0.520 | 0.573 |

**Not:** Veri sızıntısını önlemek için model, `tenure` ve `TotalCharges` arasındaki doğrusal ilişkiden etkilenmeyecek şekilde tasarlanmıştır.

### Problem 2 — Customer Segmentation (Clustering)

**KMeans** (k=4), Silhouette Score: **0.2473**

| Segment | Ortalama Tenure | Ortalama Aylık Ücret | Churn Oranı |
|---|---|---|---|
| Yüksek Değerli, Sadık Müşteri | 59 ay | 91.64 | %15.30 |
| Yeni, Yüksek Riskli Müşteri | 16 ay | 74.11 | %48.73 |
| Yaşlı, Orta Riskli Müşteri | 21 ay | 49.45 | %28.06 |
| Genç, Düşük Harcamalı, Düşük Riskli Müşteri | 27 ay | 39.14 | %17.02 |

### Problem 3 — Revenue Prediction (Regression)

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 703.38 | 0.758 |
| **Random Forest Regressor** ⭐ | 82.39* | **0.781** |

*Not: `tenure` değişkeni veri sızıntısını önlemek amacıyla modelden çıkarılmıştır; bu düzeltme öncesi R² değeri yanıltıcı şekilde 0.999 çıkmaktaydı.*

---

## Web Uygulaması

Flask üzerinde geliştirilen web uygulaması aşağıdaki sayfaları içerir:

| Sayfa | Açıklama |
|---|---|
| **Ana Sayfa** | Tüm modüllere erişim sağlayan karşılama ekranı |
| **Dashboard** | Toplam müşteri sayısı, churn oranı, ortalama gelir, segment dağılımı (canlı SQL sorgularıyla) |
| **Müşteri Yönetimi** | Müşteri listeleme (arama + sayfalama), ekleme, güncelleme, silme (CRUD) |
| **Tahmin Sayfası** | Müşteri ID ile bilgi otomatik çekme veya elle giriş; churn/gelir/segment tahmini; kural bazlı kampanya önerisi; sonuçların veritabanına kaydı |
| **Analitik Sayfa** | Segment dağılımı (canlı grafik), feature importance, segmentasyon görselleştirmesi, gelir tahmini karşılaştırması |
| **Tahmin Geçmişi** | Geçmiş tahminlerin listesi; tarih ve risk seviyesine göre filtreleme |
| **Model Yönetimi** | Tüm modellerin metriklerini görüntüleme; churn modelleri arasından aktif modeli seçme |

### Öne Çıkan Özellikler

- **Dinamik aktif model seçimi:** Tahmin sayfası, `Model Yönetimi` sayfasından seçilen aktif churn modelini (Logistic Regression / Random Forest / XGBoost) gerçek zamanlı olarak kullanır.
- **Otomatik veri getirme:** Mevcut bir müşteri ID'si girildiğinde, müşteri bilgileri veritabanından otomatik çekilip form doldurulur.
- **Kişiselleştirilmiş kampanya önerisi:** Tahmin sonucundaki risk seviyesi, segment ve tahmini gelire göre otomatik paket/kampanya önerisi üretilir (bkz. [Kişiselleştirilmiş Kampanya Önerisi Sistemi](#kişiselleştirilmiş-kampanya-önerisi-sistemi)).
- **SQL enjeksiyonuna karşı güvenli sorgular:** Tüm veritabanı sorguları parametreli (`text()` + `params`) olarak yazılmıştır.

### Kişiselleştirilmiş Kampanya Önerisi Sistemi

Her tahmin işleminin sonunda, sistem müşteriye özel bir **paket/kampanya önerisi** üretir. Bu öneri, ayrı bir makine öğrenmesi modeli eğitilmeden, o an hesaplanan churn riski, müşteri segmenti ve tahmini gelir çıktılarına dayanan kural bazlı bir mantıkla belirlenir:

| Koşul | Önerilen Paket |
|---|---|
| Risk seviyesi **High** | Sadakat Paketi + %20 İndirim Kampanyası |
| Segment: **Yüksek Değerli, Sadık Müşteri** | Premium Fiber + Ekstra Veri Paketi |
| Segment: **Genç, Düşük Harcamalı, Düşük Riskli Müşteri** | Öğrenci/Genç Kampanyası |
| Tahmini gelir **3000'den büyük** | Kurumsal Paket Önerisi |
| Yukarıdakilerin hiçbiri değilse | Standart Paket |

Koşullar sırayla değerlendirilir; bir müşteri birden fazla koşulu karşılasa bile **ilk uyan kural** geçerli olur (örneğin hem yüksek riskli hem yüksek değerli bir müşteriye önce risk azaltıcı kampanya önerilir). Üretilen öneri, hem tahmin sonucu ekranında gösterilir hem de `PredictionLogs` tablosuna kaydedilerek Tahmin Geçmişi sayfasından geriye dönük izlenebilir.

Bu yaklaşım, veri setinde "önerilen ürün" gibi bir hedef değişken bulunmadığından, mevcut model çıktılarını kullanan pratik ve hızlı uygulanabilir bir çözüm olarak tercih edilmiştir. 

---
## Kurulum

### Gereksinimler
- Python 3.9+
- PostgreSQL 16
- Homebrew (macOS için)

### Adımlar

```bash
# 1. Sanal ortam oluştur ve aktive et
python3 -m venv venv
source venv/bin/activate

# 2. Bağımlılıkları kur
pip3 install flask flask-sqlalchemy pandas numpy scikit-learn xgboost \
             matplotlib seaborn sqlalchemy psycopg2-binary joblib faker

# 3. PostgreSQL'i kur ve başlat
brew install postgresql@16
brew services start postgresql@16
createdb customer_intelligence

# 4. Veritabanı tablolarını oluştur (SQL şeması için dokümana bakınız)
psql customer_intelligence

# 5. Flask uygulamasını çalıştır
cd flask_app
python3 app.py
```

Uygulama `http://localhost:5001` adresinde çalışır.

---

## Klasör Yapısı

```
Multi-Model Customer Intelligence System/
│
├── flask_app/
│   ├── app.py                  # Flask backend, tüm route'lar
│   ├── models/                 # Eğitilmiş .pkl model dosyaları
│   ├── static/                 # Statik grafikler (feature importance vb.)
│   └── templates/              # HTML/Jinja2 şablonları
│
├── *.ipynb                     # EDA, Feature Engineering, model eğitimi notebook'ları
├── telecom_*.csv                # Veri setinin işlenmiş versiyonları
├── churn_*.csv / .png           # Churn modeli çıktıları ve grafikleri
├── revenue_*.csv / .png         # Revenue modeli çıktıları ve grafikleri
├── segmentation_*.csv / .png    # Segmentasyon çıktıları ve grafikleri
└── README.md
```

---

## Karşılaşılan Zorluklar

- **Veri sızıntısı (Data Leakage):** Revenue Prediction modelinde `tenure` değişkeninin `TotalCharges` ile neredeyse birebir doğrusal ilişkisi, ilk denemede yapay olarak yüksek bir R² (0.999) skoruna yol açmıştır. Değişken modelden çıkarılarak gerçekçi bir performans (R²=0.781) elde edilmiştir.
- **Sentetik veri sınırlamaları:** Gerçek veri setinde bulunmayan `Name`, `Age`, `SupportTicketCount` alanları, istatistiksel olarak mantıklı varsayımlarla (Faker, Poisson dağılımı) türetilmiştir; bu durum raporlarda açıkça belirtilmiştir.
- **Veritabanı şema kısıtlamaları:** `Models` tablosunun sabit şeması (Accuracy/Precision/Recall) yalnızca classification modelleri için anlamlıdır; regression ve clustering modellerinde bu alanlar boş bırakılmış, ana metrik `F1Score` alanına (R² veya Silhouette Score olarak) kaydedilmiştir.
- **Aktif model — modele özgü metrik uyumsuzluğu:** Yalnızca churn modelleri (Logistic Regression, Random Forest, XGBoost) arasında aktif model seçimi anlamlıdır; bu nedenle arayüzde "Aktif Yap" seçeneği yalnızca bu üç modelde gösterilmektedir.



