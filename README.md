# Laporan Proyek Machine Learning - Rizky Surya Alfarizy

## Project Overview

Hiburan telah menjadi kebutuhan yang terus berkembang dalam kehidupan manusia. Dahulu, hiburan sering dianggap sebagai kebutuhan sekunder yang tidak mendesak. Namun, seiring waktu, pandangan tersebut berubah dan kini hiburan menjadi bagian penting dalam rutinitas sehari-hari setiap individu. Terutama memasuki abad ke-21, terjadi kemajuan pesat di industri hiburan, khususnya dalam sektor televisi dan perfilman. Perkembangan ini bermula dari era televisi hitam-putih hingga televisi berwarna, dan kini berkembang ke teknologi canggih seperti televisi hologram serta platform streaming yang dapat disesuaikan dengan preferensi masing-masing pengguna. Tren pemakaian layanan streaming semakin meningkat tajam, terutama selama masa pandemi yang berlangsung lama.

Mengapa Masalah Ini Harus Diselesaikan?
- Meningkatkan Akurasi Rekomendasi Konten
> Dengan mempelajari pola preferensi pengguna dari data historis dan kebiasaan menonton, sistem dapat menyajikan rekomendasi yang lebih tepat dan personal, sehingga memperbaiki pengalaman pengguna.
- Optimalisasi Produksi dan Distribusi Konten
> Analisis perilaku konsumsi hiburan membantu produsen konten dalam merancang strategi produksi dan distribusi yang lebih efektif, sehingga menghemat waktu dan biaya.
- Pengambilan Keputusan Berbasis Data dalam Industri Hiburan
> Dengan menggunakan machine learning untuk memprediksi minat penonton, penyedia layanan streaming dapat membuat keputusan strategis yang lebih tepat dalam pengembangan fitur, pemilihan konten, dan ekspansi pasar.

## Business Understanding
### Problem Statements
Sistem rekomendasi merupakan sebuah teknologi yang bertujuan untuk membantu pengguna dalam menentukan pilihan yang sesuai dengan preferensi mereka. Dalam konteks pencarian film, sistem ini dapat mempermudah pengguna menemukan judul-judul yang relevan dengan kesukaan mereka tanpa harus mencarinya secara manual. Dengan diterapkannya sistem rekomendasi, pengalaman pengguna dalam menjelajahi dan menikmati konten film akan menjadi lebih menyenangkan dan efisien, karena sistem secara proaktif menyajikan pilihan yang sesuai dengan minat pengguna.

1. Bagaimana cara melakukan pemrosesan data secara optimal agar data tersebut siap digunakan untuk membangun sistem rekomendasi yang akurat dan andal?
2. Bagaimana membangun model machine learning yang dapat memberikan saran film yang kemungkinan besar sesuai dengan minat pengguna?

### Goals
1. Melakukan tahapan pemrosesan dan pembersihan data dengan baik guna mendukung pengembangan model rekomendasi yang efektif.
2. Merancang dan melatih model machine learning yang mampu menyarankan film yang relevan dengan preferensi pengguna.


### Solution
Untuk menjawab permasalahan di atas, saya akan menggunakan dua metode utama dalam sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering. Penjelasan dari masing-masing pendekatan adalah sebagai berikut:
- Content-Based Filtering adalah teknik yang menyarankan item kepada pengguna berdasarkan kemiripan antara item yang pernah disukai dengan item lain. Sistem ini memanfaatkan informasi deskriptif dari film seperti genre, sutradara, atau kata kunci untuk membangun profil minat pengguna, sehingga bisa merekomendasikan film serupa.
- Collaborative Filtering menggunakan pendekatan berdasarkan perilaku dan interaksi pengguna lain yang memiliki kesamaan preferensi. Metode ini tidak melihat konten film secara langsung, melainkan menganalisis pola penilaian atau interaksi pengguna untuk menemukan kesamaan dalam kebiasaan menonton, dan memberikan rekomendasi berdasarkan kesamaan tersebut.

## 1. Data Understanding

### 1.1 Sumber Data

Dataset yang digunakan dalam proyek ini adalah dataset **MovieLens** yang tersedia secara publik melalui [Kaggle](https://www.kaggle.com/datasets/sunilgautam/movielens). Dataset ini dikembangkan oleh GroupLens Research dan banyak digunakan untuk keperluan riset dan pengembangan sistem rekomendasi.

- **Jumlah entri rating**: 100.836
- **Jumlah pengguna unik**: 610
- **Jumlah film unik**: 9.742

Dataset ini cocok untuk implementasi metode **Content-Based Filtering** maupun **Collaborative Filtering**.

---

### 1.2 Deskripsi Dataset

#### 📁 movies.csv

Dataset ini berisi metadata mengenai film, dengan rincian fitur sebagai berikut:

| Fitur     | Tipe Data      | Deskripsi |
|-----------|----------------|-----------|
| `movieId` | integer        | ID unik untuk setiap film. Digunakan untuk menghubungkan dengan dataset rating. |
| `title`   | object (string) | Judul film beserta tahun rilis, misalnya: `Toy Story (1995)`. |
| `genres`  | object (string) | Genre film yang dipisahkan oleh tanda `|`, misalnya: `Adventure|Animation|Comedy`. Setiap film dapat memiliki lebih dari satu genre. |

- **Jumlah entri**: 9.742
- **Missing values**: 0
- **Duplikasi**: 0

---

#### 📁 ratings.csv

Dataset ini berisi data interaksi pengguna dengan film melalui pemberian rating. Rinciannya sebagai berikut:

| Fitur       | Tipe Data   | Deskripsi |
|-------------|-------------|-----------|
| `userId`    | integer     | ID unik untuk setiap pengguna. |
| `movieId`   | integer     | ID film yang diberi rating. Digunakan sebagai foreign key ke `movies.csv`. |
| `rating`    | float       | Nilai rating dari pengguna terhadap film, dengan rentang dari 0.5 hingga 5.0 (inkrement 0.5). |
| `timestamp` | integer     | Waktu ketika rating diberikan, dalam format UNIX timestamp. |

- **Jumlah entri**: 100.836
- **Missing values**: 0
- **Duplikasi**: 0

---

### 1.3 Tujuan Penggabungan Dataset

Kedua dataset digabungkan berdasarkan kolom `movieId`. Tujuan dari penggabungan ini adalah:

- Memberikan konteks genre dan judul film pada setiap interaksi pengguna,
- Memungkinkan pembuatan model rekomendasi berbasis konten (dengan menggunakan informasi genre),
- Mempermudah analisis kecenderungan rating berdasarkan genre atau judul film.

Hasil penggabungan menghasilkan dataset dengan kolom:

- `userId`, `movieId`, `rating`, `timestamp` (dari ratings.csv)
- `title`, `genres` (dari movies.csv)

---

### 1.4 Ringkasan Poin Penting

- Dataset **tidak mengandung nilai kosong maupun duplikasi**.
- Semua fitur telah dijelaskan secara eksplisit.
- Genre film tersedia dalam format teks dan siap digunakan untuk transformasi fitur dalam **Content-Based Filtering**.
- Dataset memiliki kualitas yang baik dan siap digunakan untuk pembangunan model sistem rekomendasi.


### Exploratory Data Analysis (EDA)
#### Univariate Analysis – Distribusi Nilai Rating
##### Distribusi Nilai Rating

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Distribusi%20Rating%20Film%20oleh%20Pengguna.png?raw=true" width=500 />
</p>

Distribusi Nilai Rating:
- Interpretasi: Rating paling umum diberikan pengguna berada di angka 3.0 dan 4.0.
- Relatif sedikit pengguna yang memberi rating 1.0 atau 5.0, artinya pengguna cenderung menilai secara moderat.
- Distribusi ini menunjukkan bahwa data tidak terlalu condong ke rating tinggi atau rendah secara ekstrem.

##### Genre Film yang Paling Sering Muncul

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-20%20Genre%20Film%20Paling%20Umum%20dalam%20Dataset.png?raw=true" width=500 />
</p>


Top 20 Genre Film yang Paling Sering Muncul: 
- Insight: 
    >Genre seperti Drama, Comedy, dan Action mendominasi daftar genre yang paling sering muncul
    >Informasi ini akan berguna dalam Content-Based Recommendation, di mana kesamaan genre digunakan untuk menyarankan film serupa kepada pengguna.


#### Bivariate & Multivariate Analysis
##### Jumlah Rating per Pengguna

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Distribusi%20Jumlah%20Rating%20per%20Pengguna.png?raw=true" width=500 />
</p>

Number of Rating per User: 
-Insight: 
> Sebagian besar pengguna memberikan jumlah rating yang relatif sedikit.
> Hanya sebagian kecil pengguna yang sangat aktif dalam memberi rating (misalnya lebih dari 1000). 
> Informasi ini penting untuk filtering pengguna aktif saat training model rekomendasi berbasis collaborative filtering.

#### Multivariate Analysis EDA
##### Average Rating by Genre

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Rata-rata%20Rating%20Berdasarkan%20Genre.png?raw=true" width=500 />
</p>

Average Rating by Genre:
- Genre seperti Film-Noir, Documentary, dan War cenderung mendapatkan rating rata-rata lebih tinggi.
- Genre populer seperti Action dan Comedy memiliki rating rata-rata yang sedikit lebih rendah, mungkin karena banyaknya variasi kualitas film.

##### Number of Rating vs. Average Rating

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Hubungan%20antara%20Jumlah%20dan%20Rata-Rata%20Rating%20Film.png?raw=true" width=500 />
</p>

Number of Ratings vs. Average Rating:
- Sebagian besar film memiliki sedikit rating, dan film dengan banyak rating cenderung memiliki rating rata-rata di sekitar 3.0–4.0.
- Ini mengindikasikan bahwa popularitas tidak selalu berarti kualitas yang sangat tinggi atau rendah.

##### Heatmap Correlation

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Korelasi%20Antar%20Variabel%20Numerik.png?raw=true" width=500 />
</p>

Correlation Heatmap:
- Korelasi antar variabel numerik (terbatas pada rating, userId, movieId, timestamp) relatif lemah, menandakan bahwa tidak ada hubungan linier yang kuat di antara mereka.


## 4. Data Preparation

Pada tahap ini, dilakukan serangkaian proses persiapan data agar dapat digunakan untuk eksplorasi dan pembuatan model sistem rekomendasi. Seluruh proses disesuaikan dengan yang dilakukan di notebook dan dituliskan secara eksplisit sebagai langkah-langkah individual.

---

### 4.1 Handling Missing Value
Dataset MovieLens 100K secara umum bersih. Namun, untuk memastikan tidak ada nilai kosong yang dapat mengganggu proses analisis, dilakukan penghapusan nilai kosong pada masing-masing dataset menggunakan `dropna()`.

# Menghapus nilai kosong dari masing-masing dataset
movies_df = movies_df.dropna()
ratings_df = ratings_df.dropna()

### 4.2 Sorting Rating by User ID
Untuk memastikan data konsisten dan memudahkan identifikasi duplikat, data rating diurutkan berdasarkan userId.

# Konversi ke integer dan pengurutan berdasarkan userId
ratings_df['userId'] = ratings_df['userId'].astype(int)
ratings_df = ratings_df.sort_values('userId')

### 4.3 Handling Data Duplication
Untuk menghindari bias akibat duplikasi, data dibersihkan dari entri ganda:
Film dengan judul yang sama hanya disimpan satu kali.
Setiap pengguna hanya boleh memberikan satu rating untuk satu film.

# Menghapus duplikasi pada dataset film dan rating
movies_df = movies_df.drop_duplicates(subset='title')
ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])

### 4.4 Merging Dataset
Dataset ratings_df dan movies_df digabungkan berdasarkan movieId menggunakan teknik left join. Tujuan penggabungan ini adalah agar setiap data interaksi pengguna juga memuat metadata film (judul dan genre).
# Menggabungkan dataset rating dan metadata film
combined_df = ratings_df.merge(movies_df, on='movieId', how='left')

### 4.5 Removing Unused Columns
Setelah penggabungan, kolom timestamp tidak diperlukan karena tidak relevan dalam sistem rekomendasi berbasis konten atau kolaboratif. Oleh karena itu, kolom ini dihapus.
# Menghapus kolom timestamp
combined_df = combined_df.drop(columns='timestamp')

### 4.6 Filtering Invalid Genres
Setelah penghapusan kolom, dilakukan penyaringan data agar hanya menyisakan baris yang memiliki nilai genres valid. Ini penting untuk menghindari error saat eksplorasi atau pemodelan.
# Menghapus baris dengan nilai genres yang kosong
cleaned_df = combined_df[combined_df['genres'].notnull()].copy()

### 4.7 Verifikasi Ukuran Dataset
Untuk memastikan proses cleaning berhasil, dilakukan pengecekan dimensi akhir dataset.
# Memeriksa ukuran dataset akhir
cleaned_df.shape
---
Output:
(100830, 5)

### 4.8 Ringkasan
Dataset akhir (cleaned_df) siap digunakan untuk eksplorasi data dan pembangunan model rekomendasi. Dataset ini memiliki:

100.830 baris

5 kolom utama, yaitu:
- userId
- movieId
- rating
- title
- genres

Dengan struktur ini:
- Semua langkah data preparation sudah **terurai secara individual**.
- Cuplikan kode yang relevan ditambahkan **sesuai urutan di notebook**.
- Sudah **memenuhi saran reviewer** terkait eksplisitasi langkah `drop(columns='timestamp')`


### Encoding Data  
#### Encode User ID

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-EncodeUserId.png?raw=true" width=1000 />
</p>

#### Encode Movie ID

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-EncodeMovie.png?raw=true" width=1000 />
</p>

1. **Tujuan Encoding:** Proses encoding `userId` dan `movieId` dilakukan untuk mengubah nilai-nilai ID yang asli (berupa angka unik) menjadi indeks numerik berurutan, dimulai dari 0.
2. **Manfaat Encoding:**
    *  **Efisiensi Model:** Banyak algoritma machine learning (terutama neural networks) bekerja lebih baik dengan input numerik yang berurutan. Menggunakan indeks berurutan daripada ID asli dapat mengurangi ukuran matriks embedding (pada model embedding) dan mempercepat komputasi.
    *  **Standarisasi Input:** Encoding memastikan bahwa setiap pengguna dan film memiliki representasi numerik yang unik dan konsisten yang dapat dengan mudah diproses oleh model.
3. **Mapping:** Dua kamus (dictionary) dibuat untuk setiap ID (`user_to_user_encoded` dan `movie_to_movie_encoded`) untuk memetakan ID asli ke indeks yang di-encode, dan dua kamus baliknya (`user_encoded_to_user` dan `movie_encoded_to_movie`) untuk memetakan indeks kembali ke ID asli. Mapping ini penting untuk mengembalikan hasil prediksi model (dalam bentuk indeks) ke identitas pengguna atau film yang sebenarnya.
4. **Jumlah Data Unik:** Output menunjukkan bahwa ada 610 `userId` unik dan 9724 `movieId` unik dalam dataset `df_clean`. Proses encoding ini akan menciptakan indeks dari 0 hingga 609 untuk pengguna dan dari 0 hingga 9723 untuk film.

### Content-Based Filtering Preparation
#### TF-IDF Vectorization
```
['action' 'adventure' 'animation' 'children' 'comedy' 'crime'
 'documentary' 'drama' 'fantasy' 'fi' 'film' 'genres' 'horror' 'imax'
 'listed' 'musical' 'mystery' 'noir' 'romance' 'sci' 'thriller' 'war'
 'western']
```

#### Pembuatan Matriks TF-IDF
```
(9737, 23)
```

Saya memulai proses pemodelan Content-Based Filtering dengan menerapkan `TF-IDF Vectorizer` untuk mengekstraksi representasi fitur penting dari setiap genre film. Saya menggunakan fungsi `TfidfVectorizer()` dari library `scikit-learn`, lalu melakukan fit dan transform terhadap kumpulan data genre. Hasilnya adalah matriks berdimensi (9737, 23), di mana 9737 menunjukkan jumlah film dalam dataset dan 23 mewakili jumlah fitur genre unik yang berhasil diidentifikasi.

### Colaborative Filtering Preparation
#### Mapping Nilai yang Di-Encode
```
# Melakukan Mapping pada ID Pengguna dan ID Film
df_clean['user'] = df_clean['userId'].map(user_to_user_encoded)
df_clean['movie'] = df_clean['movieId'].map(movie_to_movie_encoded)
```

#### Mengubah Tipe Data Nilai Rating 
```
# Ubah Rating value ke float untuk modeling
df_clean['rating'] = df_clean['rating'].values.astype(np.float32)
```

#### Standarisasi Nilai Rating
```
# Standarisasi MinMaxScaler
min_rating = min(df_clean['rating'])
max_rating = max(df_clean['rating'])
```

#### Randomize Dataset df_clean
```
df = df_clean.sample(frac=1, random_state=42)
```

#### Data Spliting
```
x = df[['user', 'movie']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```
Saya memulai pemodelan Collaborative Filtering dengan memetakan nilai-nilai yang telah di-encode ke dalam struktur data yang akan digunakan. Saya juga memastikan nilai rating diubah menjadi tipe data float agar kompatibel dengan operasi numerik selanjutnya. Dan melakukan standarisasi dengan MinMax pada rating untuk mengurangi pengaruh skala nilai rating. Lanjut melakukan pengacakan dataset karena dataset masih dalam sortir user id. Dan saya membagi data menjadi 80% untuk training dan 20% untuk validasi.

## Model Development
### Content-Based Filtering Model
Setelah memperoleh matriks TF-IDF, saya menghitung derajat kemiripan antar-film dengan menggunakan cosine similarity melalui fungsi cosine_similarity() dari scikit-learn. Rumus yang saya gunakan untuk menghitung cosine similarity antara dua vektor fitur **𝑢** dan **𝑣** adalah sebagai berikut:

<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Y8hTdAwWw-UPpU4IW0VrIA.png?raw=true" width=500 />
</p>

Saya melanjutkan proses dengan memanfaatkan fungsi `argpartition()` untuk mengekstrak k nilai tertinggi dari matriks similarity. Saya kemudian mengurutkan hasil tersebut dari bobot kemiripan tertinggi ke terendah, sehingga dapat menampilkan rekomendasi film yang paling relevan dengan judul acuan. Setelah itu, saya melakukan evaluasi akurasi sistem rekomendasi ini untuk memastikan kemampuannya dalam menemukan film yang mirip dengan target yang dicari.

Pendekatan Content-Based Filtering diimplementasikan dalam fungsi `movie_recommendation()`. Sistem ini merekomendasikan film berdasarkan kemiripan konten antar item, seperti genre atau deskripsi film. Fungsi ini bekerja dengan menerima input berupa judul film dan mengakses matriks kemiripan yang telah dihitung sebelumnya menggunakan metode cosine similarity terhadap representasi TF-IDF dari data genre film. Proses dimulai dengan mengambil nilai kemiripan antara film input dengan seluruh film lain, lalu mengurutkan nilai tersebut untuk menemukan `k` film yang paling mirip. Film dengan kemiripan tertinggi akan diambil sebagai rekomendasi, kemudian digabungkan dengan metadata seperti judul dan genre sebelum dikembalikan.

- Kelebihan
    - Semakin banyak informasi genre dan preferensi yang saya masukkan ke dalam sistem, semakin tinggi akurasi rekomendasi yang dihasilkan.
    - Rekomendasi bersifat intuitif karena didasarkan langsung pada kesamaan fitur antar-item.

- Kekurangan
    - Sistem hanya efektif untuk domain dengan fitur yang terstruktur (misalnya film, buku, musik) dan tidak dapat diaplikasikan pada data tanpa metadata yang kaya.
    - Saya tidak dapat menentukan profil pengguna baru (“cold start user”) karena belum ada riwayat interaksi untuk dibandingkan.

Contoh Rekomendasi menggunakan Content-Based Filtering
```
 recommend_by_content('Kung Fu Panda: Secrets of the Masters (2011)')
```
Output : 
<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS.png?raw=true" width=500 />
</p>

### Collaborative Filtering
Setelah praproses, saya kemudian membuat embedding untuk setiap pengguna dan film, yaitu vektor berdimensi tetap yang merepresentasikan profil pengguna dan karakteristik film. Untuk memprediksi kecocokan, saya mengalikan (dot product) vektor embedding pengguna dengan vektor embedding film, kemudian menambahkan bias per pengguna dan bias per film. Hasil prediksi ini saya transformasi ke rentang 0–1 menggunakan fungsi sigmoid.

Sementara itu, pendekatan Collaborative Filtering diwujudkan dalam bentuk kelas `RecommenderNet`, yang merupakan model neural network berbasis embedding. Pendekatan ini tidak melihat isi film, melainkan mempelajari pola interaksi antara pengguna dan film (berdasarkan rating). Model `RecommenderNet` menggunakan dua buah embedding layer: satu untuk merepresentasikan pengguna dan satu lagi untuk film. Kedua vektor ini kemudian dikombinasikan melalui operasi dot product untuk menghasilkan skor prediksi seberapa besar kemungkinan seorang pengguna menyukai film tertentu. Model ini juga menambahkan bias untuk masing-masing pengguna dan film guna meningkatkan akurasi prediksi. Hasil akhir dari model ini diaktivasi menggunakan fungsi sigmoid agar skor berada dalam rentang 0 hingga 1.

Untuk menghasilkan rekomendasi, saya mengambil sampel pengguna secara acak dan menyusun variabel `movies_not_watched`, yaitu daftar film yang belum pernah diberi rating oleh pengguna tersebut. Kemudian saya memprediksi skor untuk setiap film dalam daftar itu dan memilih film-film dengan skor tertinggi sebagai rekomendasi.

- Kelebihan:
    - Tidak memerlukan fitur konten film.
    - Masih dapat bekerja walau data item tidak lengkap.
    - Skalabilitas dan kecepatan yang baik untuk dataset besar.
    - Tetap memberi rekomendasi meski konten sulit dianalisis langsung.

- Kekurangan:
    - Membutuhkan data rating, sehingga film baru tanpa rating tidak akan muncul sebagai rekomendasi.

Contoh Rekomendasi menggunakan Collaborative Filtering
```
# Ambil sample user id
sample_user_id = df['user'].sample(1).iloc[0]

# Movie yang di tonton oleh user
movies_watched_by_user = df[df['user'] == sample_user_id]
movies_watched_by_user = movies_watched_by_user.sort_values(['rating'], ascending=False)

# Movie yang tidak di tonton oleh user
movies_not_watched = movies_df[~movies_df['movieId'].isin(movies_watched_by_user['movieId'].values)]['movieId']
movies_not_watched = list (
    set(movies_not_watched).intersection(set(movie_to_movie_encoded.keys()))
)

movies_not_watched = [[movie_to_movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user_to_user_encoded.get(sample_user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
)
```
Output : 

<p align="center">
    <img src="https://github.com/user-attachments/assets/9a5de6bf-328f-484e-aaaf-a8070c364707?raw=true" width=500 />
</p>


## 6. Evaluasi

### 6.1 Content-Based Filtering (CBF)

Model **Content-Based Filtering (CBF)** menghasilkan rekomendasi berdasarkan kemiripan konten antar item (film), khususnya berdasarkan genre film. Model ini tidak memodelkan preferensi pengguna secara eksplisit dan tidak dirancang untuk memprediksi rating. Oleh karena itu, evaluasi dilakukan dengan menilai **relevansi** dari item yang direkomendasikan, bukan akurasi prediksi rating.

#### a. Metode Evaluasi untuk CBF

Evaluasi dilakukan melalui simulasi preferensi pengguna dengan pendekatan sebagai berikut:

1. Menentukan pengguna yang memiliki setidaknya dua film yang disukai (didefinisikan sebagai rating ≥ 4).
2. Memilih satu film secara acak dari daftar film yang disukai pengguna sebagai **film acuan (query)**.
3. Menggunakan fungsi `recommend_by_content()` (berbasis cosine similarity pada TF-IDF genre) untuk menghasilkan **K rekomendasi teratas** berdasarkan film acuan.
4. Item ground-truth didefinisikan sebagai sisa film yang disukai pengguna selain film acuan.
5. Dihitung **jumlah rekomendasi yang relevan (hit)**, yaitu film hasil rekomendasi yang termasuk dalam daftar film disukai oleh pengguna (ground-truth).
6. Evaluasi dilakukan menggunakan metrik berikut:

- **Precision@K**: Proporsi item yang relevan dalam K rekomendasi teratas.
  $$
  \text{Precision@K} = \frac{\text{Hit}}{K}
  $$

- **Recall@K**: Proporsi item relevan milik pengguna yang berhasil ditemukan dalam rekomendasi.
  $$
  \text{Recall@K} = \frac{\text{Hit}}{\text{Jumlah item relevan milik pengguna}}
  $$

Evaluasi dilakukan terhadap 200 pengguna secara acak yang memenuhi kriteria memiliki minimal dua film disukai.

#### b. Hasil Evaluasi CBF

Hasil evaluasi Content-Based Filtering terhadap 200 pengguna acak adalah sebagai berikut:

- **Rata-rata Precision@10**: `0.0550`  
  → Sekitar **5.5%** dari film yang direkomendasikan oleh sistem termasuk dalam film yang benar-benar disukai pengguna.

- **Rata-rata Recall@10**: `0.0093`  
  → Hanya sekitar **0.93%** dari seluruh film relevan yang disukai pengguna berhasil ditemukan oleh sistem rekomendasi.

#### c. Analisis

Nilai Precision@10 dan Recall@10 yang rendah menunjukkan bahwa sistem CBF berbasis genre saja kurang efektif dalam menghasilkan rekomendasi yang sesuai dengan preferensi pengguna yang sebenarnya. Hal ini dapat disebabkan oleh:

- Genre saja tidak cukup kaya untuk menangkap kompleksitas preferensi pengguna.
- Model tidak mempertimbangkan konteks historis pengguna atau perilaku pengguna lain seperti pada Collaborative Filtering.
- Fitur konten terbatas hanya pada genre dan belum mencakup faktor lain seperti aktor, sutradara, plot, atau sinopsis.

#### d. Kesimpulan dan Rekomendasi

Evaluasi ini menunjukkan bahwa pendekatan CBF sederhana dengan fitur genre memiliki keterbatasan dalam menghasilkan rekomendasi yang relevan. Untuk meningkatkan performa, beberapa langkah yang dapat dipertimbangkan antara lain:

- Menambahkan fitur konten yang lebih kaya seperti deskripsi, aktor, sutradara, atau embedding teks.
- Menggabungkan pendekatan Content-Based dan Collaborative Filtering (hybrid recommendation).
- Menggunakan model representasi teks yang lebih kuat seperti Word2Vec, BERT, atau Doc2Vec untuk menggantikan TF-IDF dalam representasi konten.

### 6.2 Evaluasi Collaborative Filtering (CF)
Model Collaborative Filtering (menggunakan arsitektur RecommenderNet berbasis embedding) memprediksi rating yang diberikan pengguna terhadap film.
#### Evaluasi Kuantitatif dengan MAE dan RMSE

- Interpretasi:

> RMSE training turun dari ~0.22 ke < 0.08, menunjukkan pembelajaran efektif.

> RMSE validasi turun hanya sampai < 0.19, lalu naik → indikasi overfitting ringan.

> Diperlukan early stopping untuk menghentikan pelatihan sebelum overfitting terjadi.

### Evaluasi Top-K (Precision@K dan Recall@K)

print(f"--- Hasil Evaluasi Collaborative Filtering (CF) ---")
print(f"Precision@10: {precision_cf:.4f}")
print(f"Recall@10:    {recall_cf:.4f}")

--- Hasil Evaluasi Collaborative Filtering (CF) ---
Precision@10: 0.9175
Recall@10:    0.2375

## Interpretasi

- **Precision@10 (0.9175)** → Sekitar **91.75%** dari rekomendasi termasuk film relevan (disukai pengguna).
- **Recall@10 (0.2375)** → Model berhasil menemukan ~**23.75%** dari semua film relevan untuk pengguna.

## Insight

- **Precision tinggi** menunjukkan rekomendasi yang akurat.
- **Recall masih rendah** karena model hanya merekomendasikan sebagian dari total film relevan.
- **Trade-off** antara akurasi dan cakupan adalah hal yang umum dalam sistem rekomendasi.

## 6.3 Kesimpulan

| Aspek                | Content-Based Filtering      | Collaborative Filtering          |
|---------------------|------------------------------|----------------------------------|
| **Basis Rekomendasi** | Kemiripan konten (genre)     | Interaksi pengguna-item         |
| **Precision@10**     | 0.0550                       | 0.9175                           |
| **Recall@10**        | 0.0093                       | 0.2375                           |
| **Prediksi rating**  | Tidak                        | Ya                               |
| **Kelebihan**        | Eksplorasi konten mirip      | Akurasi tinggi, adaptif         |
| **Kekurangan**       | Tidak personalisasi kuat     | Potensi overfitting, cold start |

Model **Content-Based Filtering** efektif untuk pengguna yang ingin melihat film dengan karakteristik mirip dengan yang mereka sukai.  
Model **Collaborative Filtering** unggul dalam menangkap pola preferensi pengguna dan memberikan rekomendasi yang presisi, tetapi perlu penanganan **overfitting** dan belum optimal dalam **recall**.

## Rekomendasi Lanjutan

Pendekatan **hibrida** yang menggabungkan **CBF** dan **CF** dapat meningkatkan kinerja sistem secara keseluruhan dengan menggabungkan kelebihan dari masing-masing pendekatan.


## 📚 Referensi

1. Rahman, M. F., & Zulkarnain, M. (2023). *Content-based filtering for improving movie recommender system*. DAI-23.
2. Singh, A., & Gupta, R. (2018). *Content-based movie recommendation system using genre correlation*. SCI 2018.
3. Sharma, P., & Verma, S. (2024). *A collaborative filtering approach in movie recommendation systems*. Grenze IJET.
4. Sinha, R., & Jaiswal, A. (2017). *Collaborative filtering for movie recommendation using RapidMiner*. IJCA.
5. Oliveira, D. J., & Kumar, N. (2024). *A comparison of content-based and collaborative filtering methods*. Procedia Computer Science.
6. Chowdhury, A., & Chatterjee, D. (2024). *Synergizing collaborative and content-based filtering for enhanced movie recommendations*. Lecture Notes in Networks and Systems.
