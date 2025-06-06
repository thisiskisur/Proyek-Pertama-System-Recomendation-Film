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

## Data Understanding
### Sumber Data
Dalam proyek ini, saya memanfaatkan dataset MovieLens yang tersedia di platform Kaggle. Dataset ini menyimpan data penilaian film dari para pengguna, yang mencakup beberapa informasi penting seperti userId, movieId, title, genres, dan rating. Secara keseluruhan, terdapat sebanyak 100.836 entri rating yang diberikan oleh 610 pengguna terhadap 9.742 judul film yang berbeda. Dataset ini sangat sesuai untuk digunakan dalam pengembangan dan pengujian sistem rekomendasi, khususnya dengan pendekatan Content-Based Filtering maupun Collaborative Filtering. Dataset lengkapnya bisa diakses melalui tautan berikut:
[MovieLens Dataset – Kaggle](https://www.kaggle.com/datasets/sunilgautam/movielens)

### Deskripsi Variabel
#### Dataset Movies
```
Info Dataset Movies : 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9742 entries, 0 to 9741
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  9742 non-null   int64 
 1   title    9742 non-null   object
 2   genres   9742 non-null   object
dtypes: int64(1), object(2)
memory usage: 228.5+ KB
```
##### Movies Missing Values
Variabel | Missing Value
----------|----------
Movie ID | 0
Title  | 0
Genres  | 0

Tidak terdapat Missing Value pada dataset Movies

##### Movies Duplicated
```
Jumlah Duplikasi Data Movie : 0
```

##### Movies Unique
Variabel | Movie Unique
----------|----------
Movie ID | 9742
Title  | 9737
Genres  | 951

Terdapat sebanyak `9742` total Movie ID yang unik, dan `9737` judul film berbeda, dan `951` macam Genre pada dataset ini.

#### Dataset Ratings
Variabel | Keterangan
----------|----------
User ID | Identify User ID.
Movie ID  | Unique ID from Movie.
Rating | Rating Out of 5 This User Has Assigned.
Timestamp | Datetime ratings created.

```
Info Dataset Ratings : 

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100836 entries, 0 to 100835
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype  
---  ------     --------------   -----  
 0   userId     100836 non-null  int64  
 1   movieId    100836 non-null  int64  
 2   rating     100836 non-null  float64
 3   timestamp  100836 non-null  int64  
dtypes: float64(1), int64(3)
memory usage: 3.1 MB
```
##### Ratings Missing Values
Variabel | Missing Value
----------|----------
User ID | 0
Movie ID  | 0
Rating  | 0
Timestamp | 0

Tidak terdapat Missing Value pada dataset Ratings

##### Ratings Duplicated
```
Jumlah Duplikasi Data Rating : 0
```

##### Ratings Unique
Variabel | Movie Unique
----------|----------
User ID | 610
Movie  | 9724
Rating  | 10
Timestamp | 85043

Terdapat sebanyak `610` total User ID yang unik, dan `9724` Movie ID yang berbeda yang telah di rating oleh user, dan `10` data unik pada Rating mengindikasikan rating value (0.5 - 5.0), dan `85043` data unik pada timestamp.

### Penggabungan Dataset dan Pembersihan Data / Merge Dataset
```
# Menggabungkan dataframe movie dan rating
all_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')
all_df
```

- Penjelasan Teknis::
    >  Penjelasan Teknis:
        - ratings_df sebagai DataFrame utama (kiri) berisi informasi siapa yang memberikan rating ke film apa.
        - movies_df sebagai DataFrame tambahan (kanan) berisi metadata film (seperti title dan genres).
        - on='movieId' artinya penggabungan dilakukan berdasarkan nilai unik movieId.
        - how='left' memastikan semua baris pada ratings_df tetap dipertahankan, meskipun mungkin ada movieId yang tidak ditemukan pada movies_df.
        Catatan:
        Jika ada movieId di ratings_df yang tidak ditemukan di movies_df, maka kolom-kolom tambahan dari movies_df (seperti title dan genres) akan bernilai NaN. Namun, ini jarang terjadi dalam dataset bersih seperti MovieLens.
        Output:
        Perintah combined_df.info() akan menampilkan struktur dari dataset hasil gabungan, termasuk:
        - Jumlah baris dan kolom
        Tipe data per kolom
        Jumlah nilai non-null di setiap kolom

```
Struktur Dataset Gabungan:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100836 entries, 0 to 100835
Data columns (total 6 columns):
 #   Column     Non-Null Count   Dtype  
---  ------     --------------   -----  
 0   userId     100836 non-null  int64  
 1   movieId    100836 non-null  int64  
 2   rating     100836 non-null  float64
 3   timestamp  100836 non-null  int64  
 4   title      100830 non-null  object 
 5   genres     100830 non-null  object 
dtypes: float64(1), int64(3), object(2)
memory usage: 4.6+ MB
```

Manfaat penggabungan ini adalah untuk menambahkan informasi film (judul dan genre) ke data rating. Dengan begitu, kita bisa:

- Menampilkan nama film, bukan hanya movieId.
- Memperkaya Data Interaksi Pengguna:
- Menganalisis rating berdasarkan genre atau judul film.
- Membuat sistem rekomendasi yang lebih informatif dan user-friendly.
Tanpa penggabungan ini, data rating hanya berisi ID angka tanpa konteks

- Analisis dan Visualisasi Lebih Komprehensif:
    Setelah merge, kita bisa menjawab pertanyaan seperti:
    - Film apa yang paling banyak di-rating pengguna?
    - Genre mana yang paling populer?
    - Bagaimana distribusi rating suatu film tertentu?

### Exploratory Data Analysis (EDA)
#### Univariate Analysis – Distribusi Nilai Rating
##### Distribusi Nilai Rating

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Distribusi%20Rating%20Film%20oleh%20Pengguna.png" width=500 />
</p>

Distribusi Nilai Rating:
- Interpretasi: Rating paling umum diberikan pengguna berada di angka 3.0 dan 4.0.
- Relatif sedikit pengguna yang memberi rating 1.0 atau 5.0, artinya pengguna cenderung menilai secara moderat.
- Distribusi ini menunjukkan bahwa data tidak terlalu condong ke rating tinggi atau rendah secara ekstrem.

##### Genre Film yang Paling Sering Muncul

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-20%20Genre%20Film%20Paling%20Umum%20dalam%20Dataset.png" width=500 />
</p>


Top 20 Genre Film yang Paling Sering Muncul: 
- Insight: 
    >Genre seperti Drama, Comedy, dan Action mendominasi daftar genre yang paling sering muncul
    >Informasi ini akan berguna dalam Content-Based Recommendation, di mana kesamaan genre digunakan untuk menyarankan film serupa kepada pengguna.


#### Bivariate & Multivariate Analysis
##### Jumlah Rating per Pengguna

<p align="center">
    <img src="https://github.com/thisiskisur/Proyek-Pertama-System-Recomendation-Film/blob/main/Screenshot/SS-Distribusi%20Jumlah%20Rating%20per%20Pengguna.png" width=500 />
</p>

Number of Rating per User: 
-Insight: 
> Sebagian besar pengguna memberikan jumlah rating yang relatif sedikit.
> Hanya sebagian kecil pengguna yang sangat aktif dalam memberi rating (misalnya lebih dari 1000). 
> Informasi ini penting untuk filtering pengguna aktif saat training model rekomendasi berbasis collaborative filtering.

#### Multivariate Analysis EDA
##### Average Rating by Genre

<p align="center">
    <img src="https://github.com/user-attachments/assets/a4db2830-86aa-4909-8a88-eb0308304e91" width=500 />
</p>

Average Rating by Genre:
- Genre seperti Film-Noir, Documentary, dan War cenderung mendapatkan rating rata-rata lebih tinggi.
- Genre populer seperti Action dan Comedy memiliki rating rata-rata yang sedikit lebih rendah, mungkin karena banyaknya variasi kualitas film.

##### Number of Rating vs. Average Rating

<p align="center">
    <img src="https://github.com/user-attachments/assets/cd425409-dba7-49b1-b043-67a9ad12034f" width=500 />
</p>

Number of Ratings vs. Average Rating:
- Sebagian besar film memiliki sedikit rating, dan film dengan banyak rating cenderung memiliki rating rata-rata di sekitar 3.0–4.0.
- Ini mengindikasikan bahwa popularitas tidak selalu berarti kualitas yang sangat tinggi atau rendah.

##### Heatmap Correlation

<p align="center">
    <img src="https://github.com/user-attachments/assets/e74fa240-37f9-4ece-b4b2-94230e374e52" width=500 />
</p>

Correlation Heatmap:
- Korelasi antar variabel numerik (terbatas pada rating, userId, movieId, timestamp) relatif lemah, menandakan bahwa tidak ada hubungan linier yang kuat di antara mereka.

## Data Preparation
### Handling Missing Value
Penanganan missing value yang saya lakukan yaitu dengan melakukan drop data. Tetapi karena dataset yang digunakan cukup bersih, missing value hanya terdapat ketika proses penggabungan dataset.
```
# Menangani Missing Value
movies_df.dropna(axis=0, inplace=True)
ratings_df.dropna(axis=0, inplace=True)
```

### Sorting Rating by User ID
Untuk memudahkan analisis, saya melakukan sorting user ID pada data rating. Ini akan membantu saya untuk memahami perilaku pengguna secara lebih baik. Pengurutan data rating berdasarkan ID Pengguna agar mempermudah dalam melakukan penghapusan data duplikat nantinya.
```
# Sortir User ID  
ratings_df = ratings_df.sort_values('userId').astype('int')
```

### Handling Data Duplication
Untuk menghindari penghitungan rating yang tidak tepat, saya melakukan penghapusan data duplikat agar tidak terjadi bias pada data nantinya. Penghapusan data duplikat dilakukan dengan menggunakan ID pengguna dan ID film sebagai kunci. Dengan demikian, saya dapat memastikan bahwa setiap pengguna hanya memiliki satu rating per film.
```
# Menangani Duplikasi data 
movies_df.drop_duplicates(subset=['title'], keep='first', inplace=True)
ratings_df.drop_duplicates(subset=['userId','movieId'], keep='first', inplace=True)
```

### Cleaning Data

<p align="center">
    <img src="https://github.com/user-attachments/assets/76cc9177-7419-4d14-b8fd-4df5ac9c0a9f" width=1000 />
</p>

```
# Membuat Dataframe baru untuk Dataset Clean
df_clean = all_df[~pd.isnull(all_df['genres'])].copy()
df_clean.shape
```
Output
```
(100830, 5)
```

Dataset `df_clean` adalah hasil copy() dari proses merged `all_df` dimana dilakukan drop column Timestamp di dataset gabungan serta tanpa nilai null pada column genres.

Dan di dapat sebanyak `100830` baris data dan `5` buah kolom pada dataset `df_clean`.

### Encoding Data  
#### Encode User ID

<p align="center">
    <img src="https://github.com/user-attachments/assets/744dc5ee-569b-4dfe-ab64-515b1d9b002e" width=1000 />
</p>

#### Encode Movie ID

<p align="center">
    <img src="https://github.com/user-attachments/assets/d8137541-6644-420b-a823-aa0b640f84e1" width=1000 />
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
    <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Y8hTdAwWw-UPpU4IW0VrIA.png" width=500 />
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
movie_recomend = movie_recommendation('Kung Fu Panda: Secrets of the Masters (2011)')
movie_recomend
```
Output : 
<p align="center">
    <img src="https://github.com/user-attachments/assets/7650d5be-f067-4b0f-9437-8c33ac95c501" width=500 />
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
    <img src="https://github.com/user-attachments/assets/9a5de6bf-328f-484e-aaaf-a8070c364707" width=500 />
</p>


## Evaluation
### Content-Based Filtering
Untuk mengevaluasi model Content-Based Filtering, saya menggunakan metrik yang umum digunakan dalam sistem rekomendasi, yaitu Precision@K, Recall@K.

<p align="center">
    <img src="https://i.sstatic.net/W8rc6.png" width=500 />
</p>

```
precision, recall = precision_recall_at_k(model, x_val, y_val, k=10, threshold=0.5)
```
Output :
```
Precision@10: 0.9079
Recall@10:    0.2330
```

Insight dari hasil evaluasi:
1. Precision@10 yang tinggi (0.9079) menunjukkan bahwa dari 10 rekomendasi teratas yang diberikan oleh model, sekitar 90.79% di antaranya adalah film yang relevan atau disukai oleh pengguna (berdasarkan threshold rating 0.5). Ini berarti ketika model merekomendasikan film, kemungkinan besar film tersebut memang sesuai dengan preferensi pengguna.
2. Recall@10 yang relatif rendah (0.2330) menunjukkan bahwa model hanya mampu menangkap sekitar 23.30% dari total film relevan yang sebenarnya ada untuk pengguna dalam validation set. Ini berarti ada banyak film relevan lainnya yang tidak masuk dalam daftar 10 rekomendasi teratas model.

Kesimpulan:

Model ini sangat baik dalam memberikan rekomendasi yang akurat (tinggi Precision), artinya rekomendasi yang ditampilkan punya probabilitas tinggi disukai pengguna. Namun, model belum mampu menemukan sebagian besar film relevan yang sebenarnya disukai pengguna (rendah Recall).

### Collaborative Filtering
#### Mean Absolute Error
Untuk mengukur akurasi model, saya menggunakan Mean Absolute Error (MAE). MAE adalah rata-rata dari nilai absolut dari perbedaan antara nilai prediksi dan nilai sebenarnya. Nilai MAE yang lebih kecil menunjukkan model yang lebih akurat. 

<p align="center">
    <img src="https://github.com/user-attachments/assets/6f8c0e25-ee3d-41bb-bf43-6d2759738b6b" width=500 />
</p>

Mean Absolute Error (MAE):
- Nilai MAE pada Data Training:
> Dari grafik MAE, saya mengamati bahwa nilai MAE pada data training (garis biru) mengalami penurunan konsisten sepanjang proses pelatihan. Hal ini menunjukkan bahwa model yang saya latih semakin baik dalam memprediksi rating dengan kesalahan absolut rata-rata yang semakin kecil. Penurunan yang stabil juga mengindikasikan bahwa proses pembelajaran berlangsung secara efektif tanpa gangguan besar seperti fluktuasi atau divergensi.

- Nilai MAE pada Data Validasi:
> Nilai MAE pada data validasi (garis oranye) juga menunjukkan tren penurunan di awal pelatihan, tetapi cenderung stagnan mulai dari sekitar epoch ke-2 hingga akhir. Meskipun tidak terjadi peningkatan performa yang signifikan setelah titik tersebut, kestabilan nilai validasi ini mengindikasikan bahwa model tidak mengalami overfitting secara drastis, meskipun terdapat indikasi bahwa peningkatan performa terhadap data yang tidak terlihat menjadi terbatas setelah beberapa epoch awal.

- Indikasi Konvergensi:
> Perbedaan antara MAE training dan validasi yang semakin melebar secara perlahan dapat menjadi sinyal awal adanya potensi overfitting, namun grafik menunjukkan bahwa nilai validasi tetap dalam rentang yang relatif stabil. Ini menunjukkan bahwa performa model pada data validasi masih dapat diterima meskipun laju peningkatannya melambat.

#### Root Mean Squared Error
Untuk mengukur akurasi model, saya menggunakan Root Mean Squared Error (RMSE) yang merupakan hasil evaluasi model Collaborative Filtering RecommenderNet menggunakan metrik Root Mean Squared Error (RMSE) pada data pelatihan (RMSE) dan data validasi (Val RMSE) selama proses training. RMSE merupakan metrik yang umum digunakan dalam sistem rekomendasi untuk mengukur selisih rata-rata kuadrat antara rating prediksi dan rating aktual; semakin kecil nilai RMSE, semakin baik kinerja model.

<p align="center">
    <img src="https://github.com/user-attachments/assets/ff0375fd-4307-46af-96d9-b62e354ed80b" width=500 />
</p>

Pada grafik, terlihat bahwa nilai RMSE pada data pelatihan mengalami penurunan yang konsisten dari awal hingga akhir pelatihan. Ini menunjukkan bahwa model berhasil mempelajari pola dari data pelatihan dengan baik. Di sisi lain, nilai RMSE pada data validasi juga menurun secara signifikan pada beberapa epoch awal, tetapi kemudian cenderung stabil dan mengalami sedikit fluktuasi setelah sekitar epoch ke-7. Ini adalah indikasi bahwa model mencapai titik stabil dalam pembelajaran dan menghindari overfitting. Model `RecommenderNet` dibangun dengan ukuran `embedding = 50` dan dioptimalkan menggunakan `Adam Optimizer` dengan `learning rate 0.001`. Proses pelatihan dijalankan hingga maksimum `100 epoch` dengan early stopping `callback` yang dikonfigurasi untuk:
- `patience=10`: pelatihan akan dihentikan jika tidak ada perbaikan RMSE validasi selama 10 epoch berturut-turut.
- `min_delta=0.0001`: perubahan minimal yang dianggap sebagai perbaikan.
- `restore_best_weights=True`: model akan mengembalikan bobot terbaik berdasarkan kinerja validasi

Berdasarkan grafik, model `RecommenderNet` berhasil belajar dengan baik dari data pelatihan tanpa menunjukkan gejala overfitting yang signifikan. Penggunaan `callback` early stopping secara efektif menghentikan pelatihan pada waktu yang tepat dan memastikan bobot terbaik digunakan yang kemungkinan besar terjadi sebelum epoch ke-10. Nilai akhir RMSE pada data validasi sekitar `0.188` menjadi indikator kinerja model dalam memberikan prediksi rating yang mendekati nilai aktual pengguna, yang menunjukkan bahwa model cukup baik dalam memahami preferensi pengguna berdasarkan data interaksi sebelumnya.

Root Mean Squared Error (RMSE):
- Nilai RMSE pada Data Training:
> Berdasarkan grafik RMSE, saya melihat penurunan tajam pada nilai RMSE training (garis biru) selama epoch awal, yang kemudian diikuti oleh penurunan bertahap hingga akhir pelatihan. Ini menunjukkan bahwa model mampu menyesuaikan prediksinya terhadap data training dengan mengurangi deviasi kuadrat rata-rata secara progresif.

- Nilai RMSE pada Data Validasi:
> Nilai RMSE validasi (garis oranye) menurun pada awal pelatihan hingga sekitar epoch ke-7, lalu cenderung stabil dan sedikit fluktuatif hingga akhir pelatihan. Tidak terlihat adanya penurunan signifikan setelah titik tersebut, yang mengindikasikan bahwa model telah mencapai titik optimum dalam hal generalisasi pada data validasi. Namun, fluktuasi kecil setelah epoch ke-7 menunjukkan bahwa model mulai menghadapi batas kemampuannya dalam mengurangi kesalahan pada data yang tidak dilatih.

- Konvergensi dan Early Stopping:
> Pola yang tampak pada grafik RMSE validasi mendukung penggunaan strategi Early Stopping. Jika callback ini digunakan, maka kemungkinan besar pelatihan dihentikan saat tidak ada peningkatan signifikan pada performa validasi. Dengan demikian, saya dapat menghindari pelatihan berlebih dan menjaga generalisasi model tetap baik.

## Kesimpulan:

Berdasarkan analisis terhadap grafik MAE dan RMSE, saya menyimpulkan bahwa model Collaborative Filtering yang dibangun menunjukkan performa pelatihan yang baik dan generalisasi yang memadai. Penurunan konsisten pada data training menunjukkan bahwa model berhasil mempelajari pola dari data, sedangkan nilai validasi yang relatif stabil menunjukkan bahwa overfitting masih dalam batas yang dapat diterima. Penggunaan teknik Early Stopping sangat membantu dalam memilih parameter model terbaik. Nilai MAE dan RMSE akhir dapat dijadikan indikator akurasi model dalam memprediksi rating, masing-masing dari perspektif kesalahan absolut dan kesalahan kuadrat rata-rata.

## Reference
1. Rahman, M. F., & Zulkarnain, M. (2023). Content-based filtering for improving movie recommender system. Proceedings of the International Conference on Data Analytics and Intelligence (DAI-23), 124–130.

2. Singh, A., & Gupta, R. (2018). Content-based movie recommendation system using genre correlation. Proceedings of the Second International Conference on SCI 2018, 2, 88–93.

3. Sharma, P., & Verma, S. (2024). A collaborative filtering approach in movie recommendation systems. Grenze International Journal of Engineering and Technology, 10(2), 55–62.

4. Sinha, R., & Jaiswal, A. (2017). Collaborative filtering for movie recommendation using RapidMiner. International Journal of Computer Applications, 169(6), 1–5.

5. Oliveira, D. J., & Kumar, N. (2024). A comparison of content-based and collaborative filtering methods for movie recommendation systems. Procedia Computer Science, 227, 1012–1018.

6. Chowdhury, A., & Chatterjee, D. (2024). Synergizing collaborative and content-based filtering for enhanced movie recommendations. Lecture Notes in Networks and Systems, 831, 33–42.
