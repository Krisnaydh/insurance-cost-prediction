# Medical Insurance Cost Prediction

Project machine learning kedua saya: memprediksi biaya premi asuransi kesehatan individu (`charges`) menggunakan algoritma regresi berbasis *tree-based ensemble*.

## Dataset
[Link ke dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Ringkasan Proses
- **EDA & Visualisasi:** Mengevaluasi distribusi variabel target (`charges`) yang memiliki ekor panjang ke kanan (*right-skewed*) serta menganalisis korelasi antar fitur.
- **Wawasan Utama:** Status perokok (`smoker`) teridentifikasi sebagai faktor pembeda paling signifikan yang menaikkan biaya asuransi secara eksplosif.
- **Preprocessing:** Melakukan pemisahan *train/validation split* sebelum *encoding* variabel kategorikal (`sex`, `smoker`, `region`) untuk menghindari *data leakage*.
- **Modeling:** Membandingkan performa model **RandomForestRegressor** dan **XGBoostRegressor** yang tangguh terhadap data bertipe *skewed*.

## Hasil Utama
Model ensemble **Random Forest** + **XGBoost** memprediksi biaya asuransi dengan R² 0,876 dan galat rata-rata sekitar 19% pada data uji, dengan status perokok sebagai faktor yang paling menentukan.

## Tools
Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
