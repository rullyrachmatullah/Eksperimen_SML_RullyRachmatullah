import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(path_raw, path_output):
    #Fungsi untuk melakukan preprocessing data secara otomatis dan menyimpan dataset yang siap digunakan untuk training.

    # 1. Load dataset
    df = pd.read_csv(path_raw)

    # 2. Drop duplikasi
    df = df.drop_duplicates()

    # 3. Encoding data kategorikal
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 4. Standarisasi fitur numerik
    num_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Simpan dataset hasil preprocessing
    os.makedirs(os.path.dirname(path_output), exist_ok=True)
    df.to_csv(path_output, index=False)

    return df

if __name__ == "__main__":
    preprocess_data(
        path_raw="../data_raw/med-insurance.csv",
        path_output="cleaning/med_clean.csv"
    )
