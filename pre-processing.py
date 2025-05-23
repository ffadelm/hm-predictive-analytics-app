import pandas as pd
import numpy as np
from datetime import datetime
import os

# Baca file CSV
def load_data(file_path):
    try:
        # Menggunakan pandas untuk membaca data
        df = pd.read_csv(file_path)
        print(f"Data berhasil dimuat: {df.shape[0]} baris dan {df.shape[1]} kolom")
        return df
    except Exception as e:
        print(f"Error saat membaca file: {e}")
        return None

# Fungsi untuk mengecek dan menampilkan informasi awal dataset
def explore_data(df):
    print("\n=== INFORMASI DATASET ===")
    print("\nLima baris pertama:")
    print(df.head())
    
    print("\nInformasi tipe data:")
    print(df.info())
    
    print("\nStatistik deskriptif:")
    print(df.describe())
    
    print("\nCek nilai yang hilang (missing values):")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "Tidak ada nilai yang hilang")
    
    print("\nCek duplikasi data:")
    duplicates = df.duplicated().sum()
    print(f"Jumlah data duplikat: {duplicates}")
    
    # Cek nilai unik per kolom kategorikal
    print("\nNilai unik untuk kolom kategorikal:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} nilai unik")
    
    return missing_values, duplicates

# Fungsi untuk preprocessing data sesuai dengan skema star
def preprocess_data(df):
    # Buat salinan data untuk preprocessing
    processed_df = df.copy()
    
    # 1. Konversi format tanggal
    print("\n=== PREPROCESSING DATA ===")
    print("1. Konversi format tanggal...")
    processed_df['Order Date'] = pd.to_datetime(processed_df['Order Date'])
    
    # 2. Tambahkan kolom yang berguna untuk analisis waktu
    print("2. Menambahkan kolom waktu...")
    processed_df['Year'] = processed_df['Order Date'].dt.year
    processed_df['Month'] = processed_df['Order Date'].dt.month
    processed_df['Month Name'] = processed_df['Order Date'].dt.strftime('%B')
    processed_df['Quarter'] = processed_df['Order Date'].dt.quarter
    processed_df['Day'] = processed_df['Order Date'].dt.day
    
    # 3. Cek dan tangani outlier pada kolom numerik
    print("3. Memeriksa outlier...")
    numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit']
    
    for col in numeric_cols:
        Q1 = processed_df[col].quantile(0.25)
        Q3 = processed_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identifikasi outlier (di luar 1.5*IQR dari Q1 atau Q3)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)).sum()
        print(f"   - Kolom {col}: {outliers} outlier terdeteksi")
    
    # 4. Buat dimensi date sesuai dengan skema
    print("4. Membuat dimension_date...")
    # Membuat dataframe dengan tanggal unik
    unique_dates = processed_df['Order Date'].unique()
    
    dimension_date = pd.DataFrame({
        'order_date': unique_dates
    })
    
    # Tambahkan kolom yang diperlukan
    dimension_date['date_id'] = range(1, len(dimension_date) + 1)
    dimension_date['day'] = dimension_date['order_date'].dt.day
    dimension_date['month'] = dimension_date['order_date'].dt.month
    dimension_date['month_name'] = dimension_date['order_date'].dt.strftime('%B')
    dimension_date['quarter'] = dimension_date['order_date'].dt.quarter
    dimension_date['year'] = dimension_date['order_date'].dt.year
    
    # 5. Buat dimensi product sesuai dengan skema
    print("5. Membuat dimension_product...")
    dimension_product = processed_df[['Product ID', 'Category', 'Sub-Category']].drop_duplicates().reset_index(drop=True)
    dimension_product.columns = ['product_id', 'category', 'sub_category']
    
    # 6. Buat dimensi location sesuai dengan skema
    print("6. Membuat dimension_location...")
    location_data = processed_df[['Country', 'City', 'State', 'Region']].drop_duplicates().reset_index(drop=True)
    location_data['location_id'] = range(1, len(location_data) + 1)
    
    dimension_location = pd.DataFrame({
        'location_id': location_data['location_id'],
        'country': location_data['Country'],
        'city': location_data['City'],
        'state': location_data['State'],
        'region': location_data['Region']
    })
    
    # 7. Buat dimensi order sesuai dengan skema
    print("7. Membuat dimension_order...")
    order_data = processed_df[['Order ID', 'Ship Mode', 'Order Date']].drop_duplicates().reset_index(drop=True)
    
    dimension_order = pd.DataFrame({
        'order_id': order_data['Order ID'],
        'ship_mode': order_data['Ship Mode'],
        'order_date': order_data['Order Date']
    })
    
    # 8. Siapkan data untuk fact_sales
    print("8. Membuat fact_sales...")
    
    # Buat ID unik untuk fact_sales
    processed_df['row_id'] = ['ROW_' + str(i+1).zfill(5) for i in range(len(processed_df))]
    
    # Gabungkan dengan dimension_date untuk mendapatkan date_id
    date_mapping = dimension_date[['order_date', 'date_id']].set_index('order_date')
    processed_df['date_id'] = processed_df['Order Date'].map(date_mapping['date_id'])
    
    # Gabungkan dengan dimension_location untuk mendapatkan location_id
    location_mapping = location_data.set_index(['Country', 'City', 'State', 'Region'])['location_id']
    processed_df['location_id'] = processed_df.apply(
        lambda row: location_mapping.get((row['Country'], row['City'], row['State'], row['Region'])), 
        axis=1
    )
    
    # Buat fact_sales
    fact_sales = pd.DataFrame({
        'row_id': processed_df['row_id'],
        'customer_id': processed_df['Customer ID'],
        'order_id': processed_df['Order ID'],
        'product_id': processed_df['Product ID'],
        'date_id': processed_df['date_id'],
        'location_id': processed_df['location_id'],
        'sales': processed_df['Sales'],
        'quantity': processed_df['Quantity'],
        'discount': processed_df['Discount'],
        'profit': processed_df['Profit']
    })
    
    return {
        'processed_data': processed_df,
        'dimension_date': dimension_date,
        'dimension_product': dimension_product,
        'dimension_location': dimension_location,
        'dimension_order': dimension_order,
        'fact_sales': fact_sales
    }

# Fungsi untuk mendapatkan insights awal
def get_insights(processed_df):
    print("\n=== INSIGHTS AWAL ===")
    
    # 1. Total penjualan per kategori
    print("1. Total penjualan per kategori:")
    category_sales = processed_df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    print(category_sales)
    
    # 2. Total profit per kategori
    print("\n2. Total profit per kategori:")
    category_profit = processed_df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
    print(category_profit)
    
    # 3. Penjualan per region
    print("\n3. Penjualan per region:")
    region_sales = processed_df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    print(region_sales)
    
    # 4. Produk terlaris (berdasarkan kuantitas)
    print("\n4. Produk terlaris (Sub-Category):")
    top_products = processed_df.groupby('Sub-Category')['Quantity'].sum().sort_values(ascending=False)
    print(top_products)
    
    # 5. Profit margin per kategori
    print("\n5. Profit margin per kategori:")
    profit_margin = processed_df.groupby('Category').apply(lambda x: (x['Profit'].sum() / x['Sales'].sum()) * 100).sort_values(ascending=False)
    print(profit_margin)
    
    return {
        'category_sales': category_sales,
        'category_profit': category_profit,
        'region_sales': region_sales,
        'top_products': top_products,
        'profit_margin': profit_margin
    }

# Fungsi untuk ekspor data ke SQL
def generate_sql_script(data_dict, output_folder='sql_scripts'):
    # Buat folder jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n=== MEMBUAT SCRIPT SQL ===")
    
    # Buat file SQL
    sql_file_path = f"{output_folder}/create_tables.sql"
    
    with open(sql_file_path, 'w') as sql_file:
        # Tulis header
        sql_file.write("-- SQL Script untuk membuat skema data warehouse\n")
        sql_file.write("-- Generated pada: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # 1. Create dimension_date table
        sql_file.write("-- Create dimension_date table\n")
        sql_file.write("""CREATE TABLE dimension_date (
    date_id INTEGER PRIMARY KEY,
    order_date DATE,
    day VARCHAR(2),
    month VARCHAR(2),
    month_name VARCHAR(20),
    quarter VARCHAR(2),
    year VARCHAR(4)
);\n\n""")
        
        # 2. Create dimension_product table
        sql_file.write("-- Create dimension_product table\n")
        sql_file.write("""CREATE TABLE dimension_product (
    product_id VARCHAR(20) PRIMARY KEY,
    category VARCHAR(50),
    sub_category VARCHAR(50)
);\n\n""")
        
        # 3. Create dimension_location table
        sql_file.write("-- Create dimension_location table\n")
        sql_file.write("""CREATE TABLE dimension_location (
    location_id INTEGER PRIMARY KEY,
    country VARCHAR(50),
    city VARCHAR(100),
    state VARCHAR(50),
    region VARCHAR(20)
);\n\n""")
        
        # 4. Create dimension_order table
        sql_file.write("-- Create dimension_order table\n")
        sql_file.write("""CREATE TABLE dimension_order (
    order_id VARCHAR(20) PRIMARY KEY,
    ship_mode VARCHAR(20),
    order_date DATE
);\n\n""")
        
        # 5. Create fact_sales table
        sql_file.write("-- Create fact_sales table\n")
        sql_file.write("""CREATE TABLE fact_sales (
    row_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    order_id VARCHAR(20),
    product_id VARCHAR(20),
    date_id INTEGER,
    location_id INTEGER,
    sales FLOAT,
    quantity INTEGER,
    discount FLOAT,
    profit FLOAT,
    FOREIGN KEY (order_id) REFERENCES dimension_order(order_id),
    FOREIGN KEY (product_id) REFERENCES dimension_product(product_id),
    FOREIGN KEY (date_id) REFERENCES dimension_date(date_id),
    FOREIGN KEY (location_id) REFERENCES dimension_location(location_id)
);\n\n""")
    
    print(f"SQL script dibuat di {sql_file_path}")
    
    return sql_file_path

# Fungsi untuk menyimpan hasil preprocessing
def save_processed_data(data_dict, output_folder='processed_data'):
    # Buat folder jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n=== MENYIMPAN DATA ===")
    # Simpan semua dataframe
    for name, df in data_dict.items():
        if name != 'processed_data':  # Jangan simpan data mentah yang sudah diproses
            filepath = f"{output_folder}/{name}.csv"
            df.to_csv(filepath, index=False)
            print(f"Data {name} disimpan ke {filepath}")

# Fungsi utama
def main():
    # Path ke file data
    file_path = "data/HM-Sales-2018.csv"
    
    # 1. Load data
    df = load_data(file_path)
    if df is None:
        return
    
    # 2. Eksplorasi data
    missing_values, duplicates = explore_data(df)
    
    # 3. Preprocess data sesuai skema star
    data_dict = preprocess_data(df)
    
    # 4. Dapatkan insight awal
    insights = get_insights(data_dict['processed_data'])
    
    # 5. Simpan data yang sudah dipreprocessing
    save_processed_data(data_dict)
    
    # 6. Generate SQL script
    sql_file_path = generate_sql_script(data_dict)
    
    print("\n=== PREPROCESSING SELESAI ===")
    print("Data siap untuk digunakan dalam data warehouse")
    print(f"Script SQL telah dibuat di {sql_file_path}")

if __name__ == "__main__":
    main()