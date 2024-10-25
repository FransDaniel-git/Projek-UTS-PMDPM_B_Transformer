import streamlit as st
import pandas as pd
import pickle
import os
from streamlit_option_menu import option_menu

# Navigasi sidebar dengan warna dan style custom
with st.sidebar:
    st.markdown("<h2 style='color:#4C9A2A;'>🧠 Streamlit UTS ML 24/25</h2>", unsafe_allow_html=True)
    selected = option_menu('Pilih Menu',
                           ['Klasifikasi', 'Regresi', 'Catatan'],
                           icons=['archive', 'journal-bookmark', 'exclamation-circle'],
                           menu_icon="cast", default_index=0,
                           styles={
                               "container": {"background-color": "#2c2c2c"},
                               "icon": {"color": "#ffffff", "font-size": "20px"},
                               "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "grey;"},
                               "nav-link-selected": {"background-color": "#4C9A2A"},
                           })

# Halaman Klasifikasi
if selected == 'Klasifikasi':
    st.title('🏷️ Klasifikasi Properti')
    
    st.write('Untuk Inputan File dataset (csv) bisa menggunakan st.file_uploader')
    file = st.file_uploader('Masukkan File Dataset Anda', type=["csv", "txt"])
    
    def validate_file(data):
        keywords = ["luxury", "basic", "middle"]
        data_str = data.astype(str)  # Convert all data to string for searching
        if not data_str.apply(lambda x: x.str.contains('|'.join(keywords), case=False)).any().any():
                raise ValueError("The dataset does not contain the required categories: 'luxury', 'basic', 'middle'.")
    if file is not None:
        try:
              input_data = pd.read_csv(file)
              
              
              validate_file(input_data)
              st.markdown("<h3 style='text-align: center; color: #0073e6;'>📊 Data yang Diupload:</h3>", unsafe_allow_html=True)
              st.dataframe(input_data)
        except ValueError as e:
              st.markdown("<h3 style='text-align : center' font-color: red> !!⚠️ An error occurred while reading the file ⚠️!! </h3>", unsafe_allow_html=True)
    # Lokasi model
    model_directory = r'E:\clone\Project'
    model_path = os.path.join(model_directory, r'BestModel_CLF_RFC_NamaSB.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        rf_model = loaded_model
        
        # Bagian Input
        st.markdown("### Masukkan Fitur Properti:")
        col1, col2, col3 = st.columns(3)
        with col1:
            squaremeters = st.number_input('🏡 Luas Tanah (meter persegi)', 0.0)
            numberofrooms = st.number_input('🛏️ Jumlah Kamar', 0)
            floors = st.number_input('🏢 Jumlah Lantai', 0)
            hasguestroom = st.number_input("🛋️ Apakah memiliki ruang tamu?", 0)
            isnewbuilt = st.radio("🆕 Apakah bangunan baru?", ('New', 'Old'))
        with col2:
            numprevowners = st.number_input('👥 Jumlah Pemilik Sebelumnya', 0)
            citypartrange = st.number_input('🌇 Eksklusivitas Kawasan', 0)
            made = st.number_input('📅 Tahun Pembuatan', 0)
            hasyard = st.radio("🏡 Apakah memiliki halaman?", ('Ya', 'Tidak'))
            haspool = st.radio("🏊‍♂️ Apakah memiliki kolam renang?", ('Ya', 'Tidak'))
        with col3:
            basement = st.number_input('📦 Luas Basement (meter persegi)', 0.0)
            attic = st.number_input('🏠 Luas Loteng (meter persegi)', 0.0)
            garage = st.number_input('🚗 Luas Garasi (meter persegi)', 0.0)
            hasstormprotector = st.radio("🌪️ Apakah memiliki pelindung badai?", ('Ya', 'Tidak'))
            hasstorageroom = st.radio("📦 Apakah memiliki gudang?", ('Ya', 'Tidak'))

        col1, col2 = st.columns(2)
        with col1:
            citycode = st.text_input('🏙️ Kode Lokasi (City Code)')
        with col2:
            price = st.number_input('💰 Harga (dalam mata uang yang sesuai)', 0.0)

        if hasyard == 'Ya':
                onehotencoder__hasyard_no = 0
                onehotencoder__hasyard_yes = 1
        elif hasyard == 'Tidak':
                onehotencoder__hasyard_no = 1
                onehotencoder__hasyard_yes = 0
        if haspool == 'Ya':
                onehotencoder__haspool_no = 0
                onehotencoder__haspool_yes = 1
        elif haspool == 'Tidak':
                onehotencoder__haspool_no = 1
                onehotencoder__haspool_yes = 0
        if isnewbuilt == 'New':
                onehotencoder__isnewbuilt_new = 1
                onehotencoder__isnewbuilt_old = 0
        elif isnewbuilt == 'Old':
                onehotencoder__isnewbuilt_new = 0
                onehotencoder__isnewbuilt_old = 1
        if hasstormprotector == 'Ya':
                onehotencoder__hasstormprotector_no = 0
                onehotencoder__hasstormprotector_yes = 1
        elif hasstormprotector == 'Tidak':
                onehotencoder__hasstormprotector_no = 1
                onehotencoder__hasstormprotector_yes = 0
        if hasstorageroom == 'Ya':
                onehotencoder__hasstorageroom_no = 0
                onehotencoder__hasstorageroom_yes = 1
        elif hasstorageroom == 'Tidak':
                onehotencoder__hasstorageroom_no = 1
                onehotencoder__hasstorageroom_yes = 0
            
        input_data = [[onehotencoder__hasyard_no, onehotencoder__hasyard_yes, onehotencoder__haspool_no, onehotencoder__haspool_yes, onehotencoder__isnewbuilt_new, onehotencoder__isnewbuilt_old, 
               onehotencoder__hasstormprotector_no, onehotencoder__hasstormprotector_yes, onehotencoder__hasstorageroom_no, onehotencoder__hasstorageroom_yes, squaremeters, numberofrooms,
               floors, citycode, citypartrange, numprevowners, made, basement, attic, garage, hasguestroom, price]]

        if st.button("🔍 Prediksi"):
            rf_model_predict = rf_model.predict(input_data)
            st.success(f"Prediksi Kategori Properti: **{rf_model_predict[0]}**")
    else:
        st.error("⚠️ Model tidak ditemukan, silakan cek file model di direktori.")

# Halaman Regresi
if selected == 'Regresi':
    st.title('📈 Prediksi Harga Properti')

    model_directory = r'E:\clone\Project'
    model_path = os.path.join(model_directory, r'BestModel_REG_Ridge_NamaSB.pkl')

    st.write('Untuk Inputan File dataset (csv) bisa menggunakan st.file_uploader')
    file = st.file_uploader('Masukkan File Dataset Anda', type=["csv", "txt"])

    def validate_file(data):
        keywords = ["luxury", "basic", "middle"]
        data_str = data.astype(str)  # Convert all data to string for searching
        if not data_str.apply(lambda x: x.str.contains('|'.join(keywords), case=False)).any().any():
                raise ValueError("The dataset does not contain the required categories: 'luxury', 'basic', 'middle'.")
    if file is not None:
        try:
              input_data = pd.read_csv(file)
              
              
              validate_file(input_data)
              st.markdown("<h3 style='text-align: center; color: #0073e6;'>📊 Data yang Diupload:</h3>", unsafe_allow_html=True)
              st.dataframe(input_data)
        except ValueError as e:
              st.markdown("<h3 style='text-align : center' font-color: red> !!⚠️ An error occurred while reading the file ⚠️!! </h3>", unsafe_allow_html=True)
        

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        Lasso_model = loaded_model.best_estimator_

        # Bagian Input
        st.markdown("### Masukkan Fitur Properti:")
        col1, col2, col3 = st.columns(3)
        with col1:
            squaremeters = st.number_input('🏡 Luas Tanah (meter persegi)', 0.0)
            numberofrooms = st.number_input('🛏️ Jumlah Kamar', 0)
            floors = st.number_input('🏢 Jumlah Lantai', 0)
            hasguestroom = st.number_input("🛋️ Apakah memiliki ruang tamu?", 0)
            isnewbuilt = st.radio("🆕 Apakah bangunan baru?", ('New', 'Old'))
        with col2:
            numprevowners = st.number_input('👥 Jumlah Pemilik Sebelumnya', 0)
            citypartrange = st.number_input('🌇 Eksklusivitas Kawasan', 0)
            made = st.number_input('📅 Tahun Pembuatan', 0)
            hasyard = st.radio("🏡 Apakah memiliki halaman?", ('Ya', 'Tidak'))
            haspool = st.radio("🏊‍♂️ Apakah memiliki kolam renang?", ('Ya', 'Tidak'))
            
        with col3:
            basement = st.number_input('📦 Luas Basement (meter persegi)', 0.0)
            attic = st.number_input('🏠 Luas Loteng (meter persegi)', 0.0)
            garage = st.number_input('🚗 Luas Garasi (meter persegi)', 0.0)
            hasstormprotector = st.radio("🌪️ Apakah memiliki pelindung badai?", ('Ya', 'Tidak'))
            hasstorageroom = st.radio("📦 Apakah memiliki gudang?", ('Ya', 'Tidak'))
            
        col1, col2 = st.columns(2)
        with col1:
            citycode = st.text_input('🏙️ Kode Lokasi (City Code)')
        with col2:
            category = st.selectbox('Kategori Properti', ['Basic', 'Middle', 'Luxury'])
            
        

        if hasyard == 'Ya':
                onehotencoder__hasyard_no = 0
                onehotencoder__hasyard_yes = 1
        elif hasyard == 'Tidak':
                onehotencoder__hasyard_no = 1
                onehotencoder__hasyard_yes = 0
        if haspool == 'Ya':
                onehotencoder__haspool_no = 0
                onehotencoder__haspool_yes = 1
        elif haspool == 'Tidak':
                onehotencoder__haspool_no = 1
                onehotencoder__haspool_yes = 0
        if isnewbuilt == 'New':
                onehotencoder__isnewbuilt_new = 1
                onehotencoder__isnewbuilt_old = 0
        elif isnewbuilt == 'Old':
                onehotencoder__isnewbuilt_new = 0
                onehotencoder__isnewbuilt_old = 1
        if hasstormprotector == 'Ya':
                onehotencoder__hasstormprotector_no = 0
                onehotencoder__hasstormprotector_yes = 1
        elif hasstormprotector == 'Tidak':
                onehotencoder__hasstormprotector_no = 1
                onehotencoder__hasstormprotector_yes = 0
        if hasstorageroom == 'Ya':
                onehotencoder__hasstorageroom_no = 0
                onehotencoder__hasstorageroom_yes = 1
        elif hasstorageroom == 'Tidak':
                onehotencoder__hasstorageroom_no = 1
                onehotencoder__hasstorageroom_yes = 0
        if category == 'Basic':
                onehotencoder__category_Basic = 1
                onehotencoder__category_Luxury = 0
                onehotencoder__category_Middle = 0
        elif category == 'Middle':
                onehotencoder__category_Basic = 0
                onehotencoder__category_Luxury = 0
                onehotencoder__category_Middle = 1
        elif category == 'Luxury':
                onehotencoder__category_Basic = 0
                onehotencoder__category_Luxury = 1
                onehotencoder__category_Middle = 0
            
        input_data = [[onehotencoder__hasyard_no, onehotencoder__hasyard_yes, onehotencoder__haspool_no, onehotencoder__haspool_yes, onehotencoder__isnewbuilt_new, onehotencoder__isnewbuilt_old, 
               onehotencoder__hasstormprotector_no, onehotencoder__hasstormprotector_yes, onehotencoder__hasstorageroom_no, onehotencoder__hasstorageroom_yes, onehotencoder__category_Basic, onehotencoder__category_Luxury, onehotencoder__category_Middle,squaremeters, numberofrooms,
               floors, citycode, citypartrange, numprevowners, made, basement, attic, garage, hasguestroom]]  # Contoh untuk disederhanakan

        if st.button("💸 Prediksi Harga"):
            Lasso_model_predict = Lasso_model.predict(input_data)
            formatted_price = f"${Lasso_model_predict[0]:,.2f}"
            st.markdown(f"<h3 style='color: #4C9A2A;'>💵 Prediksi Harga Properti: {formatted_price}</h3>", unsafe_allow_html=True)

    else:
        st.error("⚠️ Model tidak ditemukan, silakan cek file model di direktori.")

# Halaman Catatan
if selected == 'Catatan':
    st.title('📚 Catatan')
    st.write('''1. Bagian sidebar akan menampilkan menu dari fungsi streamlit yang kita buat".''')
    st.write('2. Menu yang dibuat ada 2 yaitu Klasifikasi dan Regresi.')
    st.write('''3. Klasifikasi akan menghasilkan output kategori properti sedangkan regresi akan menghasilkan output harga properti.''')