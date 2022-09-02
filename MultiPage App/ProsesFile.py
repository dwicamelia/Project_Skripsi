import streamlit as st
import yfinance as yf
import pandas as pd
#library tanggal
from datetime import timedelta, date
import datetime 
import time
from datetime import time
import math
import numpy as np

#library normalisasi dan regresi linear
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error,mean_absolute_error, mean_absolute_percentage_error, r2_score

#library plot grafik #tidak bisa ditampilkan di streamlit
import matplotlib.pyplot as plt
import plotly
from matplotlib import style
from sklearn import tree
import plotly.graph_objects as go


headerSection = st.container()
mainSection = st.container()


with mainSection:
        st.title("Dashboard Menu")
        menu = ["Komparasi Trend grafik harga","upload file"]
        pilih = st.selectbox("Menu Prediksi Saham", menu)
        # start_date = st.date_input("Tanggal Mulai", datetime.date(2020,7,22))
        # end_date = st.date_input("Tanggal Akhir", datetime.datetime.now())
        # today = datetime.datetime.now()
        st.title("------------------------------------------------")
        
        

if pilih == "Komparasi Trend grafik harga":
        st.title("Coba Lihat Trend Grafik Harga Saham")
        st.subheader("Berikut adalah list saham: LQ45")
        st.write("Pilih Saham sesuai pilihan anda")
        stocks = ("ADRO.JK", "AMRT.JK", "ANTM.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
        "BBRI.JK","BBTN.JK","BFIN.JK","BMRI.JK","BRPT.JK","BUKA.JK","CPIN.JK","EMTK.JK","ERAA.JK","EXCL.JK","GGRM.JK", "HMSP.JK",
        "HRUM.JK", "ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ITMG.JK","JPFA.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK",
        "MNCN.JK","PGAS.JK","PTBA.JK","PTPP.JK","SMGR.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK","TPIA.JK","UNTR.JK",
        "UNVR.JK","WIKA.JK","WSKT.JK")
        selected_stocks= st.multiselect("Pilih Data Set Untuk Lihat Trend Grafik", stocks)
        start = st.date_input('Start', value = pd.to_datetime('2019-6-14'))
        end = st.date_input('End', value = pd.to_datetime('today'))
        
        if len(selected_stocks)>0:
            df=yf.download(selected_stocks, start, end)['Close']
            st.line_chart(df)


        


# mengambil data saham
    #Upload_File

if pilih == "upload file":
        st.subheader("saham")
        saham_file = st.file_uploader("saham file", type=["csv"])
        if saham_file is not None:
     # Can be used wherever a "file-like" object is accepted:
            file_container = st.expander("Check your uploade .csv")
            df = pd.read_csv(saham_file)
            saham_file.seek(0)
            df['Date'] = df['Date'].str.replace('/', '')
            st.write(df)
            file_container.write( df)
        else:
            st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
                )

            st.stop()

        st.write("Cleaning Data Terlebih dahulu agar bisa diolah")
        if st.button('Clean Data'):
            #Impute missing values (NaN)
            st.write("Berikut Data yang sudah di cleaning")
            df['SPV'] = (df['High'] - df['Low']) / df['Close'] * 100.0 # Spread/Volatility
            df['CHG'] = (df['Close'] - df['Open']) / df['Open'] * 100.0 #  % change
            #Data set Baru
            df = df[['Date','Close','Open','High','Low','CHG','Volume','SPV']] # dataframe yang baru
            df.fillna(value =-99999, inplace = True)
            st.write(df, index_col='Date', parse_dates=True)
            #Plot Original Data
            plt.title('Linear Regression | Time vs. Price (Original Data)')
            plt.xlabel('Date')
            plt.ylabel('Harga')
            df['Close'].plot()
            fig=go.Figure()
            fig=fig.update_layout(showlegend=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.legend()
            plt.show()
            st.pyplot()

            st.write("Lakukan Prediksi Harga")

        if st.button('Prediksi'):
            #Hasil predikisi
            st.write("Output Prediksi: ")
            jml_OutputPrediksi = int(math.ceil(0.01*len(df))) # 1 persen / 100 = 0.01
            df['OutputPrediksi'] = df['Close'].shift(-jml_OutputPrediksi)
            #Linear Regresi
            df['Date'] = "2019-04-01".replace("-", "")
            X = np.array(df.drop(['OutputPrediksi'],1)) # X tidak termasuk kolom "output_prediksi"
            X = preprocessing.scale(X) # normalisasi nilai menjadi -1 hingga 1
            X_prediksi = X[-jml_OutputPrediksi:] # data X untuk prediksi (1 persen elemen terakhir)
            X = X[:-jml_OutputPrediksi] # data X (99 persen elemen)
            df.dropna(inplace=True) # drop/hilangkan nilai yang "not a number"/NaN
            Y = np.array(df['OutputPrediksi']) # y adalah nilai output prediksi


            #Random Split Data  
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
            model = LinearRegression(n_jobs=-1)

            #Regressi Linear R-Square
            st.write('Akurasi Prediksi Linear Regression : R-Square')
            model = LinearRegression(n_jobs=-1) # pilih regresi linier sebagai classifier dengan proses training menggunakan semua prosesor di laptop (n_jobs=-1)
            model.fit(xtrain, ytrain) # lakukan training
            r_squared = model.score(xtrain, ytrain)# hitung score
            st.write(r_squared)
            

            st.write('Hasil Prediksi Harga')
            setPrediksi = model.predict(X_prediksi) # prediksi nilai y (output)
            st.write(setPrediksi)
            st.line_chart(setPrediksi)
        
            st.write("ini adalah prediksi berdasarkan historical harga, tidak untuk sebagai keputusan investasi. Hanya sebagai bahan pertimbangan")
            
            #Evaluasi
            y_pred = model.predict(xtest)
            st.write( "Mean Absolute Percentage Error")
            ytest, y_pred = np.array(y_pred), np.array(ytest)
            MAPE = np.mean(np.abs((ytest - y_pred)/ ytest))*100
            st.write(MAPE)
            st.write( "Root Mean Squared Error")
            RMSE = np.sqrt(np.mean((ytest - y_pred)**2))
            st.write(RMSE)

            #Dataframe_untuk Grafik
            df['prediksi'] = np.nan 
            lastDate = df.iloc[-1].name # dapatkan tanggal terakhir
            lastSecond = lastDate
            oneDay = 86400 # detik =  1 hari
            nextSecond = lastSecond + oneDay
            
            for i in setPrediksi : # untuk semua nilai yang telah di prediksi
                nextDate = datetime.datetime.fromtimestamp(nextSecond) # hitung tanggal selanjutnya
                nextSecond += 86400 # tambahkan detik selanjutnya menjadi satu hari berikutnya
                df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)]+[i] # tambahkan elemen i (nilai prediksi)  


            #Plot Predicted
            plt.title('Linear Regression | Prediksi Harga Saham ')
            plt.xlabel("Date")
            plt.ylabel("Harga")
            df['Close'].plot()
            df['prediksi'].plot()
            fig=go.Figure()
            fig=fig.update_layout(showlegend=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.plotly_chart(fig)
            plt.legend()
            st.pyplot()
            plt.show()

    #Batas_olahData_Upload_File