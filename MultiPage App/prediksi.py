import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
#library tanggal
from datetime import timedelta, date
import datetime 
import time
from datetime import time
from math import sqrt
import math
import numpy as np

#library normalisasi dan regresi linear
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
#library plot grafik #tidak bisa ditampilkan di streamlit
import matplotlib.pyplot as plt
import plotly
from matplotlib import style
from sklearn import tree
import plotly.graph_objects as go


headerSection = st.container()
mainSection = st.container()
navbarLeft = st.sidebar

with mainSection:
        st.title("Dashboard Menu Prediksi")
        st.markdown("ini Halaman Dashboard Processing")
        start_date = st.date_input("Tanggal Mulai", datetime.date(2021, 8, 8))
        end_date = st.date_input("Tanggal Akhir", datetime.datetime.now())
        today = datetime.datetime.now()

        stocks = st.text_input("Input Kode Saham.JK : ", 'BMRI.JK' )
        menu = ["Wellcome", "DataSet", "cleaning", "databaru","prediksi"]
        pilih = st.selectbox("Menu Prediksi Saham", menu)

        Data = yf.download(stocks, country='indonesia', start = start_date , end = end_date,
        actions= False, rounding= True)
        #Data set diperoleh
        Data = Data[['Open','High','Low','Close','Adj Close','Volume']] # definisi dataframe yang baru

with mainSection:

    if pilih == "Wellcome":
        st.title("Tentukan Saham Yang Ingin diPrediksi")
        st.write("ini adalah Proses Pertama dari prediksi Harga saham \n "+stocks)
        st.write('Hallo Investor/Trader')

    elif pilih == "DataSet":
        st.write("ini adalah Data set saham\n" +stocks )
        st.write(Data)
        st.write('Ini adalah Line chart dari saham\n' + stocks)
        st.line_chart(Data['Close'])

    elif pilih == "cleaning":
        #cleaning data agar tidak terjadi erorr (mengisi data nan/nonNumerik dengan outlier)
        Data.fillna(value =-99999, inplace = True)
        st.write("fillna value = -99999, digunakan untuk mengisi data yang terdiri dari data non number")
        st.write('dilakukan agar tidak erorr. -99999 merupakan data ekstrem agar machine dapat mengabaikannya')
        st.write(Data)

    elif pilih == "databaru":
    #menampilkan data baru yang kolom nya sudah di filter
        Data = Data[['Open', 'High','Low','Close' ,'Volume']]
        st.write('tabel baru setelah di filter (Preprocessing)')
        st.write(Data)

    elif pilih == "prediksi":
    #menentukan banyaknya output yang akan terprediksi
        st.write("Output Prediksi: ")
        jml_OutputPrediksi = int(math.ceil(0.01*len(Data['Close']))) # 1 persen / 100 = 0.01
        Data['OutputPrediksi'] = Data['Close'].shift(-jml_OutputPrediksi)# menggeser tabel kolom "Close" ke atas sehingga beberapa nilai terakhir menjadi NaN
        
    # Splitting the dataset into 80% training data and 20% testing data.
        x_train = Data[['Open', 'High', 'Low', 'Volume']]
        x_test = Data[['Open', 'High', 'Low', 'Volume']]
        y_train = Data['Close']
        y_test = Data['Close']


        x = np.array(Data.drop(['OutputPrediksi'],1)) # X tidak termasuk kolom "output_prediksi"
        x = preprocessing.scale(x) # normalisasi nilai menjadi -1 hingga 1
        X_prediksi = x[-jml_OutputPrediksi:] # data X untuk prediksi (1 persen elemen terakhir)
        x = x[:-jml_OutputPrediksi] # data X (99 persen elemen)
        Data.dropna(inplace=True) # drop/hilangkan nilai yang "not a number"/NaN
        Y = np.array(Data['OutputPrediksi']) # y adalah nilai output prediksi
        x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2) # cross validation dengan ukuran data yang diuji 20%
        
        
        st.write("ini adalah jumlah data yg di training")
        st.write("X Train",x_train.shape)
        st.write("X Test",x_test.shape)
        st.write("Y Train",y_train.shape)
        st.write("Y Test",y_test.shape)

            #Tabel Prediksi
        st.write("Tabel Prediksi Linear Regression")
        DataPred = Data['OutputPrediksi']
        st.write(Data, DataPred)


        #Regressi Linear R-Square
        st.write('Akurasi Coefficient Prediksi Linear Regression : R-Square')
        model = LinearRegression(n_jobs=-1) # pilih regresi linier sebagai classifier dengan proses training menggunakan semua prosesor di laptop (n_jobs=-1)
        model.fit(x_train, y_train) # lakukan training
        r_squared = model.score(x_test, y_test)# hitung score
        st.write(r_squared)

        #PREDIKSI_HARGA_KEDEPAN
        st.write('Hasil Prediksi Harga')
        setPrediksi = model.predict(X_prediksi) # prediksi nilai y (output)
        st.table(setPrediksi)
        st.line_chart(setPrediksi)
        st.write("ini adalah prediksi berdasarkan historical harga, tidak untuk sebagai keputusan investasi. Hanya sebagai bahan pertimbangan")

        #Regressi Linear Mean Square Error
        y_pred = model.predict(x_test)
        y_pred, y_test = np.array(y_pred), np.array(y_test)

        MAPE = np.mean(np.abs((y_test - y_pred)/ y_test))*100
        st.write( "Mean Absolute Percentage Error:" ,MAPE)
        
        # RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
        # RMSE = np.sqrt(((y_pred - y_test)**2).mean())
        RMSE = mean_squared_log_error(y_test, y_pred)
        st.write("Root Mean Squared Error:" ,RMSE)
        # Mean Squared Error
        MSE = mean_squared_error(y_test, y_pred)
        st.write("Mean Square Error is: " , MSE)
        r2 = r2_score(y_test, y_pred)
        st.write("The r2 is: ",r2)
        
        #Dataframe_untuk Grafik
        Data['prediksi'] = np.nan
        lastDate = Data.iloc[-1].name # dapatkan tanggal terakhir
        lastSecond = lastDate.timestamp()
        oneDay = 86400 # detik =  1 hari
        nextSecond = lastSecond + oneDay

    #Legenda
        for i in setPrediksi : # untuk semua nilai yang telah di prediksi
                nextDate = datetime.datetime.fromtimestamp(nextSecond) # hitung tanggal selanjutnya
                nextSecond += 86400 # tambahkan detik selanjutnya menjadi satu hari berikutnya
                Data.loc[nextDate] = [np.nan for _ in range(len(Data.columns)-1)]+[i] # tambahkan elemen i (nilai prediksi) 
                


        st.title('Prediksi Harga saham\n' + stocks)
            
        Data['Close'].plot()
        Data['prediksi'].plot()

        
        st.set_option('deprecation.showPyplotGlobalUse', False)

        qf = cf.QuantFig(Data, legend='top',  name=stocks)
        
        fig1 = qf.iplot(asFigure=True, dimensions=(800, 600), fill=True)
        
        plt.xlabel("Date")
        plt.ylabel("Harga")
        plt.legend()	    
        st.pyplot()
        plt.show()    
        # Render plot using plotly_chart
        st.plotly_chart(fig1)  