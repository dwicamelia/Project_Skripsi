# Multiple Linear Regression

# Importing the libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error


# Importing the dataset
saham_file = st.file_uploader("saham file", type=["csv"])

if saham_file is not None:
            file_container = st.expander("Check your uploade .csv")
            df = pd.read_csv(saham_file)
            saham_file.seek(0)
            df["Date"] = df["Date"].str.replace('/', '')
            st.write(df)
            file_container.write( df)
            columnsTiltles = ["Date","Open","High","Low","Volume","Close"]
            df = df.reindex(columns=columnsTiltles)
            X = df.iloc[:, 1:-1].values
            y = df.iloc[:, 5].values

            # Encoding categorical data
            # '''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            # labelencoder = LabelEncoder()
            # X[:, ] = labelencoder.fit_transform(X[:, ])
            # onehotencoder = OneHotEncoder(categorical_features = [])
            # X = onehotencoder.fit_transform(X).toarray()'''

            # # Avoiding the Dummy Variable Trap
            # '''X = X[:, 1:]'''

            X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

            # Feature Scaling
            # from sklearn.preprocessing import StandardScaler
            # sc_X = StandardScaler()
            # X_train = sc_X.fit_transform(X_train)
            # X_test = sc_X.transform(X_test)

            # '''sc_y = StandardScaler()
            # y_train = sc_y.fit_transform(y_train)'''

            # Fitting Multiple Linear Regression to the Training set
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
            regressor.fit(X_train1, y_train1)

            # Predicting the Test set results
            y_pred = regressor.predict(X_test1)



            from sklearn.linear_model import LinearRegression
            regressorr = LinearRegression()
            regressorr.fit(X_train1, y_train1)


            # Predicting the Test set results
            y_pred1 = regressorr.predict(X_test1)
            plt.plot(y_test1, y_pred)
            plt.plot(y_test1, y_pred1)

            y_true = y_test1.tolist()
            y_pred2 = y_pred1.tolist()

            st.write("R-square")
            r2 = r2_score(y_true, y_pred2)
            st.write(r2)

            st.write("Mean Square error")
            mse=mean_squared_error(y_true, y_pred2)
            st.write(mse)

            st.write("Root Mean Square error")

            rmse=mean_squared_log_error(y_true, y_pred2) 
            st.write(rmse)

            st.write("Mean Absolute Percentage error")
            mape = mean_absolute_percentage_error(y_true, y_pred2)
            st.write(mape)