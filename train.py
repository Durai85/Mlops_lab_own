import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv('Iris.csv')

x = df.iloc[:,1:-1]

LE = LabelEncoder()

y = LE.fit_transform(df.iloc[:,-1])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=11)

LR = LinearRegression()
LR.fit(x_train,y_train)
print("Training completed...")
y_pred = LR.predict(x_test)

MSE = mean_squared_error(y_test,y_pred)
RMSE = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("MSE: ",MSE)
print("RMSE: ",RMSE)
print("r2: ", r2)

with open("LR.pkl", "wb") as f:
    pickle.dump(LR,f)

with open("LE.pkl", "wb") as f:
    pickle.dump(LE,f)