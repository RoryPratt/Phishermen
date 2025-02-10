import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tldextract
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("phishing_site_urls.csv")[:100000]

print("Processing...")

df["Length"] = df["URL"].apply(len)

df["Binary Label"] = [int(i=="good") for i in df["Label"]]

spc = ": / ? # [ ] @ ! $ & ' ( ) * + , ; =".split()
for x in spc:
    df[x] = [i.count(x) for i in df["URL"]]

tlds = set([tldextract.extract(i).suffix for i in df["URL"]])
for tld in tlds:
    df[tld] = [int(tldextract.extract(i).suffix==tld) for i in df["URL"]]

df.to_csv("processed_data.csv", index=False)


spc = ": / ? # [ ] @ ! $ & ' ( ) * + , ; =".split()
tlds = set([tldextract.extract(i).suffix for i in df["URL"]])

print("complete")

print("training model...")

total = spc + list(tlds) + ["Length"]

total.remove('')


x = df[total].values
y = df["Binary Label"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train = x_train.reshape(-1, len(total))
x_test = x_test.reshape(-1, len(total))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = model.score(x_test, y_test) * 100
print(f"Model Accuracy: {accuracy:.2f}")
