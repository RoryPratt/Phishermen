import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tldextract
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("phishing_site_urls.csv")[:100000]

if os.path.exists("processed_data_2.csv"):
    df = pd.read_csv("processed_data_2.csv")
    print("File already preprocessed... ")

    print("Continuing to next step...")
else:

    print("Processing...")

    df["Length"] = df["URL"].apply(len)

    df["Binary Label"] = [int(i=="good") for i in df["Label"]]

    spc = ": / ? # [ ] @ ! $ & ' ( ) * + , ; =".split()
    num = "1 2 3 4 5 6 7 8 9 0".split()
    for x in num:
        df[x + "_domain"] = [tldextract.extract(i).domain.count(x) for i in df["URL"]]
    for x in spc:
        df[x + "_domain"] = [tldextract.extract(i).domain.count(x) for i in df["URL"]]
    for x in spc:
        df[x] = [i.count(x) for i in df["URL"]]

    tlds = set([tldextract.extract(i).suffix for i in df["URL"]])
    for tld in tlds:
        df[tld] = [int(tldextract.extract(i).suffix==tld) for i in df["URL"]]

    df.to_csv("processed_data.csv", index=False)

    print("complete")

spc = ": / ? # [ ] @ ! $ & ' ( ) * + , ; =".split()
tlds = set([tldextract.extract(i).suffix for i in df["URL"]])

num = "1 2 3 4 5 6 7 8 9 0".split()

print("training model...")

spcd = [i + "_domain" for i in spc]
num = [i + "_domain" for i in num]

total = list(spc + list(tlds) + ["Length"] + num + spcd)

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

y_predict = model.predict(x_test)

f1 = f1_score(y_test, y_predict)
print("F1 Score:", f1)

