import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("product data.csv")
df = df.drop(columns=["Unnamed: 0"])

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Location"] = le.fit_transform(df["Location"])
df["Interests"] = le.fit_transform(df["Interests"])
df["Product_Category_Preference"] = le.fit_transform(df["Product_Category_Preference"])

feature = df[[
    "Age", "Gender", "Location", "Interests", "Purchase_Frequency", "Average_Order_Value", "Total_Spending"
]]

similarity_matrix = cosine_similarity(feature)

model_data = {
    "similarty": similarity_matrix,
    "df":df,
    "le": le
}

import pickle 
with open("recommender.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("model trained and save")