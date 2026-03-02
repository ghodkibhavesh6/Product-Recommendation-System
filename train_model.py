import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

#load datset
df = pd.read_csv("product data.csv")

#drop extra columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

#encoder
le_gender = LabelEncoder()
le_location = LabelEncoder()
le_interest = LabelEncoder()
le_category = LabelEncoder()

df["Gender"] = le_gender.fit_transform(df["Gender"])
df["Location"] = le_location.fit_transform(df["Location"])
df["Interests"] = le_interest.fit_transform(df["Interests"])
df["Product_Category_Preference"] = le_category.fit_transform(
    df["Product_Category_Preference"]
)

#feature for similarity
features = df[
    [
        "Age",
        "Gender",
        "Location",
        "Income",
        "Interests",
        "Purchase_Frequency",
        "Average_Order_Value",
        "Total_Spending",
    ]
]

similarity_matrix = cosine_similarity(features)

model_data = {
    "similarity": similarity_matrix,
    "df": df,
    "le_category": le_category
}

#save model
import pickle
with open("recommender.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model trained and saved successfully")
