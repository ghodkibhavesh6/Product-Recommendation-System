# ==================================
# STREAMLIT PRODUCT RECOMMENDER
# ==================================

import streamlit as st
import pickle

# Load saved model
with open("recommender.pkl", "rb") as f:
    data = pickle.load(f)

similarity_matrix = data["similarity"]
df = data["df"]
le_category = data["le_category"]


# Recommendation function
def recommend_by_userid(user_id):
    idx = df[df["User_ID"] == user_id].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    similar_users = sorted_scores[1:4]

    recommended = []
    for i in similar_users:
        cat_code = df.iloc[i[0]]["Product_Category_Preference"]
        category = le_category.inverse_transform([int(cat_code)])[0]
        recommended.append(category)

    return list(set(recommended))


# UI
st.title("🛍️ Product Recommendation System")

user = st.selectbox("Select User", df["User_ID"])

if st.button("Recommend Products"):
    recs = recommend_by_userid(user)

    st.subheader("Recommended Categories:")
    for r in recs:
        st.write("👉", r)