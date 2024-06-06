import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Loan Approval Prediction App

This app predicts the loan approval probablity
""")

st.header('Applicant Input Parameters')

def user_input_features():
    no_of_dependents = st.number_input('Numbers of Dependents of the Applicant',min_value = 0, max_value=20,step=1)
    education = st.selectbox('Education',('Graduate','Not Gaduate'))
    self_employed = st.selectbox('Employment Status',('Yes','No'))
    income_annum = st.number_input('Annual Income',min_value=0,max_value = 20000000,step=1)
    loan_amount = st.number_input('Loan Amount',min_value=0,max_value=20000000,step=1)
    loan_term = st.number_input('Loan Terms in a Year',min_value=0,max_value=50,step=1)
    cibil_score = st.number_input('Credit Score',min_value=0,max_value=900,step=1)
    residential_assets_value = st.number_input('Residential Assets Value',min_value=0,max_value=20000000,step=1)
    commercial_assets_value = st.number_input('Commercial Assets Value',min_value=0,max_value=20000000,step=1)
    luxury_assets_value = st.number_input('Luxury Assets Value',min_value=0,max_value=20000000,step=1)
    bank_asset_value = st.number_input('Bank Assets Value',min_value=0,max_value=20000000,step=1)

    data = {'no_of_dependents':no_of_dependents,
            'education':education,
            'self_employed':self_employed,
            'income_annum':income_annum,
            'loan_amount':loan_amount,
            'loan_term':loan_term,
            'cibil_score':cibil_score,
            'residential_assets_value':residential_assets_value,
            'commercial_assets_value':commercial_assets_value,
            'luxury_assets_value':luxury_assets_value,
            'bank_asset_value':bank_asset_value
            }


    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()


# st.subheader('User Input parameters')
# st.write(df.to_dict())

df['education'] = df['education'].replace({'Graduate':1,'Not Graduate':0})
df['self_employed'] = df['self_employed'].replace({'Yes':1,'No':0})

model = pickle.load(open('model.pkl','rb'))

def LoanPred(df,model):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    # st.subheader('Class labels and their corresponding index number')
    # st.write(pd.DataFrame({'Labels': ['Approved','Declined']}))

    st.subheader('Prediction')
    if prediction == 0:
        st.write('Congratulations, your loan application is "Approved"')
    else:
        st.write('Sorry, your loan application is "Rejected"')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if st.button('Submit'):
    LoanPred(df,model)


