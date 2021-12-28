import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from win32com.client import Dispatch


model = pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))
st.title("Email Spam Detector")
msg=st.text_input("Enter an Email")
if st.button("Process"):
	print(msg)
	print(type(msg))
	data=[msg]
	print(data)
	vec=cv.transform(data).toarray()
	result=model.predict(vec)
	if result[0]==0:
		st.success("This is Not A Spam Email")
	else:
		st.error("This is A Spam Email")
