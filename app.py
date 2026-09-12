import streamlit as st
import streamlit_authenticator as stauth

st.write("### Gerador de Hash Temporário")
senha_digitada = st.text_input("Digite a senha que quer transformar em hash:", type="password")

if senha_digitada:
    hash_result = stauth.Hasher([senha_digitada]).generate()[0]
    st.code(hash_result, language="text")
