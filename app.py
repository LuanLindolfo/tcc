
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Dashboard Estratégico - Castanhal', layout='wide')
st.title('📊 Gestão Pública Inteligente: Censo Castanhal 2022')

aba = st.sidebar.selectbox('Selecione o Eixo', ['Visão Geral', 'Saneamento', 'Educação', 'Renda'])

if aba == 'Visão Geral':
    st.header('📍 Panorama Geral')
    col1, col2 = st.columns(2)
    col1.metric('População Total', '192.256')
    col2.metric('Densidade', '186,78 hab/km²')

elif aba == 'Saneamento':
    st.header('🏠 Infraestrutura Urbana')
    df_san = pd.DataFrame({
        'Serviço': ['Esgoto', 'Água', 'Banheiro', 'Lixo'],
        'Porcentagem': [17.19, 43.22, 98.43, 94.72]
    })
    fig = px.bar(df_san, x='Serviço', y='Porcentagem', color='Serviço', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

elif aba == 'Educação':
    st.header('📚 Nível de Instrução por Gênero')
    escolaridade_data = pd.DataFrame({
        'Nível': ['Sem Instrução', 'Fundamental', 'Médio', 'Superior'],
        'Mulheres': [20224, 13134, 29631, 10930],
        'Homens': [22455, 13775, 23828, 6626]
    })
    fig = px.bar(escolaridade_data, x='Nível', y=['Mulheres', 'Homens'], barmode='group')
    st.plotly_chart(fig, use_container_width=True)

elif aba == 'Renda':
    st.header('💰 Distribuição de Renda')
    renda_data = pd.DataFrame({
        'Classe': ['Até 1/4 SM', '1/4 a 1/2', '1/2 a 1', '1 a 2', '2 a 3', '3 a 5', '5+'],
        'Pessoas': [3368, 6848, 26202, 22647, 7647, 4775, 2785]
    })
    fig = px.pie(renda_data, values='Pessoas', names='Classe', title='Distribuição por Salário Mínimo')
    st.plotly_chart(fig, use_container_width=True)
