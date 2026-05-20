import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import defaultdict

# ── Configuração da Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🧠",
    layout="wide"
)

# ── Constantes e Mapeamentos ─────────────────────────────────────────────────
NOME_MODELO = "nlptown/bert-base-multilingual-uncased-sentiment"

MAPA_ESTRELAS = {
    1: "Muito Negativo 😠",
    2: "Negativo 😞",
    3: "Neutro / Misto 😐",
    4: "Positivo 🙂",
    5: "Muito Positivo 😄",
}

# ── Funções Auxiliares ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de IA (isso pode levar um minuto na primeira vez)...")
def carregar_modelo():
    """Carrega o pipeline do Hugging Face. Fica em cache para não recarregar."""
    return pipeline(
        "sentiment-analysis",
        model=NOME_MODELO,
        truncation=True,
        max_length=512
    )

def extrair_estrelas(label: str) -> int:
    """Extrai o número de estrelas do rótulo retornado pelo modelo."""
    match = re.search(r"(\d)", label)
    return int(match.group(1)) if match else 3

def plotar_graficos(df_resultados):
    """Gera os gráficos de barras e pizza com base nos resultados."""
    contagem = df_resultados['estrelas'].value_counts().to_dict()
    
    # Preenche categorias faltantes com 0 para o gráfico manter a estrutura
    for i in range(1, 6):
        if i not in contagem:
            contagem[i] = 0
            
    estrelas_lista = list(range(1, 6))
    totais_lista = [contagem[e] for e in estrelas_lista]
    cores = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de Barras
    bars = axes[0].bar(estrelas_lista, totais_lista, color=cores, edgecolor="white", width=0.6)
    axes[0].set_xlabel("Estrelas")
    axes[0].set_ylabel("Quantidade de Frases")
    axes[0].set_title("Quantidade por Categoria")
    axes[0].set_xticks(estrelas_lista)
    axes[0].set_xticklabels([f"{e}⭐" for e in estrelas_lista])
    
    # Adiciona rótulos em cima das barras
    for bar, valor in zip(bars, totais_lista):
        if valor > 0:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(valor),
                ha="center", va="bottom", fontweight="bold"
            )

    # Gráfico de Pizza
    dados_pizza = [(e, t) for e, t in zip(estrelas_lista, totais_lista) if t > 0]
    if dados_pizza:
        estrelas_p = [d[0] for d in dados_pizza]
        totais_p = [d[1] for d in dados_pizza]
        cores_p = [cores[e - 1] for e in estrelas_p]
        rotulos_p = [f"{e}⭐ ({t})".replace("⭐", "★") for e, t in dados_pizza]
        
        axes[1].pie(
            totais_p,
            labels=rotulos_p,
            colors=cores_p,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.75
        )
        axes[1].set_title("Proporção por Categoria")
    else:
        axes[1].axis('off')
        axes[1].set_title("Sem dados suficientes para proporção")

    plt.tight_layout()
    return fig

# ── Interface da Aplicação ───────────────────────────────────────────────────
st.title("🧠 Análise de Sentimentos 2025")
st.markdown("Classifique textos em português em uma escala de 1 a 5 estrelas usando Inteligência Artificial (**BERT**).")

# Inicializa o modelo
analisador = carregar_modelo()

# Abas para separar análise individual de lote
aba1, aba2 = st.tabs(["📝 Analisar Texto Único", "📁 Analisar Arquivo (Lote)"])

with aba1:
    texto_usuario = st.text_area("Digite o texto ou opinião para analisar:", placeholder="Ex: O serviço foi excelente, recomendo a todos!")
    
    if st.button("Analisar Sentimento", type="primary"):
        if texto_usuario.strip():
            with st.spinner("Analisando..."):
                resultado = analisador(texto_usuario)[0]
                estrelas = extrair_estrelas(resultado["label"])
                confianca = resultado["score"]
                
            st.success(f"**Resultado:** {'⭐' * estrelas} — {MAPA_ESTRELAS[estrelas]}")
            st.info(f"**Confiança da IA:** {confianca:.2%}")
        else:
            st.warning("Por favor, digite algum texto antes de analisar.")

with aba2:
    st.markdown("Faça upload de um arquivo `.txt` contendo **uma opinião por linha**.")
    arquivo_upload = st.file_uploader("Escolha um arquivo de texto", type=["txt"])
    
    if arquivo_upload is not None:
        # Lê o arquivo e separa as linhas
        conteudo = arquivo_upload.getvalue().decode("utf-8")
        frases = [linha.strip() for linha in conteudo.split('\n') if linha.strip()]
        
        st.write(f"✅ **{len(frases)} frases encontradas.**")
        
        if st.button("Processar Arquivo", type="primary"):
            barra_progresso = st.progress(0)
            texto_progresso = st.empty()
            
            resultados = []
            
            # Processa as frases com barra de progresso visual
            for i, frase in enumerate(frases):
                try:
                    resultado = analisador(frase)[0]
                    estrelas = extrair_estrelas(resultado["label"])
                    resultados.append({
                        "Frase": frase,
                        "Estrelas": estrelas,
                        "Categoria": MAPA_ESTRELAS[estrelas],
                        "Confiança": f"{resultado['score']:.2%}"
                    })
                except Exception as e:
                    st.error(f"Erro ao processar a frase: '{frase}'. Detalhe: {e}")
                
                # Atualiza progresso
                progresso_atual = (i + 1) / len(frases)
                barra_progresso.progress(progresso_atual)
                texto_progresso.text(f"Processando: {i+1} de {len(frases)}...")
            
            texto_progresso.text("Processamento concluído!")
            st.balloons()
            
            # Converte para DataFrame para visualização
            df = pd.DataFrame(resultados)
            
            st.subheader("📋 Resultados Detalhados")
            st.dataframe(df, use_container_width=True)
            
            # Botão de download CSV
            csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 Baixar Resultados como CSV",
                data=csv,
                file_name='resultados_sentimento.csv',
                mime='text/csv',
            )
            
            st.divider()
            
            # Gráficos de Visualização de Dados no final
            st.subheader("📊 Visualização dos Dados")
            fig_graficos = plotar_graficos(df)
            st.pyplot(fig_graficos)