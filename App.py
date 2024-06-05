
import streamlit  as st  
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os 


#Barra Lateral 
with st.sidebar: 
    st.title('PDF CHAT')
    st.markdown(''' 
    ## Sobre este programa:
    Este chatbot fue creado usando:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Hecho por Juan David Ríos Sánchez')

####### FUNCIÓN PRINCIPAL####


def main():
   
    st.title("Consultor PDF ")
    
    load_dotenv() #IMPORTANTE
    

    # cargar el pdf
    pdf = st.file_uploader("Suba su PDF", type="pdf")

    if pdf is not None:
        
        pdf_reader = PdfReader(pdf)
        st.write("Archivo Subido exitosamente")

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Dividirlo en varios bloques

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
        
        
# Incrustación 

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        VectorStore.save_local("vectorstore")
        x = FAISS.load_local("vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)
        

#RESPUESTA DE LA IA

    # PRIMER QUERY : CONCEPTOS SOLICITADOS

        query_1 = ("""Resume el texto en 50 palabras y de el mismo saca:
*Métricas:
                   
*/nIndicadores:
                   
*/nFórmulas:
                   
*/nReportes:
                   
*/nUnidades de medida:
                   Solo enumeralos no los expliques""")
        docs = VectorStore.similarity_search(query=query_1, k=3)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question= query_1)
        st.write(response)


        ##### MÁS INDAGACION ######
        query =  st.text_input("Indaga más sobre tu PDF:")
        if query:
                   
            docs_1 = VectorStore.similarity_search(query=query, k=3)
            llm_1 = OpenAI(temperature=0,)
            chain_1 = load_qa_chain(llm=llm_1, chain_type="stuff")
            response_1 = chain_1.run(input_documents=docs_1, question= query)
            st.write(response_1)
         
            
if __name__  == "__main__":
    main()