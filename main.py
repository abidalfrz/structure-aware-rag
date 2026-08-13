from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import retriever  
import time
from warnings import filterwarnings

filterwarnings("ignore")


model = OllamaLLM(model="qwen2.5:3b", temperature=0.3)

template = """
Anda adalah asisten resmi untuk kompetisi Sebelas Maret Statistics Data Science (SSDS) 2026.

Berikut adalah potongan informasi resmi dari Buku Panduan SSDS 2026:
{context}

Pertanyaan: {question}

Instruksi:
1. Jawab pertanyaan HANYA berdasarkan informasi yang diberikan pada konteks di atas.
2. Jika informasi tidak terdapat dalam konteks, jawab dengan jujur: "Maaf, informasi tersebut tidak tercantum dalam Buku Panduan."
3. Berikan jawaban yang jelas, terstruktur, dan profesional.
4. Hilangkan elemen markdown, simbol, atau karakter, seperti #, *, >, _, dan lainnya yang dapat mengganggu kejelasan jawaban.
Jawaban:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("==================================================")
print("      Welcome to the Document Q&A Assistant!")
print("==================================================")

while True:
    question = input("\nEnter your question (or type 'q' to quit): ")
    if question.lower() == 'q':
        break

    start_time = time.time()
    
    retrieved_docs = retriever.invoke(question)
    
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        bab_utama = doc.metadata.get('Bab Utama', 'Bagian Umum')
        sub_bab = doc.metadata.get('Sub Bab', '')
        
        header_info = f"Bagian: {bab_utama}" + (f" - {sub_bab}" if sub_bab else "")
        context_parts.append(f"--- ({header_info}) ---\n{doc.page_content}")
        
    context_text = "\n\n".join(context_parts)
    
    for i, doc in enumerate(retrieved_docs):
        bab = doc.metadata.get('Bab Utama', 'N/A')
        print(f"Chunk {i+1} | Bab: {bab} | Preview: {doc.page_content.replace(chr(10), ' ')[:80]}...")
    print("-------------------------\n")

    result = chain.invoke({   
        'context': context_text,
        'question': question
    })
    
    end_time = time.time()

    print("Answer:")
    print(result)
    print(f"\n[Time Execution: {(end_time - start_time):.2f} s]")