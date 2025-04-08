"""
RAG-Anwendung für BMW-Fahrzeugdaten
-----------------------------------
Dieser Ansatz verwendet Retrieval-Augmented Generation (RAG), um Fragen 
zu BMW-Fahrzeugen zu beantworten, indem relevante Daten aus einem Vektorspeicher 
abgerufen und an das LLM weitergegeben werden.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import get_retriever
import os

def main():
    # Stelle sicher, dass der Datenpfad korrekt ist
    data_path = "data/bmw_vehicles.csv"
    if not os.path.exists(data_path):
        # Versuche den relativen Pfad aus dem übergeordneten Verzeichnis
        data_path = "../data/bmw_vehicles.csv"
        if not os.path.exists(data_path):
            print(f"Fehler: Datendatei nicht gefunden. Bitte stellen Sie sicher, dass '{data_path}' existiert.")
            return

    # Einrichtung des Retrievers für die BMW-Fahrzeugdaten
    print("Initialisiere den Vektorspeicher für die BMW-Fahrzeugdaten...")
    retriever = get_retriever(csv_path=data_path, db_path="vector_db")
    
    # Konfiguration des LLM
    print("Initialisiere das Sprachmodell...")
    model = OllamaLLM(model="llama3.2", temperature=0.7, timeout=120)
    
    # Erstellung der Prompts für das RAG-System
    template = """
    Du bist ein Experte für BMW-Fahrzeuge und Verkaufsdaten.

    Hier sind einige relevante BMW-Verkaufsdaten, die dir bei der Beantwortung helfen können:
    {bmw_data}

    Hier ist die Frage, die du beantworten sollst: {question}

    Nutze die bereitgestellten Verkaufsdaten, um eine detaillierte und fundierte Antwort zu geben.
    Wenn du nach bestimmten Modellen, Serien, Ausstattungsmerkmalen oder Trends gefragt wirst, 
    basiere deine Antwort auf den dir zur Verfügung stehenden Daten.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    # Starte die Interaktionsschleife
    print("\n" + "="*50)
    print("BMW-Verkaufsdaten-Assistent (RAG-Ansatz)")
    print("Stelle Fragen zu BMW-Modellen, Verkaufszahlen, Kundenzufriedenheit, usw.")
    print("="*50)
    
    while True:
        print("\n")
        question = input("Deine Frage (q zum Beenden): ")
        print("\n")
        
        if question.lower() in ["q", "quit", "exit", "ende"]:
            break
        
        print("Suche nach relevanten Informationen...")
        bmw_data = retriever.invoke(question)
        
        print("Generiere Antwort...\n")
        result = chain.invoke({"bmw_data": bmw_data, "question": question})
        
        print("Antwort:\n" + "-"*50)
        print(result)
        print("-"*50)

if __name__ == "__main__":
    main()
