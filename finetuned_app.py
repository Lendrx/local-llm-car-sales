#!/usr/bin/env python3
"""
Finetuned-Modell-Anwendung für BMW-Fahrzeugdaten
-----------------------------------------------
Dieser Ansatz verwendet ein finetuned Modell, um Fragen zu BMW-Fahrzeugen 
direkt zu beantworten, ohne zusätzliche Daten zur Laufzeit abzurufen.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def main():
    print("Initialisiere das finetuned BMW-Expertenmodell...")
    # Verwende das finetuned Modell - stelle sicher, dass es vorher erstellt wurde
    model = OllamaLLM(model="bmw-expert", temperature=0.7)
    
    # Einfacher Prompt-Template für das finetuned Modell
    template = """
    Beantworte die folgende Frage zu BMW-Fahrzeugen und Verkaufsdaten:

    Frage: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    # Starte die Interaktionsschleife
    print("\n" + "="*50)
    print("BMW-Verkaufsdaten-Assistent (Finetuned-Modell)")
    print("Stelle Fragen zu BMW-Modellen, Verkaufszahlen, Kundenzufriedenheit, usw.")
    print("="*50)
    
    while True:
        print("\n")
        question = input("Deine Frage (q zum Beenden): ")
        print("\n")
        
        if question.lower() in ["q", "quit", "exit", "ende"]:
            break
        
        print("Generiere Antwort...\n")
        result = chain.invoke({"question": question})
        
        print("Antwort:\n" + "-"*50)
        print(result)
        print("-"*50)

if __name__ == "__main__":
    main()
