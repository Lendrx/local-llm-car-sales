"""
Vergleichsanalyse: RAG vs. Finetuning
-------------------------------------
Dieses Skript vergleicht die Antworten des RAG-Ansatzes und des finetuned Modells
auf die gleichen Fragen und ermöglicht eine direkte Bewertung.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import get_retriever
import time
import os

# Testfragen für den Vergleich
TEST_QUESTIONS = [
    "Welche BMW-Ausstattungsmerkmale sind bei Kunden besonders beliebt?",
    "Vergleiche die Preise zwischen BMW SUVs und Limousinen.",
    "Welche BMW-Modelle bieten das beste Preis-Leistungs-Verhältnis für Familien?",
    "Ist die BMW M-Serie bei Kunden beliebter als die regulären Modelle?",
    "Welche BMW-Modelle sind am erfolgreichsten in der Westregion?"
]

def setup_models():
    """Richtet beide Modelle ein und gibt die entsprechenden Chains zurück."""
    # Daten-Pfad überprüfen
    data_path = "data/bmw_vehicles.csv"
    if not os.path.exists(data_path):
        data_path = "../data/bmw_vehicles.csv"
    
    # RAG-Modell einrichten
    retriever = get_retriever(csv_path=data_path, db_path="vector_db")
    standard_model = OllamaLLM(model="llama3.2")
    
    rag_template = """
    Du bist ein Experte für BMW-Fahrzeuge und Verkaufsdaten.

    Hier sind einige relevante BMW-Verkaufsdaten, die dir bei der Beantwortung helfen können:
    {bmw_data}

    Hier ist die Frage, die du beantworten sollst: {question}

    Nutze die bereitgestellten Verkaufsdaten, um eine detaillierte und fundierte Antwort zu geben.
    """
    
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = rag_prompt | standard_model
    
    # Finetuned-Modell einrichten
    finetuned_model = OllamaLLM(model="bmw-expert")
    
    finetuned_template = """
    Beantworte die folgende Frage zu BMW-Fahrzeugen und Verkaufsdaten:

    Frage: {question}
    """
    
    finetuned_prompt = ChatPromptTemplate.from_template(finetuned_template)
    finetuned_chain = finetuned_prompt | finetuned_model
    
    return retriever, rag_chain, finetuned_chain

def compare_models(question, retriever, rag_chain, finetuned_chain):
    """Vergleicht die Antworten beider Modelle für eine gegebene Frage."""
    print(f"\n\n===== FRAGE: {question} =====\n")
    
    # RAG-Modell testen
    start_time = time.time()
    bmw_data = retriever.invoke(question)
    rag_result = rag_chain.invoke({"bmw_data": bmw_data, "question": question})
    rag_time = time.time() - start_time
    
    print(f"--- RAG-MODELL (Zeit: {rag_time:.2f}s) ---")
    print(rag_result)
    
    # Finetuned-Modell testen
    start_time = time.time()
    finetuned_result = finetuned_chain.invoke({"question": question})
    finetuned_time = time.time() - start_time
    
    print(f"\n--- FINETUNED-MODELL (Zeit: {finetuned_time:.2f}s) ---")
    print(finetuned_result)
    
    # Menschliche Bewertung
    print("\n--- BEWERTUNG ---")
    print("Welches Modell hat die bessere Antwort geliefert?")
    print("1: RAG-Modell")
    print("2: Finetuned-Modell")
    print("3: Beide ähnlich gut")
    rating = input("Deine Bewertung (1/2/3): ")
    return rating

def main():
    """Hauptfunktion zum Vergleich der beiden Ansätze."""
    print("BMW-MODELL-VERGLEICH: RAG vs. FINETUNED")
    print("======================================")
    print("Dieses Skript vergleicht die Antworten des RAG-Ansatzes und des finetuned Modells.")
    
    # Prüfe, ob das finetuned Modell existiert
    try:
        test_model = OllamaLLM(model="bmw-expert")
        test_model.invoke("Test")
    except Exception as e:
        print("Fehler: Das finetuned Modell 'bmw-expert' ist nicht verfügbar.")
        print("Bitte erstelle es zuerst mit './create_model.sh'")
        return
    
    # Modelle einrichten
    print("Initialisiere Modelle...")
    retriever, rag_chain, finetuned_chain = setup_models()
    
    # Vergleichsmodus wählen
    print("\nVergleichsmodus wählen:")
    print("1: Vordefinierte Testfragen (5 Fragen)")
    print("2: Eigene Fragen stellen")
    mode = input("Modus (1/2): ")
    
    results = []
    
    if mode == "1":
        # Vordefinierte Testfragen
        for i, question in enumerate(TEST_QUESTIONS):
            print(f"\nTest {i+1}/{len(TEST_QUESTIONS)}")
            rating = compare_models(question, retriever, rag_chain, finetuned_chain)
            results.append({"question": question, "rating": rating})
    else:
        # Eigene Fragen
        while True:
            question = input("\nDeine Frage (oder 'q' zum Beenden): ")
            if question.lower() in ["q", "quit", "exit", "ende"]:
                break
            
            rating = compare_models(question, retriever, rag_chain, finetuned_chain)
            results.append({"question": question, "rating": rating})
    
    # Ergebnisse zusammenfassen
    if results:
        rag_wins = sum(1 for r in results if r["rating"] == "1")
        finetuned_wins = sum(1 for r in results if r["rating"] == "2")
        ties = sum(1 for r in results if r["rating"] == "3")
        
        print("\n\n=== ZUSAMMENFASSUNG ===")
        print(f"RAG-Modell besser: {rag_wins}/{len(results)}")
        print(f"Finetuned-Modell besser: {finetuned_wins}/{len(results)}")
        print(f"Beide ähnlich gut: {ties}/{len(results)}")
        
        if rag_wins > finetuned_wins:
            print("\nFazit: Der RAG-Ansatz scheint für deine Anwendung besser zu funktionieren.")
        elif finetuned_wins > rag_wins:
            print("\nFazit: Das Finetuning scheint für deine Anwendung besser zu funktionieren.")
        else:
            print("\nFazit: Beide Ansätze funktionieren ähnlich gut. Du könntest einen hybriden Ansatz in Betracht ziehen.")

if __name__ == "__main__":
    main()
