# BMW LLM-Projekt: RAG vs. Finetuning

Dieses Lernprojekt demonstriert zwei wichtige Ansätze zur Verbesserung von LLM-Antworten für domänenspezifische Fragen:

1. **RAG (Retrieval-Augmented Generation)** - Ergänzt das LLM mit relevanten Daten aus einer Wissensdatenbank
2. **Finetuning** - Passt das LLM selbst an eine spezifische Domäne an

Am Beispiel von BMW-Fahrzeugdaten werden beide Methoden implementiert und verglichen.

## Projektstruktur

```
bmw_llm_simplified/
│
├── data/
│   ├── bmw_vehicles.csv     # BMW-Fahrzeugdaten
│   └── training_data.json   # Finetuning-Daten (Frage-Antwort-Paare)
│
├── models/
│   └── bmw_expert.modelfile # Ollama-Modelfile für Finetuning
│
├── vector_db/              # Wird erstellt für den Vektorspeicher
│
├── vector_store.py         # Implementierung des Vektorspeichers
├── rag_app.py              # RAG-Anwendung
├── finetuned_app.py        # Finetuned-Modell-Anwendung
├── compare_approaches.py   # Vergleichsskript für beide Ansätze
├── create_model.sh         # Skript zum Erstellen des finetuned Modells
└── README.md               # Projektdokumentation
```

## Voraussetzungen

- Python 3.8+
- [Ollama](https://ollama.com) installiert und konfiguriert
- Das Llama3.2-Basismodell (`ollama pull llama3.2`)
- Das mxbai-embed-large-Modell für Einbettungen (`ollama pull mxbai-embed-large`)

## Installation

1. Klone das Repository oder entpacke das Projektarchiv
2. Installiere die Python-Abhängigkeiten:

```bash
pip install langchain langchain-ollama langchain-chroma pandas
```

## Verwendung

### 1. RAG-Ansatz testen

Der RAG-Ansatz verwendet einen Vektorspeicher, um relevante BMW-Fahrzeugdaten zu finden und diese dem LLM zur Verfügung zu stellen:

```bash
python rag_app.py
```

Beim ersten Start wird ein Vektorspeicher mit den BMW-Fahrzeugdaten erstellt, was einige Minuten dauern kann.

### 2. Finetuned-Modell erstellen und testen

Zuerst muss das finetuned Modell erstellt werden:

```bash
chmod +x create_model.sh
./create_model.sh
```

Anschließend kann das finetuned Modell getestet werden:

```bash
python finetuned_app.py
```

### 3. Beide Ansätze vergleichen

Mit dem Vergleichsskript können beide Ansätze direkt gegenübergestellt werden:

```bash
python compare_approaches.py
```

Das Skript ermöglicht entweder vordefinierte Testfragen zu verwenden oder eigene Fragen zu stellen und die Qualität der Antworten zu bewerten.

## Unterschiede zwischen den Ansätzen

### RAG-Ansatz

**Vorteile:**
- Kann mit aktuellen/neuen Daten arbeiten
- Präzise Antworten auf Basis der verfügbaren Daten
- Geringeres Risiko von Halluzinationen

**Nachteile:**
- Langsamere Antwortzeit (Retrieval + Generierung)
- Qualität abhängig von der Retrieval-Leistung
- Nur Wissen aus den verfügbaren Daten

### Finetuning-Ansatz

**Vorteile:**
- Schnellere Antwortzeit
- Funktioniert komplett offline
- Kann implizites Wissen anwenden

**Nachteile:**
- Statisches Wissen (Aktualisierung erfordert erneutes Training)
- Kann bei unbekannten Fragen halluzinieren
- Training benötigt qualitativ hochwertige Trainingsdaten

## Lernziele

Dieses Projekt hilft dabei, folgende Konzepte zu verstehen:

1. **Vektordatenbanken:** Wie Daten für semantische Suche aufbereitet werden
2. **Prompt Engineering:** Optimale Gestaltung von Prompts für kontextbezogene Antworten
3. **Finetuning von LLMs:** Anpassung von Sprachmodellen für spezifische Anwendungsfälle
4. **LangChain Framework:** Verwendung für RAG-Pipelines und LLM-Integrationen
5. **Evaluierung von LLM-Antworten:** Vergleich und Bewertung verschiedener Ansätze

## Weiterführende Ideen

- **Hybridansatz:** Kombiniere beide Methoden für bessere Ergebnisse
- **Datenanalyse:** Führe explorative Datenanalyse der BMW-Fahrzeugdaten durch
- **Erweitertes Finetuning:** Verwende fortgeschrittenere Finetuning-Methoden wie LoRA/QLoRA
- **Web-Interface:** Erstelle eine Benutzeroberfläche mit Streamlit oder Gradio
- **Automatisierte Evaluation:** Entwickle quantitative Metriken für die Antwortqualität

## Ressourcen zum Weiterlesen

- [LangChain Dokumentation](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Dokumentation](https://github.com/ollama/ollama)
- [RAG vs. Finetuning](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
