# Fahrzeugdaten - LLM Projekt

Dieses Projekt vergleicht zwei Methoden, um LLMs für Fahrzeugdaten zu optimieren.

## Was macht das Projekt?

- RAG-Ansatz: Verwendet Daten aus einer Vektordatenbank
- Finetuning-Ansatz: Trainiert ein angepasstes Modell

## Wie starte ich das Projekt?

### Voraussetzungen

- Python installieren
- Ollama installieren
- Modelle herunterladen:
  ```
  ollama pull llama3.2
  ollama pull mxbai-embed-large
  ```

### Installation

```
pip install -r requirements.txt
```

### Verwendung

1. RAG-Ansatz testen:
```
python rag_app.py
```

2. Finetuned-Modell erstellen und testen:
```
chmod +x create_model.sh
./create_model.sh
python finetuned_app.py
```

3. Beide Ansätze vergleichen:
```
python compare_approaches.py
```

## Projektstruktur

- `data/` - CSV-Daten und Trainingsdaten
- `models/` - Modelfile für Finetuning
- `rag_app.py` - RAG-Implementierung
- `finetuned_app.py` - Finetuned-Modell-Implementierung
- `compare_approaches.py` - Vergleicht beide Ansätze
