#!/bin/bash
# Skript zum Erstellen des finetuned BMW-Expert-Modells mit Ollama

echo "Erstelle das finetuned BMW-Expert-Modell..."

# Prüfe, ob Ollama installiert ist
if ! command -v ollama &> /dev/null; then
    echo "Fehler: Ollama ist nicht installiert. Bitte installiere Ollama zuerst."
    echo "Installation: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Prüfe, ob das Basis-Modell vorhanden ist
if ! ollama list | grep -q "llama3.2"; then
    echo "Basis-Modell llama3.2 wird heruntergeladen..."
    ollama pull llama3.2
fi

# Erstelle das finetuned Modell
echo "Erstelle das BMW-Expert-Modell..."
ollama create bmw-expert -f models/bmw_expert.modelfile

echo "BMW-Expert-Modell wurde erfolgreich erstellt!"
echo "Du kannst es jetzt mit 'python finetuned_app.py' verwenden."
