# Einrichtung des Vektorspeichers für BMW-Fahrzeugdaten
import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class BMWVectorStore:
    def __init__(self, csv_path="data/bmw_vehicles.csv", db_path="vector_db"):
        """Initialisiert den Vektorspeicher für BMW-Fahrzeugdaten."""
        self.csv_path = csv_path
        self.db_path = db_path
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Überprüfe, ob die Datenbank bereits existiert
        self.add_documents = not os.path.exists(self.db_path)
    
    def setup(self):
        """Richtet den Vektorspeicher ein und lädt Daten wenn nötig."""
        # Erstelle den Vektorspeicher
        self.vector_store = Chroma(
            collection_name="bmw_vehicles",
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        
        # Füge Dokumente hinzu, wenn die Datenbank neu ist
        if self.add_documents:
            self._add_documents_from_csv()
        
        # Erstelle einen Retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}  # Anzahl der zurückgegebenen Dokumente
        )
        
        return self.retriever
    
    def _add_documents_from_csv(self):
        """Liest BMW-Daten aus CSV und erstellt Dokumente für den Vektorspeicher."""
        print("Lade Fahrzeugdaten in den Vektorspeicher...")
        
        # Lade die CSV-Daten
        df = pd.read_csv(self.csv_path)
        
        documents = []
        ids = []
        
        for i, row in df.iterrows():
            # Erstelle einen Text mit allen wichtigen Informationen zu einem Auto
            page_content = f"Modell: {row['Model']} {row['Series']} ({row['Year']}), " + \
                          f"Preis: €{row['Price']}, Kilometerstand: {row['Mileage']} km, " + \
                          f"Kraftstoffart: {row['Fuel_Type']}, Getriebe: {row['Transmission']}, " + \
                          f"Farbe: {row['Color']}, Verkauft bei Händler: {row['Dealership']} in der Region {row['Region']} am {row['Sale_Date']}, " + \
                          f"Merkmale: {row['Features']}, Leistungsbewertung: {row['Performance_Rating']}/10.0, " + \
                          f"Kundenzufriedenheit: {row['Customer_Satisfaction']}/5.0, Tage im Lager: {row['Days_In_Stock']}"
            
            # Speichere zusätzliche Informationen als Metadaten
            metadata = {
                "model": row["Model"],
                "series": row["Series"],
                "year": row["Year"],
                "price": row["Price"],
                "fuel_type": row["Fuel_Type"],
                "performance_rating": row["Performance_Rating"],
                "customer_satisfaction": row["Customer_Satisfaction"]
            }
            
            # Erstelle ein Dokument mit dem Text und den Metadaten
            document = Document(
                page_content=page_content,
                metadata=metadata,
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
        
        # Füge die Dokumente in den Vektorspeicher ein
        self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"{len(documents)} Fahrzeugdaten erfolgreich geladen!")

# Hilfsklasse zum einfachen Abrufen des eingerichteten Retrievers
def get_retriever(csv_path="data/bmw_vehicles.csv", db_path="vector_db"):
    """Gibt einen eingerichteten Retriever für BMW-Fahrzeugdaten zurück."""
    store = BMWVectorStore(csv_path, db_path)
    return store.setup()
