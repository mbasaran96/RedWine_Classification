# RedWine_Classification
Im folgenden Projekt soll eine Vorhersage der Weinqualität mithilfe eines Machine Learning Verfahrens erfolgen. <br>
Nachdem die Weinqualität in 6 Kategorien (3 - 8) eingestuft wird, soll diese **Mehrklassenklassifizierung** durch eine **Support Vector Machine** durchgeführt werden. <br>

Der Rotwein Datensatz (*winequality-red.csv*) wurde von folgender Quelle heruntergeladen: <br>
https://archive.ics.uci.edu/dataset/186/wine+quality

Im unteren Abschnitt werden die Pipelines für **Preprocessing** und das **Modelltraining** näher erläutert.

# RedWine_eda_preprocessing

Im Jupyter Notebook (*RedWine_eda_preprocessing.ipynb*) werden folgende Schritte zur explorativen Datenanalyse und Datenvorbereitung durchgeführt:
- Datensatz laden
- Überblick über Daten verschaffen
  - Zeilen- und Spaltenanzahl
  - Anzahl der Einträge pro Spalte
  - Fehlende Werte untersuchen, falls vorhanden
  - Duplikate entfernen, falls vorhanden
- Zielspalte (Quality) und ihre Verteilung näher betrachen
- Train/Test Split
  - Pandas dataframe in numpy tranformieren
  - train_test_split durchführen
- Verteilungen der verschiedenen Klassen untersuchen
- Korrelationsmatrix der Trainingsdaten analysieren
  - Spalten mit hoher Korrelation genauer betrachten
    - Deskriptive Statistik
    - Korrelationsmatrix
- Transformieren und Skalieren
  - Gegenüberstellung des PowerTransformers und QuantileTransformers für eine symmetrische Verteilung der Daten
  - Visualisierung mithilfe von Histogrammen
- FeatureEngineering
  - Mit der Erkenntnis aus vorheriger Korrelationen neue Spalten hinzufügen
  - Korrelationsmatrix im Hinblick auf die Zielspalte analysieren
- Make_Pipeline für die Datenvorverarbeitung 
  - PowerTransformer: Normalisiert die Verteilung der Daten
  - CustomFeatureTransformer: Hinzufügen von Spalten
  - StandardScaler: Skaliert die Features auf eine einheitliche Standardnormalverteilung
    - Transformierte Daten abspeichern
    - Pipeline abspeichern

# RedWine_SVC

Das Jupyter Notebook (*RedWine_SVC.ipynb*) beschreibt den Algorithmus für die Support Vector Machine:
- Bibliotheken laden
- Datensatz für Training und Test mit NumPy laden
- Test Daten mithilfe des CustomFeatureTransformers transformieren
- GridSearch für den SVC anwenden, um die besten Parameter zu erhalten
- Auswertungen nach dem Fit ausgeben lassen
- Confusion Matrix für den besten Estimator aus den Trainingsdaten
- Vorhersagen auf Trainings- und Testdaten
  - RMSE
  - Accuracy
  - F1-Score
  - Confusion Matrix
  - Classification Report

