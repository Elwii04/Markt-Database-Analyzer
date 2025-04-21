# Projekt: Analyse von Ad-Daten

## Projektbeschreibung

Dieses Python-Skript dient der Analyse und Auswertung von Daten die über einen Scraper gesammelt wurde. Es liest Daten aus einer SQLite-Datenbank, führt verschiedene Datenverarbeitungsschritte durch (wie das Zusammenführen von Tabellen, Filtern und Berechnen von Zeitdifferenzen) und visualisiert die Ergebnisse, z.B. die durchschnittlichen Views nach Stadt oder die Verteilung von Ad-Pushes über den Tag.

## Funktionsweise

1.  **Datenbankzugriff & Setup:**
    *   Importiert notwendige Bibliotheken: `sqlite3`, `pandas`, `numpy`, `matplotlib.pyplot`, `matplotlib.ticker` [1].
    *   Definiert den Pfad zur SQLite-Datenbank (`db_path`) [1].
    *   Stellt eine Verbindung zur Datenbank her und liest Daten aus den Tabellen `monitor_records` und `profiles` in Pandas DataFrames [1].
    *   Schließt die Datenbankverbindung [1].

2.  **Datenzusammenführung & -bereinigung:**
    *   Führt die DataFrames `df_monitor` und `df_profiles` anhand der Spalten `ad_unique_id` und `push_counter` zusammen (`inner join`) [1].
    *   Filtert Zeilen heraus, bei denen die Spalte `age` leer oder `NULL` ist [1].
    *   Konvertiert die `age`-Spalte in einen numerischen Typ (`age_numeric`) und entfernt Zeilen, bei denen die Konvertierung fehlschlägt [1].

3.  **Analyse & Visualisierung (Beispiele aus dem Kontext):**
    *   **Performance nach Stadt:** Berechnet durchschnittliche Views (vermutlich in den ersten 6 Stunden) pro Stadt, sortiert die Ergebnisse und visualisiert sie als horizontales Balkendiagramm (`barh`) mit `matplotlib`. Die Achsen werden beschriftet, ein Titel wird gesetzt und die Stadt mit den höchsten Views wird oben angezeigt [1].
    *   **Performance nach Tageszeit:** Berechnet die durchschnittliche View-Rate pro Stunde, glättet diese Rate (vermutlich mit einem gleitenden Durchschnitt) und visualisiert sie als Liniendiagramm. Auf einer zweiten Y-Achse wird die Anzahl der neuen Ad-Pushes pro Zeitintervall als Balkendiagramm dargestellt. Die Achsen werden entsprechend beschriftet und formatiert [1].

## Voraussetzungen

*   Python 3.x
*   Eine SQLite-Datenbankdatei am unter `db_path` angegebenen Ort [1]. Diese Datenbank muss die Tabellen `monitor_records` und `profiles` mit den im Skript verwendeten Spalten enthalten (z.B. `ad_unique_id`, `push_counter`, `track_time`, `ad_age_in_minutes`, `views`, `age`, `profile_url` etc.) [1].
*   Die unten aufgeführten Python-Bibliotheken.

## Installation & Abhängigkeiten

Die benötigten Python-Bibliotheken sind in der Datei `requirements.txt` aufgeführt:

```text
pandas
numpy
matplotlib
