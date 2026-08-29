import json
import os
import pandas as pd
import google.generativeai as genai
from pydantic import BaseModel, Field
import time

# Konfiguration
HISTORY_FILE = "data/polymarket_history.csv"
CLASSIFIED_FILE = "data/polymarket_classified.csv"

# API-Schluessel aus Umgebungsvariable (Protokoll §10.7 - NIE im Klartext im Code!)
# Setzen mit: export GEMINI_API_KEY="..."  (bzw. AWS Secrets Manager in der Lambda)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY nicht gesetzt (Umgebungsvariable erforderlich).")
genai.configure(api_key=GEMINI_API_KEY)

# 1. Das aktualisierte JSON-Schema (Neuer Cluster-Name)
class MarketClassification(BaseModel):
    cluster: str = Field(
        description="Eines von: 'Geopolitischer & Militärischer Schock', 'Systemische Stabilitätsrisiken', 'Makroökonomie & Zentralbanken', 'Politische Struktur & Wahlen', 'Sektorale & Unternehmensereignisse', 'Keine Relevanz'"
    )
    zone: int = Field(
        description="Muss 1 (Kill-Switch), 2 (Manager-Radar) oder 3 (Rauschen) sein."
    )
    impact_score: int = Field(
        description="Bewertung der globalen Zerstörungskraft von 1 bis 100."
    )
    reasoning: str = Field(
        description="Max. 3 Sätze. Begründe zwingend die Wahl der Zone und die Höhe des Impact-Scores."
    )
    confidence: int = Field(
        description="Sicherheit der Gesamtklassifikation von 1 bis 100."
    )
    # FIX 4 (Protokoll §10.4): Wirkrichtung + Normalisierungs-Flag
    risk_direction: str = Field(
        description="'RISK_OFF' wenn die JA-Seite des Kontrakts ein risikoerhöhendes Ereignis bezeichnet, 'RISK_ON' wenn die JA-Seite ein risikosenkendes/normalisierendes Ereignis bezeichnet."
    )
    is_normalization: bool = Field(
        description="True, wenn der Kontrakt eine Rückkehr zur Normalität bezeichnet (Entwarnungs-Kanal, z.B. 'traffic returns to normal')."
    )
    direction_reasoning: str = Field(
        description="1 Satz: Warum diese Wirkrichtung? (Pflichtfeld, Audit-Log)"
    )

def classify_market(market_title):
    # --- Few-Shot Feedback laden ---
    feedback_memory = ""
    feedback_path = "../data/human_feedback.csv"
    if os.path.exists(feedback_path):
        try:
            df_feedback = pd.read_csv(feedback_path)
            feedback_memory = "\nWICHTIG! LERNE AUS DIESEN KORREKTUREN DES PORTFOLIO-MANAGERS:\n"
            for _, r in df_feedback.iterrows():
                feedback_memory += f"- Markt '{r['title']}' -> Cluster '{r['correct_cluster']}', Zone {r['correct_zone']}. Grund: {r['reason']}\n"
            feedback_memory += "Wende diese Logik strikt auf ähnliche zukünftige Märkte an!\n"
        except Exception as e:
            pass

    # --- Der finale, strukturierte Prompt (mit Black Swan Update) ---
    prompt = f"""
    Du bist ein quantitativer Risikoanalyst für ein globales Multi-Faktor-Portfolio.
    Deine Aufgabe ist es, Prognosemärkte nach ihrer Relevanz für globale Finanzmärkte zu bewerten.

    {feedback_memory}

    MARKT-TITEL: "{market_title}"

    Führe die Analyse in drei Schritten durch:

    SCHRITT 1: KATEGORISIERUNG (Das "Was")
    Wähle exakt EINES der folgenden 5 Cluster (oder "Keine Relevanz"):
    1. "Geopolitischer & Militärischer Schock": Kriegsausbrüche, nukleare Bedrohungen, massive Blockaden globaler Handelsrouten.
    2. "Systemische Stabilitätsrisiken": Unkontrollierbare Pandemien, globaler Ausfall kritischer Infrastruktur, Kollaps des Finanzsystems, extreme Naturkatastrophen.
    3. "Makroökonomie & Zentralbanken": Leitzinsen, Inflation, Währungsentwicklungen.
    4. "Politische Struktur & Wahlen": Nationale Wahlen, Regierungsbildungen, globale Regulatorik.
    5. "Sektorale & Unternehmensereignisse": KI-Zyklen, Mega-IPOs, marktbewegende Tech-Releases.
    (Falls inhaltlich völlig irrelevant für Finanzen: "Keine Relevanz")

    SCHRITT 2: ZONEN-ZUTEILUNG (Die harte Weiche)
    Bewerte, wie akut das Risiko HEUTE ist. Wähle exakt EINE Zone (1, 2 oder 3):
    - ZONE 1 (Kill-Switch): Globale, unkorrelierte Systemrisiken. Plötzliche Ereignisse, die das Potenzial für panikartige Abverkäufe über ALLE Anlageklassen hinweg haben. (Hauptsächlich Cluster 1 & 2).
    - ZONE 2 (Manager-Radar): Informelle, makroökonomische Entwicklungen. Verändern langfristige Spielregeln, verursachen aber HEUTE keine akute unkontrollierbare Panik. (Hauptsächlich Cluster 3, 4 & 5).
    - ZONE 3 (Rauschen): Krypto-Memes, Popkultur, Klatsch, Sport.
      WICHTIGER HINWEIS: Hypothetische, aber katastrophale Ereignisse (z.B. Meteoriteneinschlag, Ausbruch eines Supervulkans, nuklearer Zwischenfall) sind KEIN Rauschen. Sie gehören zwingend in Cluster 2 ('Systemische Stabilitätsrisiken') und ZONE 1. Aliens/Zeitreisen bleiben Zone 3.

    SCHRITT 3: IMPACT SCORE (1-100)
    Bewerte die globale Zerstörungskraft oder Marktrelevanz auf einer Skala von 1 bis 100.
    - Bei Zone 1: Wie verheerend wäre der globale Crash? (z.B. Weltkrieg = 99, regionaler Konflikt = 60).
    - Bei Zone 2: Wie stark wird der globale Markt umgeschichtet?
    - Bei Zone 3: Immer 1 bis 10.

    SCHRITT 4: WIRKRICHTUNG (kritisch für die Signalverarbeitung!)
    Prüfe: Bezeichnet die JA-Seite dieses Kontrakts ein risikoERHÖHENDES oder ein risikoSENKENDES Ereignis?
    - "RISK_OFF": JA-Eintritt wäre schlecht für Aktienmärkte (Krieg bricht aus, Blockade beginnt, Regime stürzt).
    - "RISK_ON": JA-Eintritt wäre gut/entspannend für Aktienmärkte (Verkehr normalisiert sich, Waffenstillstand hält, Konflikt endet).
    ACHTUNG: Ein Kontrakt wie "Strait of Hormuz traffic returns to normal" ist RISK_ON — eine steigende Quote ist dort ENTWARNUNG, nicht Gefahr!
    Setze zusätzlich "is_normalization": true, wenn der Kontrakt explizit eine Rückkehr zur Normalität bezeichnet (Entwarnungs-Kanal).
    Begründe die Wirkrichtung in einem Satz ("direction_reasoning").

    Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt. Format:
    {{
        "cluster": "Name des Clusters",
        "zone": 1,
        "impact_score": 85,
        "confidence": 95,
        "reasoning": "Begründung.",
        "risk_direction": "RISK_OFF",
        "is_normalization": false,
        "direction_reasoning": "Begründung der Wirkrichtung."
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text.strip())
        
        if result.get("zone") not in [1, 2, 3]:
            result["zone"] = 3
        if not isinstance(result.get("impact_score"), int):
            result["impact_score"] = 10
        # FIX 4: Wirkrichtungs-Felder validieren.
        # Konservativer Default ist RISK_OFF (falsch-negativ waere gefaehrlicher),
        # aber der manuelle Override im Dashboard bleibt die letzte Instanz.
        if result.get("risk_direction") not in ["RISK_OFF", "RISK_ON"]:
            result["risk_direction"] = "RISK_OFF"
            result["direction_reasoning"] = "Default (LLM lieferte keine valide Richtung) - manuell pruefen!"
        if not isinstance(result.get("is_normalization"), bool):
            result["is_normalization"] = False

        return result
        
    except Exception as e:
        print(f"Fehler bei Klassifizierung: {e}")
        return {"cluster": "Error", "zone": 3, "impact_score": 0, "confidence": 0, "reasoning": str(e),
                "risk_direction": "RISK_OFF", "is_normalization": False,
                "direction_reasoning": "Error-Fallback - manuell pruefen!"}

def process_unclassified_markets():
    if not os.path.exists(HISTORY_FILE):
        print(f"Keine Daten gefunden unter {HISTORY_FILE}")
        return
        
    df_raw = pd.read_csv(HISTORY_FILE)
    df_active = df_raw[df_raw['status'] == 'active'].copy()

    processed_ids = set()
    if os.path.exists(CLASSIFIED_FILE):
        try:
            df_existing = pd.read_csv(CLASSIFIED_FILE)
            if 'market_id' in df_existing.columns:
                processed_ids = set(df_existing['market_id'].astype(str))
            print(f"✅ Checkpoint: {len(processed_ids)} Märkte bereits klassifiziert.")
        except pd.errors.EmptyDataError:
            pass

    df_todo = df_active[~df_active['market_id'].astype(str).isin(processed_ids)]
    print(f"⏳ Noch zu verarbeiten: {len(df_todo)} Märkte.\n")

    for index, row in df_todo.iterrows():
        market_id = str(row['market_id'])
        title = row['title']

        print(f"Analysiere: {title}")
        result = classify_market(title)

        row_dict = row.to_dict()
        row_dict.update(result)
        
        df_single = pd.DataFrame([row_dict])
        header_needed = not os.path.exists(CLASSIFIED_FILE)
        df_single.to_csv(CLASSIFIED_FILE, mode='a', header=header_needed, index=False)

        print(f" -> Cluster: {result.get('cluster')} | Zone: {result.get('zone')} | Impact: {result.get('impact_score')}\n")
        time.sleep(0.1)

    print("🎉 Alle Märkte erfolgreich klassifiziert!")

if __name__ == "__main__":
    process_unclassified_markets()