import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px
from datetime import datetime

# --- FREEZE-KONFIGURATION (Abgabe-Version, Streamlit Community Cloud) ---
FREEZE_MODE = True
FREEZE_STAND = "29.08.2026 (Tag 13 der Messung)"   # bei Bedarf am Samstag final anpassen

# RiskManager Import
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from risk_manager import RiskManager

st.set_page_config(page_title="Risk Manager Command Center", layout="wide")

# --- PFADE ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- DATEN LADEN ---
@st.cache_data(ttl=60)
def load_data():
    history_file = BASE_DIR / "data" / "polymarket_history.csv"
    classified_file = BASE_DIR / "data" / "polymarket_classified.csv"
    
    if not history_file.exists() or not classified_file.exists():
        return pd.DataFrame()
        
    df_live = pd.read_csv(history_file, dtype={'market_id': str})
    df_class = pd.read_csv(classified_file, dtype={'market_id': str})
    
    # NEU: Abwärtskompatibilität, falls Skript 2 noch nicht neu lief
    if 'impact_score' not in df_class.columns:
        df_class['impact_score'] = 50 # Standardwert setzen
        
    # NEU: Leere Cluster-Namen (NaN) mit Text füllen, um den Sortier-Fehler zu verhindern
    df_class['cluster'] = df_class['cluster'].fillna("Unklassifiziert")

    # FIX 4: Wirkrichtungs-Spalten mitladen (Abwaertskompatibilitaet fuer alte CSVs)
    if 'risk_direction' not in df_class.columns:
        df_class['risk_direction'] = 'RISK_OFF'
    if 'is_normalization' not in df_class.columns:
        df_class['is_normalization'] = False
    df_class['risk_direction'] = df_class['risk_direction'].fillna('RISK_OFF')

    df_class = df_class[['market_id', 'zone', 'cluster', 'impact_score', 'reasoning', 'risk_direction', 'is_normalization']]
    df_merged = pd.merge(df_live, df_class, on='market_id', how='inner')
    
    return df_merged

df = load_data()

if df.empty:
    st.warning("Keine vollständigen Daten gefunden. Bitte starte Skript 1 und 2.")
    st.stop()

# --- OVERRIDES LADEN (eine Wahrheit für alle Dashboard-Abschnitte) ---
# FREEZE_MODE: rein lokale Quelle (data/overrides.json, falls mitgeliefert) - kein
# boto3-Import, kein Client-Aufbau, keine Netzwerkabhängigkeit (Cloud-Robustheits-Sweep).
_BUCKET, _KEY = "blmodel-ml-input", "polymarket/overrides.json"
_OV_DEFAULT = {"impact": {}, "direction": {}, "mute": [], "zone": {}, "cluster": {}, "log": []}

if FREEZE_MODE:
    import json as _j
    _ov_lokal = BASE_DIR / "data" / "overrides.json"
    def _ov_laden():
        if _ov_lokal.exists():
            try:
                return _j.loads(_ov_lokal.read_text())
            except Exception:
                return dict(_OV_DEFAULT)
        return dict(_OV_DEFAULT)
else:
    import boto3 as _b3, json as _j
    _S3C = _b3.client("s3")
    def _ov_laden():
        try:
            return _j.loads(_S3C.get_object(Bucket=_BUCKET, Key=_KEY)["Body"].read())
        except Exception:
            return dict(_OV_DEFAULT)
_ov = _ov_laden()

# --- DATEN FILTERN & STRUKTURIEREN ---
df_active = df[df['status'] == 'active'].copy()
df_active = df_active[~df_active['market_id'].astype(str).isin(_ov.get("mute", []))]

# Wirksame Zone/Richtung/Cluster (CSV-Wert von _ov überblendet) - fuer Radar-Gruppierung & -Kennzeichnung (U3)
df_active['zone_wirksam'] = df_active.apply(
    lambda r: int(_ov.get("zone", {}).get(str(r['market_id']), r['zone'])), axis=1)
df_active['risk_direction_wirksam'] = df_active.apply(
    lambda r: _ov.get("direction", {}).get(str(r['market_id']), r['risk_direction']), axis=1)
df_active['cluster_wirksam'] = df_active.apply(
    lambda r: _ov.get("cluster", {}).get(str(r['market_id']), r['cluster']), axis=1)

# In Zonen aufteilen (Zone 3 / Audit-Log bleibt auf CSV-Zone, siehe U3)
df_zone3 = df_active[df_active['zone'] == 3].copy()

# Radar-Gruppierung auf wirksamer Zone (Overrides heben Wetten in Gruppe 1, siehe U3)
df_zone1_wirksam = df_active[df_active['zone_wirksam'] == 1].copy()
df_zone2_wirksam = df_active[df_active['zone_wirksam'] == 2].copy()

# --- MOMENTUM BERECHNUNG ---
# FIX 5 (Protokoll §10.5 / §3): FESTES Delta-5-Fenster statt "juengster Snapshot".
# Ziel: Quote(t) - Quote(t-5 Kalendertage). Fehlt t-5, nimm den zeitlich
# naechstgelegenen Snapshot im Fenster [t-7, t-3] (Fallback mit Vermerk).
# Vorher verglich der Code gegen den juengsten Nicht-Heute-Snapshot -> das Fenster
# schwankte unbemerkt zwischen 1 und 7+ Tagen, die Schwelle "+10 Pp/5T" war undefiniert.
def calculate_momentum(current_df, target_days=5, window=(3, 7)):
    archive_dir = BASE_DIR / "data" / "archive"
    if not archive_dir.exists():
        return {}, None

    today = datetime.now().date()
    candidates = []  # (abstand_zum_ziel, tage_zurueck, pfad)
    for f in archive_dir.glob("history_*.csv"):
        try:
            d = datetime.strptime(f.name.replace("history_", "").replace(".csv", ""), "%Y%m%d").date()
        except ValueError:
            continue
        days_back = (today - d).days
        if window[0] <= days_back <= window[1]:
            candidates.append((abs(days_back - target_days), days_back, f))

    if not candidates:
        return {}, None  # Kein Snapshot im 3-7T-Fenster -> kein Momentum (Log!)

    _, span_days, ref_file = sorted(candidates)[0]
    df_history = pd.read_csv(ref_file, dtype={'market_id': str})
    df_history['market_id'] = df_history['market_id'].str.replace('.0', '', regex=False)

    deltas = {}
    for _, current_row in current_df.iterrows():
        m_id = str(current_row['market_id']).replace('.0', '')
        hist_row = df_history[df_history['market_id'] == m_id]
        if not hist_row.empty:
            old_price = float(hist_row.iloc[0]['price_yes'])
            new_price = float(current_row['price_yes'])
            deltas[m_id] = (new_price - old_price) * 100

    return deltas, span_days

market_deltas, momentum_span = calculate_momentum(df_active)

# =============================================================================
# TRIGGER-UNIVERSUM (geteilte Berechnung) — wendet die §4/§6-Filter live an.
# Speist Sidebar (Schock-Index), Relevanz-Schaubild UND die Tabelle in Tab 1
# mit derselben Menge (U1 aus dem Restrukturierungs-Auftrag).
# =============================================================================
_cls_datei = BASE_DIR / "data" / "polymarket_classified.csv"
_hist_datei = BASE_DIR / "data" / "polymarket_history.csv"
_trigger_verfuegbar = _cls_datei.exists() and _hist_datei.exists()

if _trigger_verfuegbar:
    _c = pd.read_csv(_cls_datei, dtype={'market_id': str})
    _h = pd.read_csv(_hist_datei, dtype={'market_id': str})
    _u = _h.merge(_c[['market_id','cluster','zone','impact_score','risk_direction']],
                  on='market_id', how='inner')
    # Overrides überblenden (dieselbe Logik wie der Cloud-Logger)
    _ov_keys = (set(_ov.get("zone", {}).keys()) | set(_ov.get("cluster", {}).keys())
                | set(_ov.get("direction", {}).keys()))
    _u['Override'] = _u['market_id'].astype(str).isin(_ov_keys)
    _u['cluster'] = _u.apply(lambda r: _ov.get("cluster", {}).get(str(r['market_id']), r['cluster']), axis=1)
    _u['zone'] = _u.apply(lambda r: int(_ov.get("zone", {}).get(str(r['market_id']), r['zone'])), axis=1)
    _u['risk_direction'] = _u.apply(lambda r: _ov.get("direction", {}).get(str(r['market_id']), r['risk_direction']), axis=1)

    _u = _u[(_u['zone'] == 1) & (_u['impact_score'] >= 60)]
    if 'end_date' in _u.columns:
        _rest = (pd.to_datetime(_u['end_date'], errors='coerce', utc=True)
                 - pd.Timestamp.now(tz='UTC')).dt.days
        _u['Klasse'] = pd.cut(_rest, [-1, 92, 182, 99999],
                              labels=['KURZ (≤92T)', 'GRAU', 'LANG (≥183T)'])
        _u['Restlaufzeit'] = _rest
    else:
        _u['Klasse'] = 'GRAU (kein end_date)'
        _u['Restlaufzeit'] = None
    _schwelle = _u['Klasse'].astype(str).str.startswith('KURZ').map({True: 1e6, False: 2e6})
    _u['Triggerfähig'] = _u['volume'] >= _schwelle

    def _mom_label(row):
        d = market_deltas.get(str(row['market_id']))
        if d is None: return "—"
        richtung = _ov.get("direction", {}).get(str(row['market_id']), row['risk_direction'])
        g = -d if richtung == 'RISK_ON' else d
        quote_txt = f"Quote {d:+.1f} Pp"
        if g >= 10: return f"{quote_txt} → 🔥 Gefahr +{g:.1f}"
        if g >= 5:  return f"{quote_txt} → 📈 Gefahr +{g:.1f}"
        if g <= -5: return f"{quote_txt} → 🟢 Gefahr {g:.1f}"
        return f"{quote_txt} → ➡️ neutral"
    _u['Momentum(5T)'] = _u.apply(_mom_label, axis=1)

    _zeig = _u[_u['Triggerfähig']].sort_values(['cluster','volume'], ascending=[True,False])
else:
    _u = pd.DataFrame()
    _zeig = pd.DataFrame()

# Nachrücker: Zone-1-Kandidaten (wirksame Zone), die an Impact-Gate oder
# Volumen-Schwelle scheitern -> Beobachtungsschicht im Radar (U3).
if _trigger_verfuegbar:
    _triggerfaehige_ids = set(_zeig['market_id'].astype(str))
else:
    _triggerfaehige_ids = set()
df_zone1_runners_up = df_zone1_wirksam[~df_zone1_wirksam['market_id'].astype(str).isin(_triggerfaehige_ids)]

# NEU: Alle Märkte für Zone 2 (wirksam) übernehmen (Kein Top 5 Limit mehr!)
df_radar = pd.concat([df_zone1_runners_up, df_zone2_wirksam]).sort_values(
    by=['zone_wirksam', 'volume'], ascending=[True, False])

# --- ZUSTANDSDATEN AUS dem CLOUD-LOGGER (state.json) - einmal frueh geladen,
# gemeinsame Quelle fuer den Zustands-Kopf (U2) und die Portfoliobalken ---
_state_path = BASE_DIR / "data" / "logs" / "state.json"
if _state_path.exists():
    import json as _json
    _st = _json.loads(_state_path.read_text())
else:
    _st = None

# --- DATA FRESHNESS BERECHNUNG ---
def get_freshness_indicator(dataframe):
    date_col = next((col for col in dataframe.columns if col.lower() in ['date', 'timestamp', 'last_updated']), None)
    if date_col:
        latest_date = pd.to_datetime(dataframe[date_col]).max()
    else:
        history_file = BASE_DIR / "data" / "polymarket_history.csv"
        if history_file.exists():
            latest_date = datetime.fromtimestamp(history_file.stat().st_mtime)
        else:
            latest_date = datetime.now()

    days_old = (datetime.now().date() - latest_date.date()).days
    date_str = latest_date.strftime('%d.%m.%Y')
    
    if days_old <= 1:
        return "🟢 Aktuell", f"Stand: {date_str}"
    elif days_old <= 3:
        return "🟡 Verzögert", f"{days_old} Tage alt ({date_str})"
    else:
        return "🔴 Veraltet", f"Pipeline prüfen (>3 Tage alt)"

# --- UI LAYOUT ---
col_title, col_metric = st.columns([4, 1])
with col_title:
    st.title("🌐 Risk Manager: Portfolio Command Center")
with col_metric:
    status_icon, status_text = get_freshness_indicator(df)
    st.metric(label="Daten-Status", value=status_icon, delta=status_text, delta_color="off")

st.markdown("---")

if FREEZE_MODE:
    st.info(f"🧊 **Eingefrorener Stand vom {FREEZE_STAND}** — Abgabe-Version zur Projektarbeit. "
            "Diese Ansicht liest einen fixierten Datenstand; das Live-System läuft unabhängig in der Cloud weiter. "
            "Regelwerk: Overlay-Protokoll v1.0 mit Nachträgen 1–3.")

tab_monitor, tab_radar, tab_audit, tab_search, tab_control = st.tabs([
    "🚨 Kill-Switch & Live-Betrieb",
    "📡 Information Dashboard",
    "🗑️ Audit-Log",
    "🔍 Datenbank-Suche",
    "🔧 Steuerung"
])

# --- TAB 1: KILL-SWITCH & LIVE-BETRIEB ---
with tab_monitor:
    # (a) Dynamischer Zustands-Kopf (U2/K1) - gleiche Quelle wie die Portfoliobalken (state.json)
    if _st is not None:
        _cluster_info = _st.get("cluster", {})
        _gebremst = [f"{c} {s.get('zustand', '?')}" for c, s in _cluster_info.items()
                     if isinstance(s, dict) and s.get("zustand") != "VOLLGAS"]
        _kopf_status = " · ".join(_gebremst) if _gebremst else "VOLLGAS (alle Cluster)"
        st.header(f"🚨 Kill-Switch & Live-Betrieb — {_kopf_status}")
        _wirk_kopf = _st.get("wirksam", {"SPY": 1.0})
        if isinstance(_wirk_kopf, dict):
            st.caption("Wirksames Portfolio (T+1): " + " · ".join(
                f"{int(_w*100)}% {_a}" for _a, _w in sorted(_wirk_kopf.items(), key=lambda x: -x[1])))
    else:
        st.header("🚨 Kill-Switch & Live-Betrieb")
        st.info("Zustandsdaten noch nicht gespiegelt (aws s3 sync).")

    st.markdown("*Ereignisse in diesem Bereich korrelieren global. Die Bremsen schalten ausschließlich über die §4-Cluster-Regeln (siehe Live-Evaluation unten) — nicht über den Diagnose-Index rechts.*")
    
    col_main, col_sidebar = st.columns([2.5, 1])

    with col_main:
        # (b) Relevanz-Schaubild - Datenbasis: Trigger-Universum statt Top-10 (U1)
        if _zeig.empty:
            st.success("Aktuell keine akuten Schock-Szenarien im Trigger-Universum. Das Portfolio kann frei atmen.")
        else:
            plot_data = []
            for _, row in _zeig.iterrows():
                market_id = str(row['market_id'])

                plot_data.append({
                    "market_id": market_id,
                    "Wette": row['title'],
                    "Cluster": row['cluster'],
                    "Wahrscheinlichkeit (%)": float(row['price_yes']) * 100,
                    "Relevanz (Impact)": int(_ov.get("impact", {}).get(market_id, row.get('impact_score', 50))),
                    "Volumen (USD)": float(row['volume'])
                })
            
            df_plot = pd.DataFrame(plot_data)
            _richtung_map = {
                str(r['market_id']): ("🟢 Risk-On" if _ov.get("direction", {}).get(str(r['market_id']), r['risk_direction']) == "RISK_ON" else "🔴 Risk-Off")
                for _, r in _zeig.iterrows()
            }
            df_plot["Regime"] = df_plot["market_id"].map(_richtung_map).fillna("🔴 Risk-Off")

            if not df_plot.empty:
                # EINE Momentum-Quelle: market_deltas (Δ5-Rohwerte) + Wirkrichtung mit
                # Override-Vorrang - identische Logik wie _mom_label im Trigger-Universum-Block.
                _dirs = _u.assign(_k=_u["market_id"].astype(str)).set_index("_k")["risk_direction"]
                def _mom_text(mid):
                    mid = str(mid)
                    d = market_deltas.get(mid)
                    if d is None: return "—"
                    g = -d if _dirs.get(mid, "RISK_OFF") == "RISK_ON" else d
                    icon = "🔥" if g >= 10 else "📈" if g >= 5 else "🟢" if g <= -5 else "➡️"
                    return f"Quote {d:+.1f} Pp → {icon} Gefahr {g:+.1f}"
                def _mom_num(mid):
                    mid = str(mid)
                    d = market_deltas.get(mid)
                    if d is None: return 0.0
                    return -d if _dirs.get(mid, "RISK_OFF") == "RISK_ON" else d
                df_plot["Momentum"] = df_plot["market_id"].astype(str).map(_mom_text)
                df_plot["Momentum_num"] = df_plot["market_id"].astype(str).map(_mom_num)
                fig = px.scatter(
                    df_plot, x="Wahrscheinlichkeit (%)", y="Relevanz (Impact)", 
                    size="Volumen (USD)", color="Regime", 
                    color_discrete_map={"🔴 Risk-Off": "rgba(255, 65, 54, 0.8)", "🟢 Risk-On": "rgba(46, 204, 64, 0.8)"},
                    hover_name="Wette",
                    hover_data={"Momentum": True, "Cluster": True},
                    size_max=55, range_x=[-5, 105], range_y=[0, 105], height=500 
                )
                fig.update_traces(hovertemplate="<b>%{hovertext}</b><br><br>Cluster: %{customdata[1]}<br>Wahrscheinlichkeit: %{x:.1f}%<br>Impact Score: %{y}<br>Momentum: %{customdata[0]}")
                
                for trace in fig.data:
                    regime_name = trace.name
                    df_sub = df_plot[df_plot["Regime"] == regime_name]
                    trace.marker.line.width = [4 if abs(m) > 5.0 else 0 for m in df_sub["Momentum_num"]]
                    trace.marker.line.color = ["#FFD700" if abs(m) > 5.0 else "rgba(0,0,0,0)" for m in df_sub["Momentum_num"]]
                
                fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3)
                fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.3)
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), legend_title_text="Markt-Regime", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Markt-Wahrscheinlichkeit (%)", gridcolor="rgba(128,128,128,0.2)"), yaxis=dict(title="Globaler Impact Score (1-100)", gridcolor="rgba(128,128,128,0.2)"))
                st.plotly_chart(fig, width='stretch', key="plot_top10")

    # --- DIE SIDEBAR FÜR DIE RISK ENGINE ---
    with col_sidebar:
        st.header("📊 Schock-Index (Diagnose)")
        st.caption("Reines Diagnose-Thermometer über das Trigger-Universum (Zone 1, Impact ≥ 60, über Volumen-Schwelle). "
                   "Steuert NICHTS — die Overlay-Trigger laufen ausschließlich über "
                   "das §4-Regelwerk (siehe Live-Evaluation unten).")
        
        # FIX 1: I_0/k auf die MITTELWERT-Skala des Index kalibriert (I_t ~0.5-0.9 normal).
        # Der alte Wert I_0=10.0 lag auf Summen-Skala und war unerreichbar (Kill-Switch tot).
        # I_t ist DIAGNOSE - die harten Trigger laufen ueber die Paragraph-4-Regeln des Protokolls.
        rm = RiskManager(lambda_min=2.0, lambda_max=25.0, I_0=1.0, k=4.0)
        if momentum_span is not None and momentum_span != 5:
            st.caption(f"Momentum-Fenster: {momentum_span} Tage (Fallback, Soll 5) - Log-Vermerk")
        elif momentum_span is None:
            st.caption("Kein Snapshot im 3-7-Tage-Fenster - Momentum heute nicht berechenbar!")
        
        # Impact + Regime aus S3-Wahrheit (CSV + overrides.json) statt lokaler Session-Regler
        _ov_s3 = _ov if '_ov' in globals() else {"impact": {}, "direction": {}}
        _imp_s3 = {str(r['market_id']): int(_ov_s3.get("impact", {}).get(str(r['market_id']), r['impact_score']))
                   for _, r in _zeig.iterrows()}
        _reg_s3 = {str(r['market_id']):
                   ("🟢 Risk-On" if _ov_s3.get("direction", {}).get(str(r['market_id']), r['risk_direction']) == "RISK_ON"
                    else "🔴 Risk-Off")
                   for _, r in _zeig.iterrows()}
        current_shock_index = rm.calculate_shock_index(_zeig, _imp_s3, _reg_s3, market_deltas)
        
        current_lambda = rm.get_lambda(current_shock_index)
        
        st.metric("Aggregierter Schock-Index (I_t)", f"{current_shock_index:.2f}")
        
        if current_shock_index < 0.8: st.success(f"Lage: ruhig (I_t {current_shock_index:.2f})")
        elif current_shock_index < 1.1: st.warning(f"Lage: erhöht (I_t {current_shock_index:.2f})")
        else: st.error(f"Lage: hoch (I_t {current_shock_index:.2f})")

        _idx_niveau = rm.calculate_shock_index(
            _zeig, _imp_s3,
            _reg_s3, {})          # ohne Deltas = reine Niveaus
        _idx_mom = current_shock_index - _idx_niveau
        st.caption(f"Zusammensetzung: {_idx_niveau:.2f} stehende Niveaus "
                   f"({'+' if _idx_mom>=0 else ''}{_idx_mom:.2f} frisches Momentum). "
                   f"Nur Letzteres kann Bremsen auslösen — via §4-Cluster-Regeln, nicht via Index.")

        with st.expander("ℹ️ Wie lese ich diesen Index?"):
            st.markdown("""
            **Rechenweg (pro Wette, dann Mittelwert über das Trigger-Universum):**

            `Beitrag = (Gefahren-Niveau + Momentum-Zuschlag) × Impact-Gewicht`

            - **Gefahren-Niveau** (0–1): die eingepreiste Wahrscheinlichkeit des *schlechten* Ausgangs.
            Bei Risiko-Wetten die Quote selbst, bei Normalisierungs-Wetten (🟢) die Gegenquote —
            „Hormuz normalisiert sich: 46 %" fließt also als 54 % Gefahr ein.
            - **Momentum-Zuschlag**: starke Bewegung *in Gefahrenrichtung* (>±5 Pp/5T) erhöht bzw.
            senkt den Beitrag leicht (±0,1–0,2 typisch). Frische Eskalation zählt mehr als alte Angst.
            - **Impact-Gewicht** (1–5): Zerstörungskraft laut Klassifikation (Score/20).
            Ein 100er-Weltkriegs-Kontrakt wiegt fünfmal so viel wie ein 20er-Randthema.

            **Einordnung der Skala** (empirisch, Stand Kalibrierung Aug 2026):

            | I_t | Lage | Lesart |
            |---|---|---|
            | < 0,8 | ruhig | Zone 1 preist überwiegend Restrisiken (< 20 %) ein |
            | 0,8 – 1,1 | erhöht | mehrere Kontrakte mit substanzieller Gefahr **oder** frisches Eskalations-Momentum |
            | > 1,1 | hoch | breite und/oder akut steigende Gefahren-Einpreisung |

            **Wichtig:** Der Index ist ein *Stimmungs-Thermometer* der Ereignis-Märkte — er verdichtet
            viel in eine Zahl und verliert dabei genau das, was das Overlay nutzt (welcher Cluster?
            welche Laufzeit? frisch oder alt?). Deshalb: Diagnose ja, Trigger nein. Die Bremsen
            schalten ausschließlich die §4-Regeln (Cluster-Momentum ≥ +10 Pp/5T etc.).
            """)

    # (c) Trigger-Universum: Transparenz-Ansicht — wendet die §4/§6-Filter live an
    # (Berechnung von _u/_zeig weiter oben, geteilt mit Sidebar & Schaubild)
    st.markdown("---")
    st.header("🎯 Trigger-Universum: Wer darf Bremsen auslösen?")
    st.caption("Live-Anwendung der eingefrorenen Regeln (Protokoll v1.0): Zone 1 · "
               r"Impact ≥ 60 · Volumen ≥ \$2M (Langläufer ≥183T) bzw. ≥ \$1M (Kurzläufer ≤92T) · "
               "Grauzone (93–182T): Momentum zählt im Cluster-Max, keine S2-Fähigkeit. "
               "Identische Logik wie der Headless-Logger (22:15 UTC).")

    if _trigger_verfuegbar:
        st.write(f"**{len(_zeig)} triggerfähige Kontrakte** "
                 f"(von {len(_u)} Zone-1-Kandidaten; Rest scheitert an Volumen-Schwelle):")
        st.caption("**Lesehilfe Momentum:** links die rohe Quotenbewegung (5 Tage), rechts die Bedeutung "
                   "auf der Gefahren-Skala, mit der die Trigger rechnen. Bei RISK_ON-Wetten "
                   "(Normalisierung/Beruhigung) sind beide gegenläufig: fallende Quote = gute Nachricht "
                   "wird unwahrscheinlicher = Gefahr steigt. 🔥 ≥ +10 wäre S1-relevant.")
        st.dataframe(_zeig[['cluster','title','price_yes', 'Momentum(5T)', 'volume','impact_score',
                            'risk_direction','Override','Klasse','Restlaufzeit']],
                     width='stretch', hide_index=True)
    else:
        st.info("Klassifikations-/History-Daten noch nicht gespiegelt (aws s3 sync).")

    # (d) Live-Evaluation (Overlay-Protokoll v1.0, Paragraph 8) — reine Leseansicht
    # Quelle: data/logs/*.csv (S3-Spiegel). Aendert NICHTS an Messung oder Regeln.
    st.markdown("---")
    st.header("📈 Live-Evaluation: die fünf Zeitreihen")

    _eq_path = BASE_DIR / "data" / "logs" / "equity_log.csv"
    _sig_path = BASE_DIR / "data" / "logs" / "signal_log.csv"

    if not _eq_path.exists():
        st.info("Noch keine Log-Daten gespiegelt (aws s3 sync ausführen; Logs entstehen ab Go-Live täglich 22:15 UTC).")
    else:
        _eq = pd.read_csv(_eq_path, parse_dates=["date"])
        _tage = len(_eq)
        st.warning(
            f"⚠️ **Zwischenstand nach {_tage} Handelstagen — noch NICHT aussagekräftig.** "
            "Präregistrierte Auswertung: deskriptiver Zwischenbericht nach 6 Monaten, "
            "Endauswertung nach 12 Monaten (Protokoll §9). Diese Ansicht ist eine "
            "Leseansicht der append-only-Logs und hat keinerlei Rückwirkung auf die Messung. "
            "Kurzfrist-Differenzen zwischen den Linien sind überwiegend Rauschen."
        )
        if "overlay_open_schatten" not in _eq.columns:
            _eq["overlay_open_schatten"] = float("nan")   # Log-Stand vor Amendment 1
        _eq["overlay_open_schatten"] = pd.to_numeric(_eq["overlay_open_schatten"], errors="coerce")
        _plot = _eq.set_index("date")[
            ["spy_hold", "overlay_open_schatten", "vehikel_overlay", "vix_schatten", "semi_schatten"]
        ].rename(columns={
            "spy_hold": "1 · SPY (Null-Linie)",
            "overlay_open_schatten": "2 · Vehikel + Overlay (Experiment, T+1-Open)",
            "vehikel_overlay": "2b · Overlay T+1-Close (Schatten: Latenz-Vergleich)",
            "vix_schatten": "4 · VIX-Bremse (Schatten)",
            "semi_schatten": "5 · Semi-Vola-Ampel (Schatten)"})
        import plotly.express as _px
        _fig_eq = _px.line(_plot.reset_index(), x="date", y=list(_plot.columns))
        _fig_eq.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
            xaxis_title=None, yaxis_title="indexiert (Start = 1,0)",
            height=440, margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified")
        st.plotly_chart(_fig_eq, use_container_width=True)
        st.caption("Legende: Klick = Linie ein-/ausblenden · Doppelklick = Linie isolieren.")
        st.caption("Zeitreihe 3 (V3-Champion) läuft extern und wird im Zwischenbericht ergänzt. "
                   "Konvention per Amendment 3: Experiment (Linie 2) mit T+1-Open-Umsetzung — "
                   "konsistent mit der Ausführungspraxis des Execution-Tracks; T+1-Close-Variante "
                   "(2b) läuft als Schatten für den latenz-gleichen Vergleich (Zeitreihe 4 setzt "
                   "T+1-Close um). Werte 17.–18.08. deterministisch rückgerechnet (20.08.). "
                   "Erfolgskriterien §9 datumsbasiert unverändert; Fehlalarm-Budget §7 auf Linie 2.")

        c1, c2, c3 = st.columns(3)
        _ovl_wert = (_eq["overlay_open_schatten"].iloc[-1]
                     if _eq["overlay_open_schatten"].notna().any()
                     else _eq["vehikel_overlay"].iloc[-1])
        _sp = _eq["spy_hold"].iloc[-1]
        c1.metric("Overlay vs. Null-Linie", f"{(_ovl_wert/_sp-1)*100:+.2f} Pp",
                  help="Fehlalarm-Budget: max. -4 Pp p.a. ohne Treffer (§7)")
        c2.metric("Overlay vs. VIX-Bremse (Close-Konv.)", f"{(_ovl_wert/_eq['vix_schatten'].iloc[-1]-1)*100:+.2f} Pp",
                  help="Der eigentliche Gegner: schlägt Ereignis-Info den simplen VIX-Schwellwert?")
        if _sig_path.exists():
            _sig = pd.read_csv(_sig_path)
            _letzte = _sig[_sig["date"] == _sig["date"].max()]
            _zust = ", ".join(f"{r['cluster']}: {r['zustand']}" for _, r in _letzte.iterrows())
            c3.metric("Bremszustand aktuell", _letzte.iloc[0]["date"] if len(_letzte) else "—",
                      help=_zust or "keine Signale")
            st.caption(f"Cluster-Zustände (letzter Lauf): {_zust}")

        # Aktuelle Portfoliozusammensetzung (aus dem Logger-Zustand) - _st wurde
        # weiter oben einmalig geladen (gemeinsame Quelle mit dem Zustands-Kopf, U2)
        if _st is not None:
            _wirk = _st.get("wirksam", {"SPY": 1.0})
            _pend = _st.get("pending", _wirk)
            if isinstance(_wirk, str): _wirk = {"SPY": 1.0}   # Erstlauf-Platzhalter
            if isinstance(_pend, str): _pend = {"SPY": 1.0}
            st.subheader("🧺 Portfoliozusammensetzung (Vehikel + Overlay)")
            _c1, _c2 = st.columns(2)
            with _c1:
                st.caption("**Heute wirksam** (T+1-Buchhaltung)")
                for _a, _w in sorted(_wirk.items(), key=lambda x: -x[1]):
                    st.progress(_w, text=f"{_a}: {_w:.0%}")
            with _c2:
                st.caption("**Ab nächstem Handelsschluss** (gestriger Beschluss)")
                for _a, _w in sorted(_pend.items(), key=lambda x: -x[1]):
                    st.progress(_w, text=f"{_a}: {_w:.0%}")
            if _wirk != _pend:
                st.warning("Umschichtung unterwegs — Order-Vorlagen-Mail beachten (Execution-Track).")

# --- TAB 2: DAS RADAR (Read-Only & Alle Märkte) ---
with tab_radar:
    st.subheader("📡 INFORMATION DASHBOARD (Sectoral Rotation / Radar)")
    st.markdown("*Thematische Beobachtung. Diese Events beeinflussen Teilsektoren oder die langfristige Makro-Story, lösen aber keinen globalen Crash-Schutz aus.*")
    
    if not df_radar.empty:
        # NEU: Sicherstellen, dass absolut alles als String behandelt wird vor dem Sortieren
        # Gruppierung auf wirksamem Cluster (Override-überblendet, analog Zone/Richtung)
        clusters = [str(c) for c in df_radar['cluster_wirksam'].unique()]
        for cluster in sorted(clusters):
            st.markdown(f"### 🗂️ {cluster}")
            df_c = df_radar[df_radar['cluster_wirksam'] == cluster].copy()

            # Gruppierung/Anzeige auf wirksamer Zone (Override-überblendet, U3):
            # weicht die wirksame Zone vom LLM-Wert ab, wird das gekennzeichnet.
            def format_zone(row):
                z = row['zone_wirksam']
                if z == 1: label = "🔴 1 (Beobachtung – unter Schwelle)"
                elif z == 2: label = "🟡 2 (Radar)"
                else: label = str(z)
                if row['zone_wirksam'] != row['zone']:
                    label += f" (Override, LLM: {row['zone']})"
                return label

            def format_richtung(row):
                label = "🟢 Risk-On" if row['risk_direction_wirksam'] == "RISK_ON" else "🔴 Risk-Off"
                if row['risk_direction_wirksam'] != row['risk_direction']:
                    label += " (Override)"
                return label

            def format_cluster(row):
                if row['cluster_wirksam'] != row['cluster']:
                    return f"⚠️ Override (LLM: {row['cluster']})"
                return ""

            df_c['Status'] = df_c.apply(format_zone, axis=1)
            df_c['Richtung'] = df_c.apply(format_richtung, axis=1)
            df_c['Cluster-Hinweis'] = df_c.apply(format_cluster, axis=1)
            df_c['Quote'] = (df_c['price_yes'] * 100).round(1).astype(str) + ' %'
            df_c['Volumen'] = '$' + (df_c['volume'] / 1_000_000).round(2).astype(str) + 'M'
            
            df_table = df_c[['Status', 'Richtung', 'Cluster-Hinweis', 'title', 'Quote', 'Volumen', 'impact_score', 'reasoning']]
            df_table.rename(columns={'title': 'Wette', 'impact_score': 'LLM Impact', 'reasoning': 'LLM Begründung'}, inplace=True)
            
            st.dataframe(df_table, width='stretch', hide_index=True)
    else:
        st.info("Das Radar ist aktuell leer.")

    # Neuzugänge (seit letztem Review) — wöchentliche Klassifikations-Kontrolle
    st.markdown("---")
    st.header("🆕 Neuzugänge im Wetten-Universum")
    REVIEW_STICHTAG = "2026-08-18"   # nach jedem Review hier hochsetzen
    _cls_neu = BASE_DIR / "data" / "polymarket_classified.csv"
    if _cls_neu.exists():
        _n = pd.read_csv(_cls_neu, dtype={'market_id': str})
        _n['erfasst'] = _n['timestamp'].str[:10]
        _neu = _n[_n['erfasst'] >= REVIEW_STICHTAG].sort_values('volume', ascending=False)
        if len(_neu):
            st.caption(f"{len(_neu)} Wetten seit {REVIEW_STICHTAG} erstklassifiziert — "
                       "Prüffragen: Zone plausibel? Wirkrichtung korrekt (Beruhigungs-Kontrakte!)? "
                       "Kurioses mit hohem Impact? → Korrekturen via override_tool (S3).")
            st.dataframe(_neu[['erfasst','zone','cluster','title','price_yes','volume',
                               'impact_score','risk_direction','is_normalization']],
                         width='stretch', hide_index=True)
        else:
            st.success(f"Keine Neuzugänge seit {REVIEW_STICHTAG}.")

# --- TAB 3: DAS AUDIT-LOG (ZONE 3) & PAPIERKORB ---
with tab_audit:
    st.subheader("🗑️ Zone 3 (Rauschen)")
    st.write("Diese Märkte wurden vom LLM als irrelevant für Faktor-ETFs eingestuft.")
    
    df_display_audit = df_zone3[['cluster', 'title', 'price_yes', 'volume', 'reasoning']].copy()
    if not df_display_audit.empty:
        df_display_audit['price_yes'] = (df_display_audit['price_yes'] * 100).round(1).astype(str) + ' %'
        df_display_audit['volume'] = '$' + (df_display_audit['volume'] / 1_000_000).round(2).astype(str) + 'M'
        df_display_audit.rename(columns={'cluster': 'Aussortiert als', 'title': 'Wette', 'price_yes': 'Quote', 'volume': 'Volumen', 'reasoning': 'LLM Begründung'}, inplace=True)
        st.dataframe(df_display_audit, width='stretch', hide_index=True)
    else:
        st.write("Keine Märkte in Zone 3.")
        
    st.markdown("---")
    st.info("Wirksame Eingriffe & Änderungslog: siehe Tab 🔧 Steuerung.")

# --- TAB 4: DATENBANK-SUCHE ---
with tab_search:
    st.subheader("🔍 Globale Datenbank-Suche")
    search_query = st.text_input("Suche nach Stichworten (z. B. 'Taiwan', 'Fed', 'SpaceX', 'Trump')...", "")
    
    if search_query:
        mask = df_active['title'].str.contains(search_query, case=False, na=False) | df_active['reasoning'].str.contains(search_query, case=False, na=False)
        search_results = df_active[mask].copy()
        
        if not search_results.empty:
            search_results['price_yes'] = (search_results['price_yes'] * 100).round(1).astype(str) + ' %'
            search_results['volume'] = '$' + (search_results['volume'] / 1_000_000).round(2).astype(str) + 'M'
            search_results['zone'] = search_results['zone'].map({1: "🔴 1", 2: "🟡 2", 3: "🗑️ 3"})
            search_results.rename(columns={'zone': 'Zone', 'cluster': 'Cluster', 'title': 'Wette', 'price_yes': 'Quote', 'volume': 'Volumen', 'reasoning': 'LLM Begründung'}, inplace=True)
            st.dataframe(search_results[['Zone', 'Cluster', 'Wette', 'Quote', 'Volumen', 'LLM Begründung']], width='stretch', hide_index=True)
        else:
            st.warning(f"Keine Treffer für '{search_query}'.")

# --- TAB 5: STEUERUNG (Override-Editor) ---
with tab_control:
    st.header("🔧 Override-Editor (wirksame Eingriffe)")

    if FREEZE_MODE:
        st.warning("🔒 Der Override-Editor ist im eingefrorenen Stand deaktiviert — diese öffentliche "
                   "Ansicht ist eine reine Leseversion. Alle im Betrieb erfolgten Eingriffe sind unten "
                   "im Änderungsprotokoll dokumentiert.")

        with st.expander(f"📜 Änderungslog ({len(_ov.get('log', []))} Einträge) & aktive Overrides"):
            if _ov.get("log"):
                st.dataframe(pd.DataFrame(_ov["log"]), width='stretch', hide_index=True)
            st.caption(f"Aktive Overrides: {len(_ov.get('zone',{}))} Zone · "
                    f"{len(_ov.get('direction',{}))} Wirkrichtung · "
                    f"{len(_ov.get('cluster',{}))} Cluster · "
                    f"{len(_ov.get('impact',{}))} Impact · {len(_ov.get('mute',[]))} stumm")
            with st.expander("Rohdaten (overrides.json)"):
                st.json(_ov)

    else:
        st.caption("Schreibt in s3://…/overrides.json — die Datei, die der Cloud-Logger liest. "
                   "Jeder Eintrag braucht eine Begründung und wirkt nur nach vorn (§8). "
                   "Wirksam ab dem nächsten 22:15-UTC-Lauf.")

        with st.expander(f"📜 Änderungslog ({len(_ov.get('log', []))} Einträge) & aktive Overrides"):
            if _ov.get("log"):
                st.dataframe(pd.DataFrame(_ov["log"]), use_container_width=True, hide_index=True)
            st.caption(f"Aktive Overrides: {len(_ov.get('zone',{}))} Zone · "
                    f"{len(_ov.get('direction',{}))} Wirkrichtung · "
                    f"{len(_ov.get('cluster',{}))} Cluster · "
                    f"{len(_ov.get('impact',{}))} Impact · {len(_ov.get('mute',[]))} stumm")
            with st.expander("Rohdaten (overrides.json)"):
                st.json(_ov)

        _cls_all = pd.read_csv(BASE_DIR / "data" / "polymarket_classified.csv", dtype={'market_id': str})
        _auswahl = st.selectbox("Wette wählen", _cls_all['market_id'] + " — " + _cls_all['title'].str[:70])
        _mid = _auswahl.split(" — ")[0]

        # Ist-Stand der gewählten Wette anzeigen (LLM-Klassifikation + ggf. aktiver Override)
        _akt = _cls_all[_cls_all['market_id'] == _mid].iloc[0]
        _ov_z = _ov.get("zone", {}).get(_mid)
        _ov_d = _ov.get("direction", {}).get(_mid)
        _ov_i = _ov.get("impact", {}).get(_mid)
        _ov_c = _ov.get("cluster", {}).get(_mid)
        st.info(
            f"**Aktuell wirksam:** Zone **{_ov_z if _ov_z is not None else _akt['zone']}"
            f"{' (Override)' if _ov_z is not None else ''}** · "
            f"Richtung **{_ov_d or _akt['risk_direction']}"
            f"{' (Override)' if _ov_d else ''}** · "
            f"Impact **{_ov_i if _ov_i is not None else _akt['impact_score']}"
            f"{' (Override)' if _ov_i is not None else ''}** · "
            f"Cluster **{_ov_c or _akt['cluster']}"
            f"{' (Override)' if _ov_c else ''}** · "
            f"Quote {float(_akt['price_yes'])*100:.1f}% · Vol ${float(_akt['volume'])/1e6:.1f}M"
            + (" · 🔇 stumm" if _mid in _ov.get("mute", []) else "")
        )
        with st.expander("LLM-Begründung der Klassifikation"):
            st.write(_akt.get('reasoning', '—'))

        _CLUSTER_OPTIONEN = ["(unverändert)", "Geopolitischer & Militärischer Schock",
                             "Systemische Stabilitätsrisiken", "Makroökonomie & Zentralbanken",
                             "Politische Struktur & Wahlen"]
        _CLUSTER_KUERZEL = {"Geopolitischer & Militärischer Schock": "GEO",
                            "Systemische Stabilitätsrisiken": "SYS",
                            "Makroökonomie & Zentralbanken": "MAKRO",
                            "Politische Struktur & Wahlen": "POL"}

        _c1, _c2, _c3, _c4 = st.columns(4)
        _zone_neu = _c1.selectbox("Zone", ["(unverändert)", "1", "2", "3"])
        _dir_neu = _c2.selectbox("Wirkrichtung", ["(unverändert)", "RISK_ON", "RISK_OFF"])
        _imp_neu = _c3.text_input("Impact (leer = unverändert)", "")
        _cluster_neu = _c4.selectbox("Cluster", _CLUSTER_OPTIONEN)
        _mute_neu = st.checkbox("Stummschalten (mute)")
        _grund = st.text_input("Begründung (PFLICHT — geht ins Änderungslog)")

        if st.button("✍️ Override nach S3 schreiben", type="primary"):
            if not _grund.strip():
                st.error("Ohne Begründung kein Eintrag — §8.")
            else:
                _aktion = []
                if _zone_neu != "(unverändert)":
                    _ov["zone"][_mid] = int(_zone_neu); _aktion.append(f"Zone→{_zone_neu}")
                if _dir_neu != "(unverändert)":
                    _ov["direction"][_mid] = _dir_neu; _aktion.append(f"Richtung→{_dir_neu}")
                if _imp_neu.strip():
                    _ov["impact"][_mid] = int(_imp_neu); _aktion.append(f"Impact→{_imp_neu}")
                if _cluster_neu != "(unverändert)":
                    _ov.setdefault("cluster", {})[_mid] = _cluster_neu
                    _aktion.append(f"Cluster→{_CLUSTER_KUERZEL.get(_cluster_neu, _cluster_neu)}")
                if _mute_neu and _mid not in _ov["mute"]:
                    _ov["mute"].append(_mid); _aktion.append("mute")
                if not _aktion:
                    st.warning("Keine Änderung ausgewählt.")
                else:
                    _ov.setdefault("log", []).append({
                        "datum": str(pd.Timestamp.now().date()), "market_id": _mid,
                        "aktion": ", ".join(_aktion), "grund": _grund.strip()})
                    _S3C.put_object(Bucket=_BUCKET, Key=_KEY,
                                    Body=_j.dumps(_ov, indent=1).encode())
                    st.success(f"✅ Geschrieben: {_mid} — {', '.join(_aktion)} — wirkt ab 22:15 UTC.")