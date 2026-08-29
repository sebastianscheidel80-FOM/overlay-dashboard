import numpy as np

class RiskManager:
    """
    v1.0-Fixes (Overlay-Protokoll §10.1 + §10.2, 11.08.2026):
    - FIX 1 (Index-Skala): I_t ist ein Mittelwert pro Wette (Division durch n).
      I_0 war auf Summen-Skala kalibriert (10.0) und damit unerreichbar -> Kill-Switch tot.
      Neu: I_0/k auf Mittelwert-Skala (I_0=1.0, k=4.0). I_t bleibt reine DIAGNOSE-Metrik;
      die harten Trigger laufen ausschliesslich ueber die Paragraph-4-Regeln des Protokolls.
    - FIX 2 (Vorzeichen): Momentum wirkt vorzeichenrichtig relativ zur Wirkrichtung.
      Risk-Off-Kontrakt: steigende Quote = Gefahr rauf. Risk-On-Kontrakt (Normalisierung):
      FALLENDE Quote = Gefahr rauf. Entspannungs-Momentum SENKT den Beitrag (Malus),
      statt ihn wie zuvor (abs) faelschlich zu erhoehen.
    """
    def __init__(self, lambda_min, lambda_max, I_0=1.0, k=4.0):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.I_0 = I_0  # Wendepunkt auf MITTELWERT-Skala (kalibriert: Normalbereich ~0.5-0.9)
        self.k = k      # Steilheit

    def calculate_shock_index(self, df, impact_scores, regimes, deltas):
        """Aggregierter Schock-Index (Mittelwert ueber die Top-Wetten). Diagnose, kein Trigger."""
        total_shock = 0.0
        n = len(df)
        if n == 0:
            return 0.0

        for _, row in df.iterrows():
            m_id = str(row['market_id'])
            p_yes = float(row['price_yes'])
            impact = impact_scores.get(m_id, 50) / 20.0  # Normierung auf 1-5
            regime = regimes.get(m_id, "🔴 Risk-Off")

            # 1. Gefahr bestimmen (Niveau, vorzeichenrichtig)
            if regime == "🟢 Risk-On":
                p_gefahr = 1.0 - p_yes
            else:
                p_gefahr = p_yes

            # 2. Momentum VORZEICHENRICHTIG (FIX 2)
            #    delta > 0 heisst: Quote gestiegen.
            #    Risk-Off: steigende Quote = Eskalation (+). Risk-On: steigende Quote = Entwarnung (-).
            momentum = deltas.get(m_id, 0.0)
            if regime == "🟢 Risk-On":
                signed_momentum = -momentum
            else:
                signed_momentum = momentum

            momentum_term = 0.0
            if signed_momentum > 5.0:        # Eskalations-Momentum -> Aufschlag
                momentum_term = (signed_momentum / 100.0) * 1.2
            elif signed_momentum < -5.0:     # Entwarnungs-Momentum -> Abschlag
                momentum_term = (signed_momentum / 100.0) * 1.2  # negativ

            beitrag = (p_gefahr + momentum_term) * impact
            total_shock += max(beitrag, 0.0)  # kein negativer Einzelbeitrag

        return total_shock / n

    def get_lambda(self, shock_index):
        """Risikoaversion (Lambda) via Sigmoid - Diagnoseanzeige."""
        exponent = -self.k * (shock_index - self.I_0)
        sigmoid = 1.0 / (1.0 + np.exp(exponent))
        return self.lambda_min + (self.lambda_max - self.lambda_min) * sigmoid
