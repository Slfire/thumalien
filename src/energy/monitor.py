"""Suivi de l'empreinte carbone avec CodeCarbon."""

import json
import os
import time

from codecarbon import EmissionsTracker
from loguru import logger

from src.config import CODECARBON_COUNTRY, EMISSIONS_FILE


class EnergyMonitor:
    """Suit la consommation energetique des traitements."""

    def __init__(self):
        self.tracker = EmissionsTracker(
            country_iso_code=CODECARBON_COUNTRY,
            save_to_file=False,
            log_level="warning",
        )
        self._start_time = None

    def start(self):
        """Demarre le suivi energetique."""
        logger.info("Demarrage du suivi CodeCarbon")
        self._start_time = time.time()
        self.tracker.start()

    def stop(self) -> dict:
        """Arrete le suivi et retourne les metriques.

        Returns:
            Dict avec emissions_kg, duration_s, energy_kwh.
        """
        emissions = self.tracker.stop()
        duration = time.time() - self._start_time if self._start_time else 0
        result = {
            "emissions_kg": float(emissions) if emissions else 0.0,
            "duration_s": round(duration, 2),
            "energy_kwh": float(getattr(self.tracker, '_total_energy', 0) or 0),
        }
        logger.info(
            "Emissions : {:.6f} kg CO2eq en {:.1f}s",
            result["emissions_kg"],
            duration,
        )
        return result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    @staticmethod
    def save_record(record: dict, filepath: str = EMISSIONS_FILE):
        """Ajoute un enregistrement au fichier JSON d'emissions."""
        records = []
        if os.path.exists(filepath):
            with open(filepath) as f:
                try:
                    records = json.load(f)
                except json.JSONDecodeError:
                    records = []
        records.append(record)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
