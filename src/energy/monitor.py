"""Suivi de l'empreinte carbone avec CodeCarbon."""

from codecarbon import EmissionsTracker
from loguru import logger

from src.config import CODECARBON_COUNTRY


class EnergyMonitor:
    """Suit la consommation énergétique des traitements."""

    def __init__(self):
        self.tracker = EmissionsTracker(country_iso_code=CODECARBON_COUNTRY)

    def start(self):
        """Démarre le suivi énergétique."""
        logger.info("Démarrage du suivi CodeCarbon")
        self.tracker.start()

    def stop(self) -> float:
        """Arrête le suivi et retourne les émissions.

        Returns:
            Émissions en kg de CO2eq.
        """
        emissions = self.tracker.stop()
        logger.info("Émissions : {:.6f} kg CO2eq", emissions)
        return emissions
