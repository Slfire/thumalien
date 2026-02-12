# Thumalien

Pipeline de détection de fake news sur Bluesky avec analyse émotionnelle, dashboard interactif et suivi de l'empreinte carbone.

**Projet M1 Data & IA**

## Architecture

- **Collecte** : client AT Protocol (Bluesky)
- **Prétraitement** : nettoyage, tokenisation, embeddings (spaCy + Transformers)
- **Détection** : classification fake news avec score de crédibilité
- **Émotion** : analyse émotionnelle (VADER + BERT)
- **Dashboard** : visualisation Streamlit + Plotly
- **Green IT** : suivi énergétique CodeCarbon

## Installation

```bash
cp .env.example .env
# Éditer .env avec vos identifiants

pip install -r requirements.txt
```

## Lancement

```bash
# Avec Docker
docker compose up

# Sans Docker
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/
```
