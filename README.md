# Thumalien

Pipeline de détection de fake news sur Bluesky avec analyse émotionnelle, dashboard interactif et suivi de l'empreinte carbone.

**Projet M1 Data & IA**

## Architecture

- **Collecte** : Jetstream WebSocket (firehose Bluesky temps réel)
- **Prétraitement** : nettoyage, tokenisation, embeddings (spaCy + Transformers)
- **Détection** : classification fake news (DistilBERT multilingue fine-tuné avec LoRA)
- **Émotion** : analyse émotionnelle (VADER baseline + modèle fine-tuné 6 émotions)
- **Dashboard** : visualisation Streamlit + Plotly
- **Green IT** : suivi énergétique CodeCarbon

## Installation

```bash
cp .env.example .env
# Optionnel : configurer la langue, le nombre de posts, etc.

pip install -r requirements.txt
```

## Entraînement des modèles

### Télécharger les datasets Kaggle

```bash
mkdir -p data/kaggle/fake-news data/kaggle/emotion

# Fake News : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# → Placer True.csv et Fake.csv dans data/kaggle/fake-news/

# Emotion : https://www.kaggle.com/datasets/parulpandey/emotion-dataset
# → Placer training.csv, validation.csv, test.csv dans data/kaggle/emotion/
```

### Lancer l'entraînement

```bash
# Détection fake news (~20-30min sur MacBook M1)
python -m src.training.train_fake_news

# Analyse émotionnelle (~15-20min sur MacBook M1)
python -m src.training.train_emotion
```

Les modèles sont sauvegardés dans `models/fake_news_model/` et `models/emotion_model/`.

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
