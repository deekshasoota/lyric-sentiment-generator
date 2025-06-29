# lyric-sentiment-generator
---
title: Lyrics Emotion Analyzer
emoji: 🎶
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: true
license: mit
---

# 🎶 Lyrics Emotion Analyzer

A fine-tuned multilingual emotion analyzer for song lyrics using 🤗 Hugging Face Transformers and Gradio.  
Built with the `SamLowe/roberta-base-go_emotions` model, it detects subtle emotions in user-submitted lyrics.

## 🚀 Features

- 🤖 Uses `roberta-base-go_emotions` for nuanced multi-label emotion detection
- 🌈 Highlights top emotion keywords with color-coded spans
- 📊 Bar chart for top 6 emotions with confidence scores
- 👍👎 User feedback buttons for reinforcement signal collection
- 🎨 Fully customized Gradio dark theme for elegant UI

## 🧠 Technologies Used

- Hugging Face Transformers (`transformers`)
- PyTorch (`torch`)
- Gradio Blocks (`gradio==5.35.0`)
- Matplotlib for visualizing emotion scores
- Regex for keyword highlighting

## 🛠️ Setup Instructions

To run this app locally:

```bash
git clone https://github.com/YOUR_USERNAME/lyrics-emotion-analyzer.git
cd lyrics-emotion-analyzer
pip install -r requirements.txt
python app.py

## 🧪 Sample Input

Paste lyrics like:

You are the sunshine of my life,
That’s why I’ll always be around.
