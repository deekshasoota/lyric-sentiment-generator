from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt

# Emotion classification model
emotion_pipeline = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)


# Emoji mappings
emoji_map = {
    "joy": "😄", "anger": "😠", "sadness": "😢", "fear": "😟", "love": "💘",
    "surprise": "😲", "admiration": "🤩", "neutral": "😐", "optimism": "🌞",
    "pride": "🎖️", "jealousy": "😤", "disgust": "🤢", "remorse": "😔"
}

feedback_data = []

# Emotion detection
def analyze_emotions(lyrics):
    try:
        predictions = emotion_pipeline(lyrics[:512])[0]
        top_emotions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:5]

        fig, ax = plt.subplots()
        labels = [e["label"].capitalize() for e in top_emotions]
        scores = [e["score"] * 100 for e in top_emotions]
        ax.bar(labels, scores, color="#A780B9")
        ax.set_ylabel("Confidence (%)")
        ax.set_title("Top Emotions")
        ax.set_ylim(0, 100)

        emoji_summary = "\n".join(
            [f"{emoji_map.get(e['label'].lower(),'🎵')} {e['label'].capitalize()} — {e['score'] * 100:.1f}%" for e in top_emotions]
        )

        return emoji_summary, fig
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Feedback logging
def collect_feedback(emotion_text, liked):
    feedback_data.append({"emotion": emotion_text, "liked": liked})
    return "✅ Thanks for your feedback!"

# Theme
theme = gr.themes.Soft(primary_hue="purple")

# UI
demo = gr.Blocks(theme=theme, css="""
.gradio-container {
    background-color: #000000 !important;
    color: white !important;
}
.gr-button {
    background-color: #BB86FC !important;
    color: black !important;
    font-weight: bold;
}
.gr-textbox textarea {
    background-color: #1E1E1E !important;
    color: white !important;
}
.gr-plot canvas {
    background-color: #1E1E1E !important;
}
""")

with demo:
    gr.Markdown("## 🎶 Lyrics Emotion Analyzer (with Feedback)")
    gr.Markdown("Paste lyrics to detect top 5 emotions and provide feedback.")

    lyrics = gr.Textbox(label="🎤 Lyrics", lines=6)
    emotion_out = gr.Textbox(label="Detected Emotions", interactive=False)
    plot = gr.Plot(label="Top Emotions Chart")

    analyze = gr.Button("🔍 Analyze Lyrics")

    with gr.Row():
        like = gr.Button("👍 Emotion matched")
        dislike = gr.Button("👎 Didn't match")

    feedback = gr.Textbox(label="Feedback Status", interactive=False)

    analyze.click(fn=analyze_emotions, inputs=lyrics, outputs=[emotion_out, plot])
    like.click(fn=lambda e: collect_feedback(e, True), inputs=emotion_out, outputs=feedback)
    dislike.click(fn=lambda e: collect_feedback(e, False), inputs=emotion_out, outputs=feedback)

if __name__ == "__main__":
    demo.launch()
