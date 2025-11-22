"""
Day 1: First LLM - Sentiment Analysis
Goal: Understand how pretrained models work
"""

from transformers import pipeline

# Load sentiment classifier
classifier = pipeline("sentiment-analysis")

# Test sentences
test_sentences = [
    "I'm excited to learn about LLMs!",
    "This is amazing!",
    "I'm frustrated and confused.",
    "Great, another bug to fix.",
]

print("=" * 50)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 50)

for sentence in test_sentences:
    result = classifier(sentence)
    label = result[0]['label']
    score = result[0]['score']
    print(f"\n'{sentence}'")
    print(f"→ {label} ({score:.2%})")
