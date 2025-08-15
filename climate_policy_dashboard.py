import os
import nltk
import streamlit as st
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from collections import defaultdict

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Folder Path (relative to repo)
folder_path = './Preprocessed_text'
texts = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokens = word_tokenize(text.lower())
            texts.append(tokens)
            filenames.append(filename)

# LDA Model
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=20)
topic_labels = [f"Topic {i}: {lda_model.print_topic(i, topn=3)}" for i in range(5)]

# Sector Classification (simplified)
sector_data = defaultdict(list)
for i, filename in enumerate(filenames):
    text = ' '.join(texts[i])
    sector = 'unknown'
    for sec, keywords in sector_keywords.items():
        if any(keyword in text for keyword in keywords):
            sector = sec
            break
    year = filename.split('_')[0] if '_' in filename else 'unknown_year'
    topics = lda_model[corpus[i]]
    sector_data[sector].append({'year': year, 'topics': topics})

# Dashboard
st.title("Climate Policy Trends Dashboard")
valid_sectors = ['energy', 'transport', 'urban_development', 'unknown']
selected_sector = st.text_input("Search Sector (energy, transport, urban_development, unknown):").lower().strip()
if not selected_sector or selected_sector not in valid_sectors:
    st.error("Please enter a valid sector.")
    st.stop()

years_available = sorted(set(data['year'] for data in sector_data[selected_sector]))
selected_years = st.multiselect("Select Years (or all for combined):", years_available, default=years_available)
selected_topic = st.selectbox("Select Topic", list(range(5)) + [-1], format_func=lambda x: topic_labels[x] if x >= 0 else "All Topics")

# Filter Data
topic_data = defaultdict(list)
for data in sector_data[selected_sector]:
    year = data['year']
    if year in selected_years or not selected_years:
        for topic_id, prob in data['topics']:
            topic_data[topic_id].append(prob)

# Debug
st.write("Selected Sector:", selected_sector)
st.write("Selected Years:", selected_years)
st.write("Topic Data:", dict(topic_data))

# Trend Line Chart
plt.figure(figsize=(10, 6))
for topic_id in topic_data:
    years = [d['year'] for d in sector_data[selected_sector] if d['year'] in selected_years]
    probs = topic_data[topic_id]
    plt.plot(years, probs, label=topic_labels[topic_id], marker='o')
plt.xlabel('Year')
plt.ylabel('Topic Probability')
plt.title(f'Topic Trends - {selected_sector.capitalize()}')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Combined Bar Chart
if len(selected_years) > 1 and topic_data:
    combined_data = [sum(topic_data[tid]) / len(topic_data[tid]) if topic_data[tid] else 0 for tid in range(5)]
    plt.figure(figsize=(10, 6))
    plt.bar(topic_labels, combined_data)
    plt.title(f'Combined Topic Probabilities - {selected_sector.capitalize()}')
    st.pyplot(plt)
else:
    st.warning("No data for combined chart. Select multiple years or check data.")
