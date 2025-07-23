import streamlit as st
import spacy
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from collections import Counter
import base64
from io import StringIO
import docx
import PyPDF2

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

def main():
    st.set_page_config(
        page_title="NLP Playground",
        page_icon="🔤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔤 NLP Playground: Interactive NLP Demos")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("🎯 Navigation")
    demo_type = st.sidebar.selectbox(
        "Choose NLP Demo:",
        [
            "Text Input & Upload",
            "Tokenization & Lemmatization",
            "Named Entity Recognition",
            "POS Tagging",
            "Dependency Parsing",
            "Sentiment Analysis",
            "All-in-One Analysis"
        ]
    )

    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        st.stop()

    # Text input section
    if demo_type == "Text Input & Upload":
        text_input_demo()
    elif demo_type == "Tokenization & Lemmatization":
        tokenization_demo(get_text())
    elif demo_type == "Named Entity Recognition":
        ner_demo(get_text(), nlp)
    elif demo_type == "POS Tagging":
        pos_demo(get_text(), nlp)
    elif demo_type == "Dependency Parsing":
        dependency_demo(get_text(), nlp)
    elif demo_type == "Sentiment Analysis":
        sentiment_demo(get_text())
    elif demo_type == "All-in-One Analysis":
        all_in_one_demo(get_text(), nlp)

def get_text():
    """Get text from session state or provide default"""
    if 'analyzed_text' not in st.session_state:
        st.session_state.analyzed_text = "Natural Language Processing is a fascinating field that combines linguistics and computer science."
    return st.session_state.analyzed_text

def text_input_demo():
    st.header("📝 Text Input & File Upload")
    
    # Text input methods
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "File Upload"]
    )
    
    if input_method == "Direct Text Input":
        text = st.text_area(
            "Enter your text:",
            value=get_text(),
            height=200,
            help="Enter any text you want to analyze"
        )
        
        if st.button("Analyze This Text"):
            st.session_state.analyzed_text = text
            st.success("Text saved! Navigate to other demos to analyze it.")
            
    else:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx'],
            help="Upload a text file, PDF, or Word document"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                
                st.session_state.analyzed_text = text
                st.success(f"File uploaded successfully! ({len(text)} characters)")
                st.text_area("Preview:", text[:500] + "..." if len(text) > 500 else text, height=150)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX file"""
    doc = docx.Document(uploaded_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def tokenization_demo(text):
    st.header("🔤 Tokenization & Lemmatization")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Tokenization")
        tokens = word_tokenize(text)
        
        # Display tokens with highlighting
        token_html = ""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
        for i, token in enumerate(tokens):
            color = colors[i % len(colors)]
            token_html += f'<span style="background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block;">{token}</span>'
        
        st.markdown(token_html, unsafe_allow_html=True)
        st.write(f"**Total tokens:** {len(tokens)}")
        
        # Token statistics
        token_lengths = [len(token) for token in tokens]
        st.write(f"**Average token length:** {sum(token_lengths)/len(token_lengths):.2f}")
        
    with col2:
        st.subheader("Lemmatization")
        lemmatizer = WordNetLemmatizer()
        
        # Create lemmatization table
        lemma_data = []
        for token in tokens[:20]:  # Show first 20 tokens
            if token.isalpha():
                lemma = lemmatizer.lemmatize(token.lower())
                lemma_data.append({"Original": token, "Lemmatized": lemma})
        
        if lemma_data:
            df = pd.DataFrame(lemma_data)
            st.dataframe(df, use_container_width=True)
        
    # Sentence tokenization
    st.subheader("Sentence Tokenization")
    sentences = sent_tokenize(text)
    for i, sentence in enumerate(sentences, 1):
        st.write(f"**Sentence {i}:** {sentence}")

def ner_demo(text, nlp):
    st.header("🏷️ Named Entity Recognition")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
    
    if entities:
        # Display text with highlighted entities
        st.subheader("Highlighted Entities")
        highlighted_text = text
        
        # Entity colors
        entity_colors = {
            "PERSON": "#FF6B6B",
            "ORG": "#4ECDC4", 
            "GPE": "#45B7D1",
            "DATE": "#96CEB4",
            "TIME": "#FFEAA7",
            "MONEY": "#DDA0DD",
            "PERCENT": "#98D8C8",
            "CARDINAL": "#F7DC6F",
            "ORDINAL": "#BB8FCE"
        }
        
        # Sort entities by start position (reverse order for replacement)
        entities_sorted = sorted(entities, key=lambda x: x[2], reverse=True)
        
        for ent_text, ent_label, start, end in entities_sorted:
            color = entity_colors.get(ent_label, "#CCCCCC")
            replacement = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{ent_text} ({ent_label})</mark>'
            highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
        
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Entity statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entity Details")
            entity_df = pd.DataFrame(entities, columns=["Text", "Label", "Start", "End"])
            st.dataframe(entity_df, use_container_width=True)
        
        with col2:
            st.subheader("Entity Distribution")
            entity_counts = Counter([ent[1] for ent in entities])
            fig = px.pie(
                values=list(entity_counts.values()),
                names=list(entity_counts.keys()),
                title="Entity Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Entity explanations
        st.subheader("Entity Type Explanations")
        explanations = {
            "PERSON": "People, including fictional characters",
            "ORG": "Organizations, companies, agencies, institutions",
            "GPE": "Countries, cities, states (Geopolitical entities)",
            "DATE": "Absolute or relative dates or periods",
            "TIME": "Times smaller than a day",
            "MONEY": "Monetary values, including unit",
            "PERCENT": "Percentage, including '%'",
            "CARDINAL": "Numerals that do not fall under another type",
            "ORDINAL": "First, second, etc."
        }
        
        found_types = set(ent[1] for ent in entities)
        for ent_type in found_types:
            if ent_type in explanations:
                st.write(f"**{ent_type}:** {explanations[ent_type]}")
    else:
        st.info("No named entities found in the text.")

def pos_demo(text, nlp):
    st.header("📚 Part-of-Speech Tagging")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    doc = nlp(text)
    
    # POS tagging results
    pos_data = []
    for token in doc:
        if not token.is_space:
            pos_data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Tag": token.tag_,
                "Lemma": token.lemma_,
                "Description": spacy.explain(token.pos_)
            })
    
    if pos_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("POS Tags Table")
            df = pd.DataFrame(pos_data)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("POS Distribution")
            pos_counts = Counter([item["POS"] for item in pos_data])
            fig = px.bar(
                x=list(pos_counts.keys()),
                y=list(pos_counts.values()),
                title="Part-of-Speech Distribution"
            )
            fig.update_layout(xaxis_title="POS Tag", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        # Highlighted text with POS colors
        st.subheader("Text with POS Highlighting")
        pos_colors = {
            "NOUN": "#FF6B6B", "VERB": "#4ECDC4", "ADJ": "#45B7D1",
            "ADV": "#96CEB4", "PRON": "#FFEAA7", "DET": "#DDA0DD",
            "ADP": "#98D8C8", "CONJ": "#F7DC6F", "NUM": "#BB8FCE"
        }
        
        highlighted_text = ""
        for token in doc:
            if not token.is_space:
                color = pos_colors.get(token.pos_, "#CCCCCC")
                highlighted_text += f'<span style="background-color: {color}; padding: 1px 3px; margin: 1px; border-radius: 2px;" title="{token.pos_}: {spacy.explain(token.pos_)}">{token.text}</span> '
            else:
                highlighted_text += " "
        
        st.markdown(highlighted_text, unsafe_allow_html=True)

def dependency_demo(text, nlp):
    st.header("🌳 Dependency Parsing")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    # Process first sentence for dependency parsing
    doc = nlp(text)
    sentences = list(doc.sents)
    
    if sentences:
        sentence = sentences[0]
        st.subheader(f"Dependency Tree for: '{sentence.text}'")
        
        # Create dependency data
        dep_data = []
        for token in sentence:
            dep_data.append({
                "Token": token.text,
                "Dependency": token.dep_,
                "Head": token.head.text,
                "Description": spacy.explain(token.dep_)
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dependency Relations")
            df = pd.DataFrame(dep_data)
            st.dataframe(df, use_container_width=True)
        
        with col2:
            st.subheader("Dependency Visualization")
            # Create a simple tree-like visualization
            fig = go.Figure()
            
            # Add nodes and edges for dependency tree
            tokens = [token.text for token in sentence]
            deps = [token.dep_ for token in sentence]
            
            # Create a simple network layout
            x_pos = list(range(len(tokens)))
            y_pos = [0] * len(tokens)
            
            # Draw connections
            for i, token in enumerate(sentence):
                head_idx = next((j for j, t in enumerate(sentence) if t == token.head), i)
                if head_idx != i:
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[head_idx]],
                        y=[y_pos[i], y_pos[head_idx] + 0.1],
                        mode='lines',
                        line=dict(color='gray'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
            
            # Add tokens
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                text=tokens,
                textposition="bottom center",
                marker=dict(size=20, color='lightblue'),
                showlegend=False,
                hovertext=[f"{token}: {dep}" for token, dep in zip(tokens, deps)],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Dependency Relations",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

def sentiment_demo(text):
    st.header("😊 Sentiment Analysis")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    # Overall sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Polarity gauge
        fig_pol = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=polarity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Polarity"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkgreen" if polarity > 0 else "darkred" if polarity < 0 else "gray"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "lightcoral"},
                    {'range': [-0.5, 0], 'color': "lightyellow"},
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig_pol.update_layout(height=300)
        st.plotly_chart(fig_pol, use_container_width=True)
    
    with col2:
        # Subjectivity gauge
        fig_sub = go.Figure(go.Indicator(
            mode="gauge+number",
            value=subjectivity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Subjectivity"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightblue"},
                    {'range': [0.5, 1], 'color': "blue"}
                ]
            }
        ))
        fig_sub.update_layout(height=300)
        st.plotly_chart(fig_sub, use_container_width=True)
    
    with col3:
        st.subheader("Interpretation")
        
        # Polarity interpretation
        if polarity > 0.1:
            pol_text = "Positive 😊"
            pol_color = "green"
        elif polarity < -0.1:
            pol_text = "Negative 😞"
            pol_color = "red"
        else:
            pol_text = "Neutral 😐"
            pol_color = "gray"
        
        st.markdown(f"**Sentiment:** <span style='color: {pol_color}'>{pol_text}</span>", unsafe_allow_html=True)
        st.write(f"**Polarity Score:** {polarity:.3f}")
        st.write(f"**Subjectivity Score:** {subjectivity:.3f}")
        
        st.write("**Scale:**")
        st.write("Polarity: -1 (negative) to +1 (positive)")
        st.write("Subjectivity: 0 (objective) to 1 (subjective)")
    
    # Sentence-by-sentence analysis
    st.subheader("Sentence-by-Sentence Analysis")
    sentences = sent_tokenize(text)
    
    if len(sentences) > 1:
        sent_data = []
        for i, sentence in enumerate(sentences, 1):
            sent_blob = TextBlob(sentence)
            sent_data.append({
                "Sentence": i,
                "Text": sentence,
                "Polarity": sent_blob.sentiment.polarity,
                "Subjectivity": sent_blob.sentiment.subjectivity
            })
        
        df = pd.DataFrame(sent_data)
        st.dataframe(df, use_container_width=True)
        
        # Sentiment trend
        fig = px.line(
            df, x="Sentence", y="Polarity",
            title="Sentiment Trend Across Sentences",
            markers=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

def all_in_one_demo(text, nlp):
    st.header("🎯 All-in-One NLP Analysis")
    
    if not text.strip():
        st.warning("Please enter some text in the 'Text Input & Upload' section first.")
        return
    
    st.subheader("📊 Text Statistics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Characters", len(text))
    with col2:
        st.metric("Words", len(text.split()))
    with col3:
        st.metric("Sentences", len(sent_tokenize(text)))
    with col4:
        st.metric("Paragraphs", len([p for p in text.split('\n\n') if p.strip()]))
    
    # Quick analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Entities", "😊 Sentiment", "📚 POS", "🔤 Tokens"])
    
    with tab1:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
            st.dataframe(entity_df, use_container_width=True)
        else:
            st.info("No entities found")
    
    with tab2:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment_emoji = "😊" if polarity > 0.1 else "😞" if polarity < -0.1 else "😐"
        st.write(f"**Overall Sentiment:** {sentiment_emoji} ({polarity:.3f})")
        
        # Quick sentiment bar
        fig = go.Figure(go.Bar(
            x=['Sentiment'],
            y=[polarity],
            marker_color='green' if polarity > 0 else 'red' if polarity < 0 else 'gray'
        ))
        fig.update_layout(yaxis_range=[-1, 1], height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        doc = nlp(text)
        pos_counts = Counter([token.pos_ for token in doc if not token.is_space])
        if pos_counts:
            fig = px.pie(values=list(pos_counts.values()), names=list(pos_counts.keys()))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        tokens = word_tokenize(text)
        st.write(f"**Token count:** {len(tokens)}")
        st.write(f"**Unique tokens:** {len(set(tokens))}")
        
        # Most common words
        word_freq = Counter([token.lower() for token in tokens if token.isalpha()])
        common_words = word_freq.most_common(10)
        if common_words:
            words, counts = zip(*common_words)
            fig = px.bar(x=list(words), y=list(counts), title="Top 10 Most Common Words")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()