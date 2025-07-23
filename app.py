import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re
from collections import Counter
import base64
from io import StringIO
import docx
import PyPDF2

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        required_data = [
            ('punkt_tab', 'tokenizers'),
            ('punkt', 'tokenizers'), 
            ('wordnet', 'corpora'),
            ('averaged_perceptron_tagger', 'taggers'),
            ('stopwords', 'corpora'),
            ('omw-1.4', 'corpora')
        ]
        
        for data, category in required_data:
            try:
                if category == 'tokenizers':
                    nltk.data.find(f'{category}/{data}')
                else:
                    nltk.data.find(f'{category}/{data}')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except:
                    pass  # Fail silently
        
        return True
    except Exception:
        return False

def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        return sent_tokenize(text)
    except:
        # Simple fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

def safe_word_tokenize(text):
    """Safe word tokenization with fallback"""
    try:
        return word_tokenize(text)
    except:
        # Simple fallback word splitting
        return re.findall(r'\b\w+\b', text)

def safe_pos_tag(tokens):
    """Safe POS tagging with fallback"""
    try:
        return pos_tag(tokens)
    except:
        # Simple fallback - assume everything is a noun
        return [(token, 'NN') for token in tokens]

def extract_entities_simple(text):
    """Simple named entity extraction using TextBlob and regex"""
    blob = TextBlob(text)
    
    entities = []
    
    # Extract proper nouns as potential entities
    try:
        pos_tags = blob.tags
        for word, tag in pos_tags:
            if tag in ['NNP', 'NNPS']:  # Proper nouns
                entities.append((word, 'PERSON/ORG'))
    except:
        pass
    
    # Simple regex patterns for common entities
    patterns = {
        'DATE': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)',
        'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
        'PERCENT': r'\d+(?:\.\d+)?%',
        'TIME': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append((match.group(), entity_type))
    
    return entities

def main():
    st.set_page_config(
        page_title="NLP Playground",
        page_icon="🔤",
        layout="wide"
    )

    # Download NLTK data
    download_nltk_data()

    st.title("🔤 NLP Playground: Interactive Text Analysis")
    st.markdown("**Powered by TextBlob & NLTK - No spaCy Required!**")
    st.markdown("---")

    # Text input section
    st.header("📝 Text Input")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Direct Text Input", "File Upload"],
        horizontal=True
    )
    
    if input_method == "Direct Text Input":
        text = st.text_area(
            "Enter your text:",
            value="Natural Language Processing is a fascinating field that combines linguistics and computer science. It helps computers understand, interpret, and generate human language in meaningful ways.",
            height=150
        )
        
        # Add analyze button
        if st.button("🔍 Analyze Text", type="primary"):
            st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                
                st.success(f"File uploaded! ({len(text)} characters)")
                st.text_area("Preview:", text[:300] + "..." if len(text) > 300 else text, height=100)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                text = ""
        else:
            text = "Please upload a file to analyze."

    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return

    st.markdown("---")

    # Basic Statistics
    st.header("📊 Text Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Characters", len(text))
    with col2:
        st.metric("Words", len(text.split()))
    with col3:
        sentences = safe_sent_tokenize(text)
        st.metric("Sentences", len(sentences))
    with col4:
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        st.metric("Paragraphs", paragraphs)

    st.markdown("---")

    # Tokenization & Lemmatization
    st.header("🔤 Tokenization & Lemmatization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Tokenization")
        tokens = safe_word_tokenize(text)
        
        # Display tokens with highlighting
        if tokens:
            token_html = ""
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
            for i, token in enumerate(tokens[:50]):  # Show first 50 tokens
                color = colors[i % len(colors)]
                token_html += f'<span style="background-color: {color}; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block;">{token}</span> '
            
            st.markdown(token_html, unsafe_allow_html=True)
            if len(tokens) > 50:
                st.write(f"... and {len(tokens) - 50} more tokens")
            st.write(f"**Total tokens:** {len(tokens)}")
        
    with col2:
        st.subheader("Lemmatization")
        if tokens:
            try:
                lemmatizer = WordNetLemmatizer()
                lemma_data = []
                for token in tokens[:15]:  # Show first 15 tokens
                    if token.isalpha():
                        lemma = lemmatizer.lemmatize(token.lower())
                        lemma_data.append({"Original": token, "Lemmatized": lemma})
                
                if lemma_data:
                    df = pd.DataFrame(lemma_data)
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Lemmatization error: {e}")

    st.markdown("---")

    # Named Entity Recognition (Simple version)
    st.header("🏷️ Named Entity Recognition")
    
    entities = extract_entities_simple(text)
    
    if entities:
        # Display text with highlighted entities
        st.subheader("Highlighted Entities")
        highlighted_text = text
        
        # Entity colors
        entity_colors = {
            "PERSON/ORG": "#FF6B6B", "DATE": "#4ECDC4", "MONEY": "#45B7D1",
            "PERCENT": "#96CEB4", "TIME": "#FFEAA7", "EMAIL": "#DDA0DD",
            "PHONE": "#98D8C8"
        }
        
        # Highlight entities (simple approach)
        for ent_text, ent_label in entities:
            color = entity_colors.get(ent_label, "#CCCCCC")
            if ent_text in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    ent_text, 
                    f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{ent_text} ({ent_label})</mark>',
                    1  # Replace only first occurrence
                )
        
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Entity statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entity Details")
            if entities:
                entity_df = pd.DataFrame(entities, columns=["Text", "Label"])
                st.dataframe(entity_df, use_container_width=True)
        
        with col2:
            st.subheader("Entity Distribution")
            if entities:
                entity_counts = Counter([ent[1] for ent in entities])
                fig = px.pie(
                    values=list(entity_counts.values()),
                    names=list(entity_counts.keys()),
                    title="Entity Types"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entities detected in the text.")

    st.markdown("---")

    # Part-of-Speech Tagging
    st.header("📚 Part-of-Speech Tagging")
    
    # POS tagging results
    if tokens:
        pos_tags = safe_pos_tag(tokens[:50])  # Limit to first 50 tokens
        
        pos_data = []
        for token, pos in pos_tags:
            if token.strip():
                # Simple POS explanations
                pos_explanations = {
                    'NN': 'Noun', 'NNS': 'Plural Noun', 'NNP': 'Proper Noun',
                    'VB': 'Verb', 'VBD': 'Past Tense Verb', 'VBG': 'Gerund/Present Participle',
                    'VBN': 'Past Participle', 'VBP': 'Present Verb', 'VBZ': '3rd Person Singular Verb',
                    'JJ': 'Adjective', 'JJR': 'Comparative Adjective', 'JJS': 'Superlative Adjective',
                    'RB': 'Adverb', 'RBR': 'Comparative Adverb', 'RBS': 'Superlative Adverb',
                    'DT': 'Determiner', 'IN': 'Preposition', 'CC': 'Conjunction',
                    'PRP': 'Pronoun', 'TO': 'To', 'CD': 'Number'
                }
                
                pos_data.append({
                    "Token": token,
                    "POS": pos,
                    "Description": pos_explanations.get(pos, pos)
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
                st.plotly_chart(fig, use_container_width=True)
            
            # Highlighted text with POS colors
            st.subheader("Text with POS Highlighting")
            pos_colors = {
                "NN": "#FF6B6B", "NNS": "#FF6B6B", "NNP": "#FF6B6B",
                "VB": "#4ECDC4", "VBD": "#4ECDC4", "VBG": "#4ECDC4", "VBN": "#4ECDC4", "VBP": "#4ECDC4", "VBZ": "#4ECDC4",
                "JJ": "#45B7D1", "JJR": "#45B7D1", "JJS": "#45B7D1",
                "RB": "#96CEB4", "RBR": "#96CEB4", "RBS": "#96CEB4",
                "DT": "#FFEAA7", "IN": "#DDA0DD", "CC": "#98D8C8"
            }
            
            highlighted_text = ""
            for token, pos in pos_tags:
                if token.strip():
                    color = pos_colors.get(pos, "#CCCCCC")
                    highlighted_text += f'<span style="background-color: {color}; padding: 1px 3px; margin: 1px; border-radius: 2px;" title="{pos}">{token}</span> '
            
            st.markdown(highlighted_text, unsafe_allow_html=True)

    st.markdown("---")

    # Sentiment Analysis
    st.header("😊 Sentiment Analysis")
    
    # Overall sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Polarity gauge
        fig_pol = go.Figure(go.Indicator(
            mode="gauge+number",
            value=polarity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Polarity"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkgreen" if polarity > 0 else "darkred" if polarity < 0 else "gray"},
                'steps': [
                    {'range': [-1, -0.5], 'color': "lightcoral"},
                    {'range': [-0.5, 0], 'color': "lightyellow"},
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "green"}
                ]
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
                'axis': {'range': [0, 1]},
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
        st.write(f"**Polarity:** {polarity:.3f}")
        st.write(f"**Subjectivity:** {subjectivity:.3f}")
        
        st.write("**Scale:**")
        st.write("• Polarity: -1 (negative) to +1 (positive)")
        st.write("• Subjectivity: 0 (objective) to 1 (subjective)")
    
    # Sentence-by-sentence analysis
    if len(sentences) > 1:
        st.subheader("Sentence-by-Sentence Analysis")
        
        sent_data = []
        for i, sentence in enumerate(sentences, 1):
            sentence_text = sentence.strip()
            if sentence_text:
                sent_blob = TextBlob(sentence_text)
                sent_data.append({
                    "Sentence": i,
                    "Text": sentence_text,
                    "Polarity": sent_blob.sentiment.polarity,
                    "Subjectivity": sent_blob.sentiment.subjectivity
                })
        
        if sent_data:
            df = pd.DataFrame(sent_data)
            st.dataframe(df, use_container_width=True)
            
            # Sentiment trend
            if len(sent_data) > 1:
                fig = px.line(
                    df, x="Sentence", y="Polarity",
                    title="Sentiment Trend Across Sentences",
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Word Frequency Analysis
    st.header("📈 Word Frequency Analysis")
    
    # Get word frequencies
    words = [token.lower() for token in safe_word_tokenize(text) if token.isalpha() and len(token) > 2]
    
    if words:
        word_freq = Counter(words)
        top_words = word_freq.most_common(15)
        
        if top_words:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 15 Most Common Words")
                words_list, counts_list = zip(*top_words)
                fig = px.bar(
                    x=list(counts_list), 
                    y=list(words_list),
                    orientation='h',
                    title="Word Frequency"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Word Cloud Data")
                freq_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
                st.dataframe(freq_df, use_container_width=True)

    st.markdown("---")
    st.markdown("**🔤 NLP Playground** - Built with Streamlit, TextBlob & NLTK")
    st.markdown("*No spaCy Required!* 🎉")

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {e}")

if __name__ == "__main__":
    main()