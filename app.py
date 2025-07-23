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
@st.cache_resource
def download_nltk_data():
    try:
        # Try the new punkt_tab first
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                # Fallback to regular punkt if punkt_tab fails
                nltk.download('punkt', quiet=True)
        
        # Download other required data
        required_data = ['wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4']
        for data in required_data:
            try:
                nltk.data.find(f'corpora/{data}' if data in ['wordnet', 'stopwords', 'omw-1.4'] else f'taggers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
        
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy English model not found. Please install it using: python -m spacy download en_core_web_sm")
        return None

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

def main():
    st.set_page_config(
        page_title="NLP Playground",
        page_icon="🔤",
        layout="wide"
    )

    # Download NLTK data
    if not download_nltk_data():
        st.warning("Some NLTK features may not work properly.")

    # Load spaCy model
    nlp = load_spacy_model()
    if nlp is None:
        st.stop()

    st.title("🔤 NLP Playground: Interactive Text Analysis")
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
            value="Natural Language Processing is a fascinating field that combines linguistics and computer science. It helps computers understand, interpret, and generate human language.",
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

    # Named Entity Recognition
    st.header("🏷️ Named Entity Recognition")
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
    
    if entities:
        # Display text with highlighted entities
        st.subheader("Highlighted Entities")
        highlighted_text = text
        
        # Entity colors
        entity_colors = {
            "PERSON": "#FF6B6B", "ORG": "#4ECDC4", "GPE": "#45B7D1",
            "DATE": "#96CEB4", "TIME": "#FFEAA7", "MONEY": "#DDA0DD",
            "PERCENT": "#98D8C8", "CARDINAL": "#F7DC6F", "ORDINAL": "#BB8FCE"
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
                title="Entity Types"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No named entities found in the text.")

    st.markdown("---")

    # Part-of-Speech Tagging
    st.header("📚 Part-of-Speech Tagging")
    
    # POS tagging results
    pos_data = []
    for token in doc:
        if not token.is_space and token.text.strip():
            pos_data.append({
                "Token": token.text,
                "POS": token.pos_,
                "Tag": token.tag_,
                "Lemma": token.lemma_,
                "Description": spacy.explain(token.pos_) or token.pos_
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
            "NOUN": "#FF6B6B", "VERB": "#4ECDC4", "ADJ": "#45B7D1",
            "ADV": "#96CEB4", "PRON": "#FFEAA7", "DET": "#DDA0DD",
            "ADP": "#98D8C8", "CONJ": "#F7DC6F", "NUM": "#BB8FCE"
        }
        
        highlighted_text = ""
        for token in doc:
            if not token.is_space and token.text.strip():
                color = pos_colors.get(token.pos_, "#CCCCCC")
                highlighted_text += f'<span style="background-color: {color}; padding: 1px 3px; margin: 1px; border-radius: 2px;" title="{token.pos_}: {spacy.explain(token.pos_) or token.pos_}">{token.text}</span> '
            else:
                highlighted_text += " "
        
        st.markdown(highlighted_text, unsafe_allow_html=True)

    st.markdown("---")

    # Dependency Parsing
    st.header("🌳 Dependency Parsing")
    
    # Process first sentence for dependency parsing
    sentences = list(doc.sents)
    
    if sentences:
        sentence = sentences[0]
        st.subheader(f"Dependency Tree for: '{sentence.text}'")
        
        # Create dependency data
        dep_data = []
        for token in sentence:
            if not token.is_space and token.text.strip():
                dep_data.append({
                    "Token": token.text,
                    "Dependency": token.dep_,
                    "Head": token.head.text,
                    "Description": spacy.explain(token.dep_) or token.dep_
                })
        
        if dep_data:
            st.subheader("Dependency Relations")
            df = pd.DataFrame(dep_data)
            st.dataframe(df, use_container_width=True)
            
            # Simple dependency visualization
            st.subheader("Dependency Tree Visualization")
            
            try:
                # Create a simple bar chart showing dependency types
                dep_counts = Counter([item["Dependency"] for item in dep_data])
                
                fig = px.bar(
                    x=list(dep_counts.keys()),
                    y=list(dep_counts.values()),
                    title="Dependency Types in Sentence",
                    labels={'x': 'Dependency Type', 'y': 'Count'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                # Fallback: Simple text visualization
                st.write("**Dependency Tree (Text Format):**")
                
                # Create a simple text-based dependency tree
                dep_text = ""
                for item in dep_data:
                    dep_text += f"**{item['Token']}** --[{item['Dependency']}]--> **{item['Head']}**\n\n"
                
                st.markdown(dep_text)
                
                # Show dependency connections as a table
                st.write("**Dependency Connections:**")
                connections = []
                for item in dep_data:
                    if item['Token'] != item['Head']:
                        connections.append({
                            "From": item['Token'],
                            "Relation": item['Dependency'], 
                            "To": item['Head']
                        })
                
                if connections:
                    conn_df = pd.DataFrame(connections)
                    st.dataframe(conn_df, use_container_width=True)

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
            # Convert spaCy Span to string
            sentence_text = sentence.text if hasattr(sentence, 'text') else str(sentence)
            sent_blob = TextBlob(sentence_text)
            sent_data.append({
                "Sentence": i,
                "Text": sentence_text,
                "Polarity": sent_blob.sentiment.polarity,
                "Subjectivity": sent_blob.sentiment.subjectivity
            })
        
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
    st.markdown("**🔤 NLP Playground** - Built with Streamlit, spaCy, NLTK, and TextBlob")

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