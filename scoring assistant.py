# Import required libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr
import re
from collections import Counter, defaultdict
import requests
import time
from typing import List, Dict, Tuple
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df[['PMID', 'Title', 'Abstract']].dropna()
    df['text'] = "Title: " + df['Title'] + "\nAbstract: " + df['Abstract']
    return df

class ProteinExtractor:
    def __init__(self):
        # More precise protein patterns
        self.protein_patterns = [
            r'\b[A-Z][A-Z0-9]{1,10}[0-9]+[A-Z]*\b',  # EGFR, TP53, PIK3CA style
            r'\bp[0-9]{2,3}[A-Za-z]*\b',  # p53, p21, p27 style
            r'\b[A-Z]{2,6}[0-9]*[A-Z]*\b',  # PTEN, VEGF, mTOR style
            r'\b[A-Z][a-z]{2,8}[0-9]+[A-Z]*\b',  # Bax1, Ras1 style
            r'\b[A-Z][A-Z]{1,4}-[0-9]+[A-Z]*\b',  # MDM-2 style
        ]
        
        # Known protein families and suffixes that indicate proteins
        self.protein_indicators = {
            'kinase', 'phosphatase', 'receptor', 'factor', 'enzyme', 'channel',
            'transporter', 'ligase', 'protease', 'oxidase', 'reductase',
            'synthase', 'hydrolase', 'transferase', 'isomerase', 'lyase',
            'protein', 'gene', 'oncogene', 'suppressor'
        }
        
        # Extended false positives list
        self.false_positives = {
            'Abstract', 'Title', 'Methods', 'Results', 'Conclusion', 'Background',
            'DNA', 'RNA', 'mRNA', 'cDNA', 'PCR', 'qPCR', 'RT-PCR', 'Western', 
            'Northern', 'Southern', 'ELISA', 'ICC', 'IHC', 'FACS', 'FISH',
            'PBS', 'DMSO', 'FBS', 'BSA', 'EDTA', 'Tris', 'HEPES', 'RPMI', 
            'DMEM', 'SDS', 'PAGE', 'ANOVA', 'SPSS', 'Fig', 'Table', 'USA',
            'UK', 'Germany', 'China', 'Japan', 'WHO', 'FDA', 'NIH', 'NCBI',
            'PubMed', 'Google', 'Microsoft', 'Apple', 'IBM', 'Inc', 'Ltd',
            'Co', 'Corp', 'University', 'Institute', 'Hospital', 'Center',
            'Department', 'School', 'College', 'Laboratory', 'Lab', 'Research'
        }
        
        # Functionality keywords
        self.functionality_keywords = {
            'kinase', 'phosphatase', 'transcription factor', 'receptor', 'enzyme',
            'channel', 'transporter', 'ligase', 'protease', 'oxidase', 'reductase',
            'synthase', 'hydrolase', 'transferase', 'isomerase', 'lyase',
            'oncogene', 'tumor suppressor', 'growth factor', 'cytokine',
            'signaling', 'pathway', 'cascade', 'regulation', 'activation',
            'inhibition', 'binding', 'interaction', 'expression', 'overexpression'
        }
        
        # Research relevance indicators
        self.relevance_keywords = {
            'novel', 'new', 'recently discovered', 'emerging', 'previously unknown',
            'first time', 'newly identified', 'uncharacterized', 'poorly characterized',
            'little known', 'recently identified', 'emerging role', 'potential target',
            'promising', 'candidate', 'putative', 'potential'
        }

    def extract_proteins_from_text(self, text: str) -> List[str]:
        """Extract potential protein names from text with improved filtering"""
        proteins = set()
        
        # Apply regex patterns
        for pattern in self.protein_patterns:
            matches = re.findall(pattern, text)
            proteins.update(matches)
        
        # Look for protein indicators in context
        words = text.split()
        for i, word in enumerate(words):
            # Check if word matches pattern and has protein context
            for pattern in self.protein_patterns:
                if re.match(pattern, word):
                    # Check surrounding context for protein indicators
                    context_window = words[max(0, i-3):i+4]
                    context_text = ' '.join(context_window).lower()
                    
                    if any(indicator in context_text for indicator in self.protein_indicators):
                        proteins.add(word)
        
        # Clean and filter proteins
        cleaned_proteins = []
        for protein in proteins:
            protein = protein.strip()
            if (len(protein) >= 2 and 
                protein not in self.false_positives and
                not protein.isdigit() and
                not all(c.isdigit() or c in '.-' for c in protein) and
                not protein.lower().startswith(('http', 'www', 'doi', 'pmid')) and
                len(protein) <= 15):  # Reasonable protein name length
                cleaned_proteins.append(protein)
        
        return list(set(cleaned_proteins))

class ProteinScorer:
    def __init__(self, extractor: ProteinExtractor, total_abstracts: int):
        self.extractor = extractor
        self.total_abstracts = total_abstracts
        
    def score_protein(self, protein: str, contexts: List[str], frequency: int) -> Dict:
        """Score a protein based on literature scarcity, research relevance, and functionality"""
        
        # Combine all contexts mentioning this protein
        full_context = ' '.join(contexts).lower()
        protein_lower = protein.lower()
        
        # Initialize scores
        scores = {
            'undercharacterization_score': 0.0,
            'research_relevance_score': 0.0, 
            'functionality_score': 0.0,
            'total_score': 0.0,
            'evidence': {
                'undercharacterization': [],
                'research_relevance': [],
                'functionality': []
            }
        }
        
        # 1. Undercharacterization Score (0.0-1.0): Higher score for less literature
        # Uses inverse frequency with logarithmic scaling
        if frequency > 0:
            # Normalize frequency relative to total abstracts
            freq_ratio = frequency / self.total_abstracts
            # Apply inverse logarithmic scaling (less literature = higher score)
            underchar_base = max(0.0, 1.0 - np.log10(frequency + 1) / np.log10(self.total_abstracts + 1))
            
            # Boost score if undercharacterization keywords are present
            underchar_keywords_found = []
            keyword_boost = 0.0
            for keyword in ['unknown function', 'uncharacterized', 'poorly understood', 
                           'limited knowledge', 'mechanism unclear', 'role unclear',
                           'function unknown', 'poorly characterized', 'understudied',
                           'little known', 'needs investigation']:
                if keyword in full_context:
                    keyword_boost += 0.1
                    underchar_keywords_found.append(keyword)
            
            scores['undercharacterization_score'] = min(1.0, underchar_base + keyword_boost)
            scores['evidence']['undercharacterization'] = underchar_keywords_found
        
        # 2. Research Relevance Score (0.0-1.0): Based on research interest indicators
        relevance_keywords_found = []
        relevance_score = 0.0
        for keyword in self.extractor.relevance_keywords:
            if keyword in full_context:
                relevance_score += 0.15  # Each keyword adds to relevance
                relevance_keywords_found.append(keyword)
        
        scores['research_relevance_score'] = min(1.0, relevance_score)
        scores['evidence']['research_relevance'] = relevance_keywords_found
        
        # 3. Functionality Score (0.0-1.0): Based on functional relevance
        func_keywords_found = []
        func_score = 0.0
        for keyword in self.extractor.functionality_keywords:
            if keyword in full_context:
                func_score += 0.1  # Each functional keyword adds value
                func_keywords_found.append(keyword)
        
        # Additional functional context scoring
        functional_contexts = [
            'glioma', 'glioblastoma', 'astrocytoma', 'oligodendroglioma',
            'brain tumor', 'brain cancer', 'neural', 'neuronal',
            'therapeutic target', 'biomarker', 'prognosis', 'diagnosis'
        ]
        
        for context in functional_contexts:
            if context in full_context:
                func_score += 0.05
                func_keywords_found.append(f'context_{context.replace(" ", "_")}')
        
        scores['functionality_score'] = min(1.0, func_score)
        scores['evidence']['functionality'] = func_keywords_found
        
        # Calculate weighted total score (prioritizing undercharacterization)
        scores['total_score'] = (0.5 * scores['undercharacterization_score'] + 
                               0.25 * scores['research_relevance_score'] + 
                               0.25 * scores['functionality_score'])
        
        return scores

class GliomaProteinAnalyzer:
    def __init__(self, data_path):
        # Load data
        self.df = load_data(data_path)
        self.documents = self.df['text'].tolist()
        
        # Initialize components
        self.extractor = ProteinExtractor()
        self.scorer = ProteinScorer(self.extractor, len(self.documents))
        
        # Initialize models for semantic search
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # Create vector database
        self._create_vector_db()
        
        # Extract and analyze proteins
        self.protein_analysis = self._analyze_all_proteins()
    
    def _create_vector_db(self):
        """Create FAISS vector database for semantic search"""
        embeddings = self.embedder.encode(self.documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
    
    def _analyze_all_proteins(self) -> Dict:
        """Extract and analyze all proteins from the dataset"""
        print("Extracting proteins from abstracts...")
        
        protein_contexts = defaultdict(list)
        protein_pmids = defaultdict(set)
        
        # Extract proteins from each document
        for idx, (text, pmid) in enumerate(zip(self.documents, self.df['PMID'])):
            proteins = self.extractor.extract_proteins_from_text(text)
            for protein in proteins:
                protein_contexts[protein].append(text)
                protein_pmids[protein].add(pmid)
        
        print(f"Found {len(protein_contexts)} unique proteins")
        
        # Score each protein
        protein_scores = {}
        for protein, contexts in protein_contexts.items():
            frequency = len(contexts)
            scores = self.scorer.score_protein(protein, contexts, frequency)
            scores['frequency'] = frequency
            scores['pmids'] = list(protein_pmids[protein])
            protein_scores[protein] = scores
        
        return protein_scores
    
    def get_top_proteins(self, n=20, min_score=0.1) -> pd.DataFrame:
        """Get top scoring proteins"""
        results = []
        
        for protein, data in self.protein_analysis.items():
            if data['total_score'] >= min_score:
                results.append({
                    'Protein': protein,
                    'Total_Score': round(data['total_score'], 3),
                    'Undercharacterization_Score': round(data['undercharacterization_score'], 3),
                    'Research_Relevance_Score': round(data['research_relevance_score'], 3),
                    'Functionality_Score': round(data['functionality_score'], 3),
                    'Frequency': data['frequency'],
                    'PMIDs_Count': len(data['pmids']),
                    'Literature_Scarcity': 'High' if data['frequency'] <= 3 else 'Medium' if data['frequency'] <= 10 else 'Low'
                })
        
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('Total_Score', ascending=False).head(n)
        
        return df_results
    
    def get_protein_details(self, protein_name: str) -> Dict:
        """Get detailed information about a specific protein"""
        if protein_name not in self.protein_analysis:
            return {"error": "Protein not found"}
        
        data = self.protein_analysis[protein_name]
        
        # Get relevant abstracts
        query_embedding = self.embedder.encode([f"protein {protein_name} glioma"])
        distances, indices = self.index.search(query_embedding, 5)
        relevant_abstracts = [self.documents[i] for i in indices[0]]
        
        return {
            'protein_name': protein_name,
            'scores': data,
            'relevant_abstracts': relevant_abstracts[:3],
            'total_mentions': data['frequency'],
            'pmids': data['pmids'][:10]
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        
        # Prepare data for plotting
        proteins = list(self.protein_analysis.keys())
        total_scores = [self.protein_analysis[p]['total_score'] for p in proteins]
        underchar_scores = [self.protein_analysis[p]['undercharacterization_score'] for p in proteins]
        relevance_scores = [self.protein_analysis[p]['research_relevance_score'] for p in proteins]
        func_scores = [self.protein_analysis[p]['functionality_score'] for p in proteins]
        frequencies = [self.protein_analysis[p]['frequency'] for p in proteins]
        
        # Get top 20 proteins for detailed plots
        top_proteins_df = self.get_top_proteins(n=20, min_score=0.0)
        
        if top_proteins_df.empty:
            return None, None, None, None
        
        # 1. Score Distribution Histogram
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=total_scores, nbinsx=20, name='Total Scores', 
                                   marker_color='lightblue', opacity=0.7))
        fig1.update_layout(
            title='Distribution of Protein Total Scores',
            xaxis_title='Total Score (0-1)',
            yaxis_title='Number of Proteins',
            template='plotly_white'
        )
        
        # 2. Top 20 Proteins Bar Chart
        fig2 = px.bar(
            top_proteins_df.head(20), 
            x='Protein', 
            y='Total_Score',
            color='Literature_Scarcity',
            title='Top 20 Proteins by Total Score',
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        )
        fig2.update_layout(xaxis_tickangle=-45, template='plotly_white')
        
        # 3. Score Components Stacked Bar Chart
        fig3 = go.Figure()
        top_20 = top_proteins_df.head(20)
        
        fig3.add_trace(go.Bar(
            name='Undercharacterization',
            x=top_20['Protein'],
            y=top_20['Undercharacterization_Score'],
            marker_color='red',
            opacity=0.8
        ))
        fig3.add_trace(go.Bar(
            name='Research Relevance', 
            x=top_20['Protein'],
            y=top_20['Research_Relevance_Score'],
            marker_color='blue',
            opacity=0.8
        ))
        fig3.add_trace(go.Bar(
            name='Functionality',
            x=top_20['Protein'], 
            y=top_20['Functionality_Score'],
            marker_color='green',
            opacity=0.8
        ))
        
        fig3.update_layout(
            title='Score Components for Top 20 Proteins',
            xaxis_title='Proteins',
            yaxis_title='Score (0-1)',
            barmode='stack',
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        
        # 4. Scatter Plot: Frequency vs Total Score
        fig4 = px.scatter(
            x=frequencies,
            y=total_scores,
            hover_data=[proteins],
            title='Literature Frequency vs Total Score',
            labels={'x': 'Literature Frequency', 'y': 'Total Score'},
            trendline='ols'
        )
        fig4.update_layout(template='plotly_white')
        
        return fig1, fig2, fig3, fig4
    
    def get_statistics(self):
        """Get system statistics"""
        total_proteins = len(self.protein_analysis)
        high_scoring = len([p for p, d in self.protein_analysis.items() if d['total_score'] >= 0.5])
        low_literature = len([p for p, d in self.protein_analysis.items() if d['frequency'] <= 3])
        
        freq_stats = [d['frequency'] for d in self.protein_analysis.values()]
        score_stats = [d['total_score'] for d in self.protein_analysis.values()]
        
        return {
            'total_proteins': total_proteins,
            'high_scoring': high_scoring,
            'low_literature': low_literature,
            'avg_frequency': np.mean(freq_stats),
            'avg_score': np.mean(score_stats),
            'max_score': np.max(score_stats),
            'min_score': np.min(score_stats)
        }

# Initialize system
print("Initializing Glioma Protein Analyzer...")
analyzer = GliomaProteinAnalyzer("abstracts.xlsx")

# Gradio Interface Functions
def get_top_proteins_interface(num_proteins, min_score):
    df = analyzer.get_top_proteins(n=int(num_proteins), min_score=float(min_score))
    return df

def get_protein_details_interface(protein_name):
    details = analyzer.get_protein_details(protein_name)
    if "error" in details:
        return details["error"], "", ""
    
    score_info = f"""
    Protein: {details['protein_name']}
    Total Score: {details['scores']['total_score']:.3f}
    - Undercharacterization Score: {details['scores']['undercharacterization_score']:.3f}/1.0
    - Research Relevance Score: {details['scores']['research_relevance_score']:.3f}/1.0  
    - Functionality Score: {details['scores']['functionality_score']:.3f}/1.0
    
    Literature Frequency: {details['total_mentions']} abstracts
    Literature Scarcity Level: {'High' if details['total_mentions'] <= 3 else 'Medium' if details['total_mentions'] <= 10 else 'Low'}
    Number of PMIDs: {len(details['pmids'])}
    
    Evidence Summary:
    - Undercharacterization: {', '.join(details['scores']['evidence']['undercharacterization']) or 'None'}
    - Research Relevance: {', '.join(details['scores']['evidence']['research_relevance']) or 'None'}
    - Functionality: {', '.join(details['scores']['evidence']['functionality']) or 'None'}
    """
    
    abstracts_text = "\n\n".join([f"Abstract {i+1}:\n{abs[:500]}..." 
                                 for i, abs in enumerate(details['relevant_abstracts'])])
    
    pmids_text = "Associated PMIDs: " + ", ".join(map(str, details['pmids']))
    
    return score_info, abstracts_text, pmids_text

def create_visualizations_interface():
    fig1, fig2, fig3, fig4 = analyzer.create_visualizations()
    return fig1, fig2, fig3, fig4

# Create Gradio interface
with gr.Blocks(title="Glioma Protein Analysis System", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🧬 Glioma Protein Analysis System")
    gr.Markdown("Analyze proteins from glioma research abstracts with literature scarcity-based scoring (0-1 scale)")
    
    with gr.Tab("🏆 Top Proteins"):
        gr.Markdown("## Top Scoring Proteins (Less Literature = Higher Score)")
        with gr.Row():
            num_proteins = gr.Number(value=20, label="Number of proteins to show")
            min_score = gr.Number(value=0.1, label="Minimum total score (0-1)", step=0.1)
        
        get_top_btn = gr.Button("Get Top Proteins", variant="primary")
        top_proteins_output = gr.Dataframe()
        
        get_top_btn.click(
            get_top_proteins_interface,
            inputs=[num_proteins, min_score],
            outputs=top_proteins_output
        )
    
    with gr.Tab("🔍 Protein Details"):
        gr.Markdown("## Detailed Protein Information")
        protein_input = gr.Textbox(label="Enter protein name (e.g., TP53, EGFR, PTEN)", 
                                  placeholder="Enter any protein name")
        get_details_btn = gr.Button("Get Protein Details", variant="primary")
        
        with gr.Row():
            with gr.Column():
                protein_scores = gr.Textbox(label="Protein Scores & Evidence", lines=15)
            with gr.Column():
                relevant_abstracts = gr.Textbox(label="Relevant Abstracts", lines=15)
        
        pmids_output = gr.Textbox(label="Associated PMIDs", lines=3)
        
        get_details_btn.click(
            get_protein_details_interface,
            inputs=protein_input,
            outputs=[protein_scores, relevant_abstracts, pmids_output]
        )
    
    with gr.Tab("📊 Visualizations"):
        gr.Markdown("## Protein Analysis Visualizations")
        
        create_viz_btn = gr.Button("Generate Visualizations", variant="primary")
        
        with gr.Row():
            with gr.Column():
                plot1 = gr.Plot(label="Score Distribution")
                plot3 = gr.Plot(label="Score Components")
            with gr.Column():
                plot2 = gr.Plot(label="Top 20 Proteins")
                plot4 = gr.Plot(label="Frequency vs Score")
        
        create_viz_btn.click(
            create_visualizations_interface,
            outputs=[plot1, plot2, plot3, plot4]
        )
    
    with gr.Tab("📈 Statistics"):
        stats = analyzer.get_statistics()
        
        gr.Markdown(f"""
        ## 📊 Analysis Statistics
        
        ### Overall Metrics
        - **Total Unique Proteins Found:** {stats['total_proteins']:,}
        - **High-Scoring Proteins (≥0.5):** {stats['high_scoring']:,}
        - **Low Literature Proteins (≤3 abstracts):** {stats['low_literature']:,}
        - **Total Abstracts Analyzed:** {len(analyzer.documents):,}
        
        ### Score Statistics
        - **Average Total Score:** {stats['avg_score']:.3f}
        - **Maximum Score:** {stats['max_score']:.3f}  
        - **Minimum Score:** {stats['min_score']:.3f}
        - **Average Literature Frequency:** {stats['avg_frequency']:.1f}
        
        ### 🎯 Scoring System (0-1 Scale)
        
        **1. Undercharacterization Score (Weight: 50%)**
        - Based on inverse literature frequency: fewer papers = higher score
        - Boosted by keywords: "uncharacterized", "unknown function", "poorly understood"
        - Formula: 1.0 - log₁₀(frequency+1)/log₁₀(total_abstracts+1) + keyword_boost
        
        **2. Research Relevance Score (Weight: 25%)**  
        - Keywords: "novel", "newly identified", "recently discovered", "emerging"
        - Each keyword adds 0.15 points (max 1.0)
        
        **3. Functionality Score (Weight: 25%)**
        - Keywords: "kinase", "receptor", "transcription factor", functional terms
        - Additional context scoring for glioma-related research terms
        - Each keyword adds 0.1 points (max 1.0)
        
        ### 💡 Interpretation Guide
        - **Score 0.7-1.0:** Highly promising, undercharacterized targets
        - **Score 0.5-0.7:** Good candidates for further research  
        - **Score 0.3-0.5:** Moderately interesting proteins
        - **Score 0.0-0.3:** Well-characterized or less relevant
        
        **Literature Scarcity Levels:**
        - 🔴 **High:** ≤3 abstracts (potentially underexplored)
        - 🟡 **Medium:** 4-10 abstracts (moderate coverage)
        - 🟢 **Low:** >10 abstracts (well-studied)
        """)

# Run the application
if __name__ == "__main__":
    interface.launch(share=True)