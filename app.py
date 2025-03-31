import streamlit as st
import torch
import numpy as np
import pandas as pd
import PyPDF2
import re
import docx2txt
import io
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class ResumeMatcherBERT:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.max_length = 512
        self.chunk_overlap = 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.section_patterns = {
            'skills': [
                r'skills', r'technical skills', r'core skills', r'key skills',
                r'competencies', r'technologies', r'tech stack'
            ],
            'experience': [
                r'experience', r'work experience', r'professional experience',
                r'employment history', r'work history', r'career history'
            ],
            'projects': [
                r'projects', r'project experience', r'key projects',
                r'professional projects', r'personal projects'
            ],
            'summary': [
                r'summary', r'professional summary', r'career summary',
                r'profile', r'professional profile', r'career profile'
            ],
            'objective': [
                r'objective', r'career objective', r'professional objective',
                r'career goal', r'professional goal'
            ]
        }

    def extract_text_from_file(self, file) -> str:
        """Extract text from different file formats including PDF, DOCX, and TXT."""
        try:
            file_name = file.name.lower()
            file_content = file.read()
            file.seek(0)

            if file_name.endswith('.pdf'):
                return self._extract_pdf_text(io.BytesIO(file_content))
            elif file_name.endswith('.docx'):
                return self._extract_docx_text(io.BytesIO(file_content))
            elif file_name.endswith('.txt'):
                return file_content.decode('utf-8')
            else:
                st.error(f"Unsupported file format: {file_name}")
                return ""
        except Exception as e:
            st.error(f"Error extracting text from file: {e}")
            return ""

    def _extract_pdf_text(self, pdf_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def _extract_docx_text(self, docx_file) -> str:
        try:
            text = docx2txt.process(docx_file)
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect resume sections using improved section detection.
        Returns a dictionary of {section_type: section_content}.
        """
        text = text.strip()
        sections = {}
        lines = text.split('\n')
        header_indices = []

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if len(line_lower) < 30 and (line_lower.endswith(':') or line_lower.isupper() or
                                         not line_lower.endswith('.')):
                for section_type, patterns in self.section_patterns.items():
                    for pattern in patterns:
                        if re.search(f"\\b{pattern}\\b", line_lower):
                            header_indices.append((i, section_type))
                            break

        header_indices.sort(key=lambda x: x[0])

        for i, (header_idx, section_type) in enumerate(header_indices):
            start_idx = header_idx + 1
            if i < len(header_indices) - 1:
                end_idx = header_indices[i + 1][0]
            else:
                end_idx = len(lines)

            section_content = '\n'.join(lines[start_idx:end_idx]).strip()
            sections[section_type] = section_content
        if not sections:
            sections['full_text'] = text

        return sections

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with improved section detection and handling.
        """
        text = re.sub(r'[^\w\s\n]', ' ', text)

        sections = self.detect_sections(text)

        prioritized_text = ""

        for section_type in ['summary', 'objective']:
            if section_type in sections:
                prioritized_text += sections[section_type] + " "

        if 'skills' in sections:
            prioritized_text += sections['skills'] + " "

        for section_type in ['experience', 'projects']:
            if section_type in sections:
                prioritized_text += sections[section_type] + " "

        if not prioritized_text.strip() and 'full_text' in sections:
            prioritized_text = sections['full_text']

        preprocessed_text = re.sub(r'\s+', ' ', prioritized_text).strip()

        return preprocessed_text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit within BERT's token limit.
        Uses overlapping chunks to maintain context across chunk boundaries.
        """
        tokens = self.tokenizer.tokenize(text)

        # If text fits within limit, return as single chunk
        if len(tokens) <= self.max_length - 2:  # Account for [CLS] and [SEP] tokens
            return [text]

        effective_chunk_size = self.max_length - 2 - self.chunk_overlap

        chunks = []
        for i in range(0, len(tokens), effective_chunk_size):
            end_idx = min(i + self.max_length - 2, len(tokens))
            chunk_tokens = tokens[i:end_idx]

            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)

            if end_idx == len(tokens):
                break

        return chunks

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for text, handling long text with chunking.
        Returns average embedding across all chunks.
        """
        chunks = self.chunk_text(text)

        chunk_embeddings = []

        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            chunk_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            chunk_embeddings.append(chunk_embedding[0])

        if chunk_embeddings:
            final_embedding = np.mean(chunk_embeddings, axis=0)
            return final_embedding
        else:
            return np.zeros((self.model.config.hidden_size,))

    def calculate_section_similarities(self, job_desc: str, resume: str) -> Dict[str, float]:
        """
        Calculate similarity scores between job description and resume sections.
        Returns a dictionary of {section_type: similarity_score}.
        """
        job_sections = self.detect_sections(job_desc)
        resume_sections = self.detect_sections(resume)
        section_similarities = {}

        for section_type, resume_content in resume_sections.items():
            if not resume_content.strip():
                continue
            resume_section_embedding = self.get_bert_embedding(resume_content)

            best_similarity = 0

            for job_section_type, job_content in job_sections.items():
                if not job_content.strip():
                    continue

                job_section_embedding = self.get_bert_embedding(job_content)

                similarity = cosine_similarity(
                    resume_section_embedding.reshape(1, -1),
                    job_section_embedding.reshape(1, -1)
                )[0][0]

                best_similarity = max(best_similarity, similarity)

            job_desc_embedding = self.get_bert_embedding(job_desc)
            similarity_with_full = cosine_similarity(
                resume_section_embedding.reshape(1, -1),
                job_desc_embedding.reshape(1, -1)
            )[0][0]

            best_similarity = max(best_similarity, similarity_with_full)
            section_similarities[section_type] = best_similarity * 100

        return section_similarities

    def match_resumes(self, job_description: str, resumes: List[str], resume_files: List[Any]) -> List[Dict]:
        """
        Match job description with resumes and calculate similarity scores.
        Returns sorted list of matches with detailed section analysis.
        """
        preprocessed_job_desc = self.preprocess_text(job_description)
        job_desc_embedding = self.get_bert_embedding(preprocessed_job_desc)
        resume_matches = []

        for idx, (resume_text, resume_file) in enumerate(zip(resumes, resume_files)):
            try:
                preprocessed_resume = self.preprocess_text(resume_text)
                resume_embedding = self.get_bert_embedding(preprocessed_resume)

                overall_similarity = cosine_similarity(
                    job_desc_embedding.reshape(1, -1),
                    resume_embedding.reshape(1, -1)
                )[0][0]

                section_similarities = self.calculate_section_similarities(
                    job_description, resume_text
                )

                match_result = {
                    'resume_index': idx,
                    'similarity_score': overall_similarity * 100,
                    'filename': resume_file.name,
                    'section_scores': section_similarities
                }

                resume_matches.append(match_result)

            except Exception as e:
                st.warning(f"Could not process resume {idx + 1}: {e}")

        sorted_matches = sorted(
            resume_matches,
            key=lambda x: x['similarity_score'],
            reverse=True
        )

        return sorted_matches[:10]


def main():
    st.set_page_config(
        page_title="Job and Resume Matcher",
        page_icon=":page_facing_up:",
        layout="wide"
    )

    matcher = ResumeMatcherBERT()

    st.title("Job and Resume Matcher")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Job Description")
        job_description = st.text_area(
            "Paste the complete job description:",
            height=300,
            placeholder="Enter detailed job requirements, skills, responsibilities..."
        )

    with col2:
        st.header("Resume Upload")
        resume_files = st.file_uploader(
            "Upload Multiple Resumes",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload resumes in PDF, DOCX, or TXT format"
        )

    if st.button("Match Resumes", type="primary"):
        if not job_description:
            st.error("Please enter a job description")
        elif not resume_files:
            st.error("Please upload at least one resume")
        else:
            with st.spinner("Processing resumes..."):
                resume_texts = []
                for resume_file in resume_files:
                    resume_text = matcher.extract_text_from_file(resume_file)
                    if resume_text:
                        resume_texts.append(resume_text)

                if resume_texts:
                    try:
                        matches = matcher.match_resumes(job_description, resume_texts, resume_files)
                        st.header("Resume Matching Results")

                        results_df = pd.DataFrame([
                            {
                                'Filename': match['filename'],
                                'Match Percentage': match['similarity_score']
                            }
                            for match in matches
                        ])

                        st.dataframe(
                            results_df,
                            column_config={
                                "Match Percentage": st.column_config.ProgressColumn(
                                    "Match Percentage",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100,
                                )
                            },
                            hide_index=True
                        )

                        st.subheader("Detailed Match Insights")

                        for match in matches:
                            with st.expander(f"{match['filename']} - {match['similarity_score']:.2f}%"):
                                if 'section_scores' in match and match['section_scores']:
                                    section_df = pd.DataFrame([
                                        {'Section': section.title(), 'Match': score}
                                        for section, score in match['section_scores'].items()
                                    ])

                                    section_df = section_df.sort_values('Match', ascending=False)

                                    st.dataframe(
                                        section_df,
                                        column_config={
                                            "Match": st.column_config.ProgressColumn(
                                                "Match",
                                                format="%.2f%%",
                                                min_value=0,
                                                max_value=100,
                                            )
                                        },
                                        hide_index=True
                                    )
                                else:
                                    st.info("No section-specific scores available")

                    except Exception as e:
                        st.error(f"Error in resume matching: {e}")
                else:
                    st.error("Could not extract text from any of the uploaded resumes")


if __name__ == "__main__":
    main()