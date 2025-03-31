# Advanced Resume Matcher

A powerful application that uses BERT embeddings to match resumes with job descriptions based on semantic similarity. This tool helps recruiters and hiring managers identify the most suitable candidates by analyzing the semantic similarity between job descriptions and resumes.

## Features

- **Semantic Matching**: Uses BERT embeddings to understand the context and meaning of text beyond simple keyword matching
- **Multi-Format Support**: Processes resumes in PDF, DOCX, and TXT formats
- **Detailed Analysis**: Provides section-by-section similarity scores and overall match percentages
- **User-Friendly Interface**: Clean Streamlit interface with progress indicators and expandable sections

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/advanced-resume-matcher.git
   cd advanced-resume-matcher
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install streamlit torch numpy pandas pypdf2 docx2txt transformers scikit-learn
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the application:
   - Paste a job description in the left panel
   - Upload one or more resumes (PDF, DOCX, or TXT) in the right panel
   - Click "Match Resumes" to process and analyze
   - View the results, sorted by match percentage
   - Expand each resume result to see detailed section-by-section matching scores

## How It Works

1. **Text Extraction**: The application extracts text from uploaded resume files.

2. **Section Detection**: It identifies different sections in both the job description and resumes using patterns and formatting cues.

3. **Text Preprocessing**: The text is cleaned and preprocessed to remove irrelevant information.

4. **Chunking**: Longer texts are split into overlapping chunks to handle BERT's token limit.

5. **BERT Embedding**: Each chunk is processed through BERT to generate embeddings that capture semantic meaning.

6. **Similarity Calculation**: The application calculates cosine similarity between job description and resume embeddings.

7. **Section Analysis**: It analyzes each resume section separately to provide detailed matching insights.

8. **Result Ranking**: Results are ranked by overall similarity score and presented in an easy-to-understand format.

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- Transformers
- PyPDF2
- docx2txt
- pandas
- scikit-learn

## Advanced Configuration

The application uses sensible defaults, but you can modify these parameters in the code:

- `max_length`: Maximum token length for BERT (default: 512)
- `chunk_overlap`: Number of overlapping tokens between chunks (default: 100)
- Section patterns in the `section_patterns` dictionary

## Notes for Developers

- The application automatically uses GPU acceleration when available, which significantly improves processing speed.
- For large batches of resumes, consider implementing parallel processing for further optimization.
- The section detection logic can be extended with additional patterns to improve accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This tool uses the HuggingFace Transformers library for BERT implementation
- Built with Streamlit for an interactive user interface