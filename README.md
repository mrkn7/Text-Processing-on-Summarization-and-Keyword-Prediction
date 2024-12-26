# Text-Processing-on-Summarization-and-Keyword-Prediction

# Text Processing Project: Summarization, Keyword Prediction, and Classification

This repository contains a modular implementation of text processing techniques for academic abstracts. The project focuses on the following tasks:

1. **Text Summarization**: Extracting key topics from abstracts using LDA (Latent Dirichlet Allocation).
2. **Keyword Prediction**: Identifying significant keywords from the text.
3. **Text Classification**: Classifying abstracts into categories using TF-IDF and Naive Bayes.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [License](#license)

## Introduction

Academic abstracts often require summarization, keyword extraction, and classification for various research applications. This project provides a systematic approach to process such texts and achieve the mentioned goals. The implementation is modular, allowing each task to be used independently.

## Project Structure

```
Text-Processing/
│
├── src/                      
│   ├── preprocess.py          
│   ├── lda_model.py           
│   ├── text_classification.py  
├── notebooks/                  
├── README.md                 
├── LICENSE                    
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/mrkn7/text-processing.git
   ```

2. Navigate to the project directory:
   ```bash
   cd text-processing
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

## Usage

### 1. Preprocessing Text
Use the `preprocess.py` script to clean and tokenize text:
```bash
python src/preprocess.py
```

### 2. Extracting Topics with LDA
Run the `lda_model.py` script to train an LDA model and extract topics:
```bash
python src/lda_model.py
```

### 3. Classifying Text
Use the `text_classification.py` script to classify abstracts:
```bash
python src/text_classification.py
```

## Results

- **Text Summarization**: Extracted topics and their associated keywords.
- **Keyword Prediction**: Identified the most relevant terms in each abstract.
- **Text Classification**: Achieved a classification accuracy of approximately 85% on the test dataset.

For detailed outputs, see the `notebooks/` folder.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

