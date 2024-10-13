import os
import pandas as pd
import spacy
import nltk
import re
from nltk.corpus import stopwords
from docx import Document
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import pdfplumber

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")


class TextPreprocessor:
    def __init__(self):
        # Load the English model for spaCy
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize stop words
        self.stop_words = set(stopwords.words("english"))

    def tokenize(self, text):
        """
        Tokenizes the input text after removing non-word characters.
        """
        cleaned_text = re.sub(r"[^\w\s]", "", text)
        doc = self.nlp(cleaned_text)
        tokens = [token.text for token in doc]
        return tokens

    def clean_text(self, text):
        """
        Cleans the text by tokenizing, removing stop words, and lowercasing.
        """
        cleaned_tokens = self.tokenize(text)
        filtered_tokens = [
            word.lower()
            for word in cleaned_tokens
            if word.lower() not in self.stop_words
        ]
        return " ".join(filtered_tokens)


class PreprocessUseCases:
    def __init__(self) -> None:
        # Initialize the T5 tokenizer and model for summarization
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def __read_docx(self, file_path):
        """
        Reads a .docx file and extracts its full text.
        """
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)

    def __summarize_one_text(self, input_text):
        """
        Summarizes the input text using the T5 model.
        """
        input_ids = self.tokenizer.encode(
            "summarize: " + input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        summary_ids = self.model.generate(
            input_ids, max_length=150, num_beams=4, early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def get_summarized_data(self, paths_list, uploads_dir="uploads"):
        """
        Processes a list of .docx file paths, summarizes their content, and returns a DataFrame.
        """
        data = []
        for path in paths_list:
            if path.endswith(".docx"):
                full_path = os.path.join(uploads_dir, path)
                text = self.__read_docx(full_path)
                summary = self.__summarize_one_text(text)
                data.append([full_path, text, summary])
        df = pd.DataFrame(data, columns=["path_of_file", "text", "summary"])
        return df


class PreprocessRegulations:
    def __init__(self, path="test_data/Регламенты сертификации") -> None:
        """
        Initializes the PreprocessRegulations class with the specified directory path.
        """
        self.path = path
        model_name = (
            "bert-base-uncased"  # You can choose other models, e.g., 'bert-base-cased'
        )
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def __read_pdf(self, path):
        """
        Reads a PDF file and extracts its text, splitting it into sections based on a regex pattern.
        """
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # Split text into sections assuming sections start with '5.' followed by numbers
        sections = re.split(r"(\n5(?:\.\d+)+\.)", text)
        sections = sections[
            1:
        ]  # Remove the first split part which is before the first match

        df = {
            "punkt": [sections[i].strip() for i in range(0, len(sections), 2)],
            "text": [
                sections[i + 1].strip()
                for i in range(0, len(sections), 2)
                if i + 1 < len(sections)
            ],
        }
        dt = pd.DataFrame(df)

        # Ensure 'punkt' is a string before applying .str
        dt["punkt"] = dt["punkt"].astype(str)

        # Extract different levels of sections with expand=False
        dt["section1"] = dt["punkt"].str.extract(r"(5\.\d)", expand=False)
        dt["section2"] = dt["punkt"].str.extract(r"(5\.\d+\.\d+)", expand=False)
        dt["section3"] = dt["punkt"].str.extract(r"(5\.\d+\.\d+\.\d+)", expand=False)
        dt["section4"] = dt["punkt"].str.extract(
            r"(5\.\d+\.\d+\.\d+\.\d+)", expand=False
        )

        dt = dt.fillna("-1")

        # Define helper functions to get main texts based on sections
        def get_main_text(row, df):
            parts = row["punkt"].split(".")
            if len(parts) >= 2:
                main_section = f"{parts[0]}.{parts[1]}"  # e.g., '5.1'
                match = df.loc[df["section1"] == main_section, "text"]

                # Return the first match, or None if no match is found
                result = match.iloc[0] if not match.empty else None
                return result
            return None

        def get_main_text1(row, df):
            parts = row["punkt"].split(".")
            if len(parts) >= 3:
                main_section = f"{parts[0]}.{parts[1]}.{parts[2]}"  # e.g., '5.1.1'
                match = df.loc[df["section2"] == main_section, "text"]
                return match.values[0] if not match.empty else None
            return None

        def get_main_text2(row, df):
            parts = row["punkt"].split(".")
            if len(parts) >= 4:
                main_section = (
                    f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"  # e.g., '5.1.1.1'
                )
                match = df.loc[df["section3"] == main_section, "text"]
                return match.values[0] if not match.empty else None
            return None

        def get_main_text_5(row, df):
            parts = row["punkt"].split(".")
            if len(parts) >= 5:
                main_section = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"  # e.g., '5.1.1.1.1'
                match = df.loc[df["section4"] == main_section, "text"]
                return match.values[0] if not match.empty else None
            return None

        dt["Глава"] = dt.apply(lambda row: get_main_text(row, dt), axis=1)
        dt["Подглава"] = dt.apply(lambda row: get_main_text1(row, dt), axis=1)
        dt["Подпункт"] = dt.apply(lambda row: get_main_text2(row, dt), axis=1)
        dt["под-подподпункт"] = dt.apply(lambda row: get_main_text_5(row, dt), axis=1)

        cl = dt.drop(["section1", "section2", "section3", "section4"], axis=1)
        return cl

    def __prepare_regulations_df(self, path):
        """
        Walks through the specified directory and processes all PDF files.
        """
        all_data = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    df = self.__read_pdf(file_path)
                    subdirectory_name = os.path.basename(root)
                    df.insert(0, "Subdirectory", subdirectory_name)
                    all_data.append(df)
        if all_data:
            regulations_df = pd.concat(all_data, ignore_index=True)
            return regulations_df
        else:
            return pd.DataFrame()

    def get_embeddings_df(self):
        """
        Generates embeddings for different sections of the regulations using BERT.
        """
        text_preprocessor = TextPreprocessor()

        def get_embedding(text):
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token embedding
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        df = self.__prepare_regulations_df(self.path)

        # Generate embeddings for each hierarchical section
        for i in ["Глава", "Подглава", "Подпункт", "под-подподпункт"]:
            df[f"emb_{i}"] = (
                df[i]
                .fillna("")
                .apply(lambda x: text_preprocessor.clean_text(x))
                .apply(lambda x: get_embedding(x.lower()) if x else None)
            )
        return df


class GetPairs:
    def __init__(self):
        # Initialize the BERT tokenizer and model (ensure consistency with PreprocessRegulations)
        model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def calculate_cosine_similarity(self, embedding, target_embedding_2d):
        """
        Calculates cosine similarity between two embeddings.
        """
        if embedding is not None and target_embedding_2d is not None:
            return cosine_similarity(embedding, target_embedding_2d).flatten()[0]
        else:
            return 0.0

    def get_embedding(self, text):
        """
        Generates an embedding for the given text using BERT.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def get_two_texts(self, df_usecase, df_regulations):
        """
        Finds the most similar regulation sections for each use case summary and exports the results to Excel.
        """
        text_preprocessor = TextPreprocessor()
        data = []

        for idx, target in enumerate(df_usecase["summary"].values):
            cleaned_target = text_preprocessor.clean_text(target)
            target_embedding = self.get_embedding(cleaned_target.lower())

            best_match = {
                "usecase_text": target,
                "certifiable_object": "",
                "regulation_summary": "",
            }

            for i in ["Глава", "Подглава", "Подпункт", "под-подподпункт"]:
                similarity_col = f"similarity_{i}"
                emb_col = f"emb_{i}"
                if emb_col in df_regulations.columns:
                    df_regulations[similarity_col] = df_regulations[emb_col].apply(
                        lambda x: self.calculate_cosine_similarity(x, target_embedding)
                    )
                    max_similarity = df_regulations[similarity_col].max()
                    if (
                        pd.notnull(max_similarity)
                        and max_similarity > best_match[similarity_col]
                    ):
                        # best_match[f"similarity_{i}"] = max_similarity
                        match_row = df_regulations.loc[
                            df_regulations[similarity_col] == max_similarity
                        ].iloc[0]
                        best_match["certifiable_object"] = match_row["Subdirectory"]

                        # Объединяем текст, разделяя его точками
                        if best_match["regulation_summary"]:
                            best_match["regulation_summary"] += f". {match_row[i]}"
                        else:
                            best_match["regulation_summary"] = match_row[i]

            data.append(best_match)

        # Create DataFrame from the collected data
        result_df = pd.DataFrame(data)
        # Export the results to an Excel file
        output_path = "results/model_data.xlsx"
        result_df.to_excel(output_path, index=False)
        return output_path
