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
from functools import lru_cache

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

    def __read_txt(self, full_path):
        # Implementation for reading .txt files
        with open(full_path, "r", encoding="utf-8") as file:
            return file.read()

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

        # Ensure inputs are on the same device as the model
        input_ids = input_ids.to(self.model.device)

        summary_ids = self.model.generate(
            input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,  # Set this to an appropriate start token if needed
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def get_summarized_data(self, paths_list):
        """
        Processes a list of .docx and .txt file paths, summarizes their content, and returns a DataFrame.
        """
        data = []
        for path in paths_list:
            if path.endswith(".docx"):
                text = self.__read_docx(path)
            elif path.endswith(".txt"):
                text = self.__read_txt(path)
            else:
                continue  # Skip files that are not .docx or .txt

            summary = self.__summarize_one_text(text)
            data.append([text, summary])

        df = pd.DataFrame(data, columns=["text", "summary"])
        return df


class PreprocessRegulations:
    def __init__(self, path="test_data/Регламенты сертификации") -> None:
        """
        Initializes the PreprocessRegulations class with the specified directory path.
        """
        self.path = path
        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    @lru_cache(maxsize=128)
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

    @lru_cache(maxsize=128)
    def __prepare_regulations_df(self, path):
        """
        Walks through the specified directory and processes all PDF files.
        """
        df_prev = pd.DataFrame()
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    df = self.__read_pdf(file_path)
                    subdirectory_name = os.path.basename(root)
                    df.insert(0, "Subdirectory", subdirectory_name)
                    df_prev = pd.concat([df_prev, df], ignore_index=True)
        return df_prev

    @lru_cache(maxsize=128)
    def get_embedding(self, text):
        """
        Generates an embedding for the given text using BERT and leverages GPU (CUDA) if available.
        """
        # Ensure the model is on the correct device
        self.model.to(self.device)

        # Tokenize the input text and move tensors to the same device
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(
            self.device
        )  # Move inputs to the same device as the model

        # Disable gradient calculation for efficiency
        with torch.no_grad():
            # Forward pass to get model outputs
            outputs = self.model(**inputs)

            # Extract the [CLS] token's embedding (first token of the sequence)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Move the embeddings back to the CPU and convert to numpy
            return cls_embedding.cpu().numpy()

    @lru_cache(maxsize=128)
    def get_embeddings_df(self):
        """
        Generates embeddings for different sections of the regulations using BERT.
        """
        text_preprocessor = TextPreprocessor()
        df = self.__prepare_regulations_df(self.path)

        # Generate embeddings for each hierarchical section
        for i in ["Глава", "Подглава", "Подпункт", "под-подподпункт"]:
            df[f"emb_{i}"] = (
                df[i]
                .fillna("")
                .apply(lambda x: text_preprocessor.clean_text(x))
                .apply(lambda x: self.get_embedding(x.lower()) if x else None)
            )
        return df


class GetPairs:
    def __init__(self):
        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # Initialize the T5 tokenizer and model for summarization
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

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
        Generates an embedding for the given text using BERT and leverages GPU (CUDA) if available.
        """
        # Ensure the model is on the correct device
        self.model.to(self.device)

        # Tokenize the input text and move tensors to the same device
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(
            self.device
        )  # Move inputs to the same device as the model

        # Disable gradient calculation for efficiency
        with torch.no_grad():
            # Forward pass to get model outputs
            outputs = self.model(**inputs)

            # Extract the [CLS] token's embedding (first token of the sequence)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Move the embeddings back to the CPU and convert to numpy
            return cls_embedding.cpu().numpy()

    def get_two_texts(self, df_usecase, df_regulations):
        """
        Finds the most similar regulation sections for each use case summary and exports the results to Excel.
        """
        text_preprocessor = TextPreprocessor()
        data = []

        def summarize_one_text(input_text):
            """
            Summarizes the input text using the T5 model.
            """
            input_ids = self.tokenizer.encode(
                "summarize: " + input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            )

            # Ensure inputs are on the same device as the model
            input_ids = input_ids.to(self.model_t5.device)

            summary_ids = self.model_t5.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                decoder_start_token_id=self.tokenizer.pad_token_id,  # Set this to an appropriate start token if needed
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        # df_regulations = df_regulations.dropna(subset="Подпункт").copy()

        # Initialize similarity keys for best_match
        similarity_keys = ["Глава", "Подглава", "Подпункт", "под-подподпункт"]
        for idx, target in enumerate(df_usecase["summary"].values):
            cleaned_target = text_preprocessor.clean_text(target)
            target_embedding = self.get_embedding(cleaned_target.lower())

            best_match = {
                "usecase_text": target,
                "certifiable_object": "",
                "regulation_summary": "",
            }

            # Initialize similarity scores in best_match
            for key in similarity_keys:
                best_match[f"similarity_{key}"] = 0  # Set default similarity to 0

            for i in similarity_keys:
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
                        # Find the matching row
                        match_row = df_regulations.loc[
                            df_regulations[similarity_col] == max_similarity
                        ].iloc[0]
                        best_match["certifiable_object"] = match_row["Subdirectory"]

                        # Объединяем текст, разделяя его точками
                        if best_match["regulation_summary"]:
                            best_match["regulation_summary"] += f". {match_row[i]}"
                        else:
                            best_match["regulation_summary"] = match_row[i]

                        # Store the maximum similarity for the current section
                        best_match[similarity_col] = max_similarity

            data.append(best_match)

        # Create DataFrame from the collected data
        result_df = pd.DataFrame(data)

        # Summarize the regulation_summary column
        result_df["regulation_summary"] = result_df["regulation_summary"].apply(
            lambda x: summarize_one_text(x) if x else ""
        )

        return result_df
