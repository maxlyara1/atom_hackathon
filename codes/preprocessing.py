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
import os
import pdfplumber
import re


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))

    def tokenize(self, text):
        cleaned_text = re.sub(r"[^\w\s]", "", text)
        doc = self.nlp(cleaned_text)
        tokens = [token.text for token in doc]
        return tokens

    def clean_text(self, text):
        cleaned_tokens = self.tokenize(text)
        filtered_tokens = [
            word.lower()
            for word in cleaned_tokens
            if word.lower() not in self.stop_words
        ]
        return " ".join(filtered_tokens)


class PreprocessUseCases:
    def __init__(self) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Function to read .docx file
    def __read_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def __summarize_one_text(self, input_text):
        # Initialize the T5 tokenizer and model
        tokenizer = self.tokenizer
        model = self.model

        # Prepare the input for the T5 model
        input_ids = tokenizer.encode(
            "summarize: " + input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        # Generate the summary
        summary_ids = model.generate(
            input_ids, max_length=150, num_beams=4, early_stopping=True
        )
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    def get_summarized_data(self, paths_list):
        data = []
        for path in paths_list:
            if path.endswith(".docx"):
                full_path = os.path.join("uploads", path)
                # Read the text from the .docx file
                text = self.__read_docx(full_path)
                summary = self.__summarize_one_text(text)
                # Append the path and text to the data list
                data.append([full_path, text, summary])
        df = pd.DataFrame(data, columns=["path_of_file", "text", "summary"])
        return df


class PreprocessRegulations:
    def __init__(self) -> None:
        # нужно обработать все папки и все файлы внутри этой директории
        self.path = "train_data/Регламенты сертификации"
        model_name = "bert-base-uncased"  # можно выбрать другие модели, например, 'bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def __read_pdf(self, path):
        # Открытие PDF файла
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        # Разделение текста по пунктам (предположим, что пункты начинаются с цифры и точки, например "1.")
        sections = re.split(r"(\n5(?:\.\d+)+\.)", text)
        sections = sections[1:]
        df = {
            "punkt": [
                sections[i].replace("\n", "")
                for i in range(len(sections))
                if i % 2 == 0
            ],
            "text": [sections[i] for i in range(len(sections)) if i % 2 == 1],
        }
        dt = pd.DataFrame(df)
        dt["section1"] = dt["punkt"].str.extract(r"(5\.\d)")
        dt["section2"] = dt["punkt"].str.extract(r"(5\.\d+\.\d+)")
        dt["section3"] = dt["punkt"].str.extract(r"(5\.\d+\.\d+\.\d+)")
        dt["section4"] = dt["punkt"].str.extract(r"(5\.\d+\.\d+\.\d+\.\d+)")
        dt = dt.fillna("-1")

        def get_main_text(row, df):
            main_section = (
                row["punkt"].split(".")[0] + "." + row["punkt"].split(".")[1]
            )  # Берем первую часть '5.1' или '5.2'
            match = df.loc[df["section1"] == main_section, "text"]
            return match.values[0] if not match.empty else None

        def get_main_text1(row, df):
            # Формируем main_section для уровня '5.1.1'
            parts = row["punkt"].split(".")
            if len(parts) >= 3:
                main_section = (
                    f"{parts[0]}.{parts[1]}.{parts[2]}"  # Берем первые три части
                )
            else:
                return None  # Возвращаем None, если меньше трех уровней

            # Находим совпадение по section1 и возвращаем текст
            match = df.loc[df["section2"] == main_section, "text"]
            return match.values[0] if not match.empty else None

        def get_main_text2(row, df):
            # Формируем main_section для уровня '5.1.1.1'
            parts = row["punkt"].split(".")
            if len(parts) >= 4:
                main_section = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"  # Берем первые четыре части
            else:
                return None  # Возвращаем None, если меньше четырех уровней

            # Находим совпадение по section2 и возвращаем текст
            match = df.loc[df["section3"] == main_section, "text"]

            return match.values[0] if not match.empty else None

        def get_main_text_5(row, df):
            # Формируем main_section для уровня '5.1.1.1.1'
            parts = row["punkt"].split(".")
            if len(parts) >= 5:
                main_section = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"  # Берем первые пять частей
            else:
                return None  # Возвращаем None, если меньше пяти уровней

            # Находим совпадение по section и возвращаем текст
            match = df.loc[df["section4"] == main_section, "text"]
            return match.values[0] if not match.empty else None

        # Применяем функцию к DataFrame
        dt["Глава"] = dt.apply(lambda row: get_main_text(row, dt), axis=1)
        dt["Подглава"] = dt.apply(lambda row: get_main_text1(row, dt), axis=1)
        dt["Подпункт"] = dt.apply(lambda row: get_main_text2(row, dt), axis=1)
        dt["под-подподпункт"] = dt.apply(lambda row: get_main_text_5(row, dt), axis=1)
        cl = dt.drop(["section1", "section2", "section3", "section4"], axis=1)
        return cl

    def __prepare_regulations_df(self, path):
        all_data = []
        # Проход по всем файлам в директории и подпапках
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    # Чтение PDF файлов и извлечение данных
                    df = self.__read_pdf(file_path)
                    # Получаем имя дочерней папки
                    subdirectory_name = os.path.basename(root)
                    # Добавляем новую колонку с названием папки
                    df.insert(0, "Subdirectory", subdirectory_name)
                    all_data.append(df)

        # Объединение всех данных в один DataFrame
        regulations_df = pd.concat(all_data, ignore_index=True)
        return regulations_df

    def get_embeddings_df(self):
        text_preprocessor = TextPreprocessor()

        def get_embedding(self, text):
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем эмбеддинг токена [CLS], который представляет всё предложение
                return outputs.last_hidden_state[:, 0, :].cpu().numpy()

        dk = self.__prepare_regulations_df(self.path)

        id = dk.index.to_list()

        for i in ["Глава", "Подглава", "подпункт", "под-подпункт"]:

            df = dk.loc[id]  # Используйте loc для работы с индексами

            # Применение функций для вычисления эмбеддингов и схожести
            df["emb"] = (
                df[i]
                .apply(
                    lambda x: text_preprocessor.clean_text(
                        x, text_preprocessor.stop_words
                    )
                )
                .apply(lambda x: get_embedding(x.lower()))
            )

            df["similarity"] = df["emb"].apply(
                lambda x: self.calculate_cosine_similarity(x, target_embedding)
            )

            # Отображение DataFrame
            # Поиск максимального значения схожести
            max_similarity = df["similarity"].max()

            # Обновление списка индексов для следующей итерации
            id = df[df["similarity"] == max_similarity].index.to_list()

        return df


class GetPairs:
    pass
