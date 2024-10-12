import os
import pandas as pd
from docx import Document
from transformers import T5ForConditionalGeneration, T5Tokenizer


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

    def __creating_dataframe(self, paths_list):
        data = []
        for path in paths_list:
            if path.endswith(".docx"):
                full_path = os.path.join("uploads", path)
                # Read the text from the .docx file
                text = self.__read_docx(full_path)
                # Append the path and text to the data list
                data.append([full_path, text])
        df = pd.DataFrame(data, columns=["path_of_file", "text"])
        return df

    def get_summurized_data(self, paths_list):
        """
        For each text in the 'text' column, generate a summary and return
        a DataFrame with the original text and its summary.

        :param paths_list: List of file paths to .docx files
        :return: pandas DataFrame with 'path_of_file', 'text', and 'summary' columns
        """
        # Create the initial DataFrame
        df = self.__creating_dataframe(paths_list)

        # Initialize a list to hold summaries
        summaries = []
        # Iterate over each text and generate summary
        for text in df["text"]:
            summary = self.__summarize_one_text(text)
            summaries.append(summary)

        # Add the summaries to the DataFrame
        df["summary"] = summaries

        return df


class PreprocessRegulations:
    def aa(self):
        pass
