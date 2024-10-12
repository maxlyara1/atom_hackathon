import pandas as pd
import io

class MyModel:
    def process_text(self, text: str):
        # Reverse the input text
        return text[::-1]

    def process_files(self, file_paths):
        result_file = io.BytesIO()
        pd.DataFrame().to_excel(result_file, index=False)
        result_file.seek(0)  # Move to the beginning of the BytesIO object
        return result_file  # Return the BytesIO object
