import pandas as pd
from llama_cpp import Llama


class MyModel:
    def __init__(self) -> None:
        # Load the model with CUDA support
        self.llm = Llama.from_pretrained(
            repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="meta-llama-3.1-8b-instruct.f16.gguf",
            n_gpu_layers=40,  # Load layers to the GPU
            use_mmap=True,  # Use memory-mapped files to reduce RAM usage
            use_mlock=True,  # Lock model weights in memory to prevent swapping
        )

    def model_work(self, regulation_text, use_case_text):
        # Define the types to use
        types = [
            "The developed system is not subject to certification. Verification is not required.",
            "The case mentions certifiable objects, and the regulations are complied with. It is necessary to specify the regulations that this case affects in the development.",
            "The case mentions certifiable objects that are subject to certification restrictions, but they are not described in the case. The case needs to be supplemented with a description of the restrictions from the regulations.",
            "The case mentions certifiable objects, and the requirement for development contradicts (does not comply with) the certification regulations. Corrections need to be made.",
        ]

        # Create the prompt
        prompt = f"""
        The use case is like a hypothesis which has to be correct. To check if the use case is correct you have to read through the regulation text.

        Regulation text: {regulation_text}
        Use case text: {use_case_text}

        Try to determine the type. And follow its instructions.
        Types:
        0. {types[0]}
        1. {types[1]}
        2. {types[2]}
        3. {types[3]}
        Answer briefly.
        """

        print("Запрос отправлен модели")
        # Generate the completion
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,  # Adjust this value as necessary
        )

        # Extract the model's response
        answer = response["choices"][0]["message"]["content"]

        return answer

    def process_file(self, df):
        # Initialize a list to hold processed data
        processed_data = []

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            use_case_text = row["usecase_text"]
            certifiable_object = row["certifiable_object"]
            regulation_summary = row["regulation_summary"]

            # Call model_work() to get the model's response for this row
            model_response = self.model_work(regulation_summary, use_case_text)

            # Append the row data including the model response
            processed_data.append(
                {
                    "usecase_text": use_case_text,  # Текст юзкейса
                    "certifiable_object": certifiable_object,  # Сертифицируемый объект
                    "regulation_summary": regulation_summary,  # Выжимка из регламента
                    "model_response": model_response,  # Ответ модели
                }
            )

        # Create a new DataFrame from the processed data
        df_result = pd.DataFrame(processed_data)

        # Define the output file path
        output_path = "uploads/model_data.xlsx"

        # Save the DataFrame to an Excel file
        df_result.to_excel(output_path, index=False)

        # Return the path to the saved file
        return output_path
