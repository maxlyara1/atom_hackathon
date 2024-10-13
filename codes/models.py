import pandas as pd
from llama_cpp import Llama


class MyModel:
    def __init__(self) -> None:
        self.llm = Llama.from_pretrained(
            repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="meta-llama-3.1-8b-instruct.f16.gguf",
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

        # Generate the completion
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,  # Adjust this value as necessary
        )

        # Print the response
        answer = response["choices"][0]["message"]["content"]

        return answer  # Return the BytesIO object

    def process_files(self, df):
        pass
