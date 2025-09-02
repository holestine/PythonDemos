from openai import OpenAI
from keys import openai_key # you'll need create the file keys.py and put "openai_key = 'your key'" inside
from extract_training_data import DataScraper, write_data

# Prompt Pairs

FRAUD_PROMPT = {
    'system': "You are to act as a financial fraud detection expert. Use the following data for additional context \
to help answer questions. Ask for more information if needed. If you don't know the answer, say that you don't know. \
Keep answers concise using a maximum of three sentences.",
    'human': "What are the best ways to detect financial fraud?"
}

CONTRACT_PROMPT = {
    'system': "You are to analyze the following contract and answer questions with brief responses. Ask questions if you need additional information.",
    'human': "When does this contract begin?"
}

SUMMARIZE_PROMPT = {
    'system': "You are to analyze the following documents and summarize the content and make a conclusion. You should identify any important data. Keep responses concise using a maximum of three sentences.",
    'human': "What's the summary?"
}


class ChatBot:

    def __init__(self, llm_init_params=None) -> None:
        '''
        Initialize the chatbot
        prompt: 
        data_file: A file containing additional data to use in reasoning.
        '''
        
        # Get Open AI client
        self.__client = OpenAI(api_key=openai_key)

        # Get the data for the LLM
        if llm_init_params is not None:
            # Read data file
            with open(llm_init_params['additional_data_file'], 'r', encoding="utf-8") as f:
                data = ''.join(line for line in f)

            # Create message to fine tune LLM
            self.system_prompt = {'role':'system', 'content':f"{llm_init_params['system_prompt']} \n {data}"}
            self.messages = [self.system_prompt]
        else:
            self.messages = []

    def get_response(self, prompt, model="chatgpt-4o-latest", temperature=0):
        """ Get a response based on the current conversation history

        Args:
            prompt (string): The human query.
            model (str, optional): The OpenAI model to use (gpt-5 etc.). Defaults to "chatgpt-4o-latest".
            temperature (int, optional): Amount of randomness in answer. Defaults to 0.

        Returns:
            object: The response.
        """

        #  gives quicker responses

        self.messages.append({'role':'user', 'content':f"{prompt}"})

        response = self.__client.chat.completions.create(model=model, messages=self.messages, temperature=temperature)
        response = response.choices[0].message.content

        self.messages.append({'role':'assistant', 'content':f"{response}"})

        return response

if __name__ == "__main__":

    prompt = CONTRACT_PROMPT
    data_file = "data/contract.txt"
    pdf_dir='Contract'

    #prompt = SUMMARIZE_PROMPT
    #data_file = "data/pdf_data.txt"
    #pdf_dir = "PDFs"

    ds = DataScraper()
    pdf_data = ds.extract_pdf_text(pdf_dir=pdf_dir)
    write_data(pdf_data, data_file=data_file)

    llm_init_params = {
        "system_prompt": prompt['system'], 
        "additional_data_file": data_file
        }
    
    chatbot = ChatBot(llm_init_params)

    human_prompt = prompt['human']
    while human_prompt != 'goodbye':
        response = chatbot.get_response(human_prompt)
        human_prompt = input(f"\n{response}\n\n")
