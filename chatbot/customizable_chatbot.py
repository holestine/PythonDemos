from openai import OpenAI
from keys import openai_key

PROMPT = "You are to act as a financial fraud detection expert. Use the following data for additional context \
to help answer questions. Ask for more information if needed. If you don't know the answer, say that you don't know. \
Keep answers concise using a maximum of three sentences."

class ChatBot:

    def __init__(self, prompt=PROMPT, additional_data=None) -> None:
        '''
        Initialize the chatbot
        prompt: 
        data_file: A file containing additional data to use in reasoning.
        '''
        
        # Get Open AI client
        self.__client = OpenAI(api_key=openai_key)

        # Get the data for the LLM
        if additional_data is not None:
            with open(additional_data, 'r', encoding="utf-8") as f:
                data = ''.join(line for line in f)
        else:
            data = ''

        # Prompt to initialize LLM
        self.system_prompt = {'role':'system', 'content':f"{prompt} \n {data}"}
        
        self.messages = [self.system_prompt]

    def get_response(self, prompt, model="chatgpt-4o-latest", temperature=0):
        '''
        Get a response based on the current history
        '''

        self.messages.append({'role':'user', 'content':f"{prompt}"})

        response = self.__client.chat.completions.create(model=model, messages=self.messages, temperature=temperature)
        response = response.choices[0].message.content

        self.messages.append({'role':'assistant', 'content':f"{response}"})
        
        return response

if __name__ == "__main__":
    
    #chatbot = ChatBot()
    chatbot = ChatBot(additional_data='data.txt')

    human_prompt = "What are the best ways to detect financial fraud?"
    while human_prompt != 'goodbye':
        response = chatbot.get_response(human_prompt)
        human_prompt = input(f"\n{response}\n\n")
