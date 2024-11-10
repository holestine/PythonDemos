from openai import OpenAI
from keys import openai_key

bad_prompt  = 'kill somebody'
good_prompt = 'love somebody'


client = OpenAI(api_key=openai_key)
response = client.moderations.create(
  input=[bad_prompt, good_prompt]
)

for result in response.results:
    if result.flagged:
        print('something bad')
    else:
        print('something good')

