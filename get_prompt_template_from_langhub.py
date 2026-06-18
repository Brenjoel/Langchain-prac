from langchainhub import Client
import json

client = Client()
prompt = client.pull("hwchase17/react")

print("="*50)
print(prompt)
print("="*50)


prompt_dict = json.loads(prompt)

print(prompt_dict["kwargs"]["template"])