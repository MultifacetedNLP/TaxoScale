import json
import requests
import time
import openai
from openai import OpenAI

or_client = openai.OpenAI(
    api_key="#",
    base_url="https://openrouter.ai/api/v1",
)
#OR_MODEL = "openai/gpt-4o-mini"

def open_route_ai_models(prompt, model_name, max_tokens=800, max_retries=10, max_wait=128, wait_time = 2, json_output=False, provider=None, request_timeout=300, key="#"):
   
   for attempt in range(max_retries):
      try:
         
         # Base payload
         payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stop": None,
                "temperature": 0.0,
                "top_p": 1,
                "top_k": 1,
                "seed": 12345,
            }
         
         if provider:
            payload["provider"] = {
               'only': [provider]
            }
         
         # Add JSON response format if requested
         if json_output:
            payload["response_format"] = {"type": "json_object"}
         
         response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
               "Authorization": f"Bearer {key}",
            },
            data=json.dumps(payload),
            timeout=request_timeout,  # <-- NEW: this enforces per-call timeout
         )
         
         # Parse the JSON response
         response_data = response.json()
         # Extract the relevant part of the response
         completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
         
         # Check if response is valid and not empty
         if not completion:
            raise ValueError("Received empty response or no choices.")
            
         # If everything is fine, return the content
         return completion
        
      except Exception as e:
         # If we haven't reached the last retry, wait and then retry
         if attempt < max_retries - 1:
            print(f"Attempt {attempt + 1} failed: {e}\nRetrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, max_wait)
         else:
            # Raise exception if all retries have failed
            raise Exception(f"Error after {max_retries} attempts: {e}")
         
def openai_chatgpt_models(prompt, model_name, max_retries=3, retry_delay=5):

   last_error = None
   for attempt in range(1, max_retries + 1):
      try:
            # response = client.chat.completions.create(
            #    model=model_name,
            #    messages=[{"role": "user", "content": prompt.strip()}],
            #    max_tokens=2000,
            #    temperature=0.0,
            #    top_p=1.0,
            #    presence_penalty=0,
            #    frequency_penalty=0,
            #    seed=12345
            # )
            # return response.choices[0].message.content
            response = or_client.chat.completions.create(
               model=model_name,
               messages=[{"role": "user", "content": prompt.strip()}],
               max_tokens=2000,
               temperature=0.0,
               top_p=1.0,
               presence_penalty=0,
               frequency_penalty=0,
               seed=12345
            )
            return response.choices[0].message.content
      except Exception as e:
            last_error = e
            if attempt < max_retries:
               time.sleep(retry_delay)
   raise Exception(f"Error while calling OpenAI API after {max_retries} attempts: {last_error}")