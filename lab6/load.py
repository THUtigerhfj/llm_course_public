from dotenv import load_dotenv
import os
import threading
from openai import OpenAI
import time
import random
from openai import RateLimitError

import os
os.environ['HTTP_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
os.environ['HTTPS_PROXY']="http://Clash:QOAF8Rmd@10.1.0.213:7890"
os.environ['ALL_PROXY']="socks5://Clash:QOAF8Rmd@10.1.0.213:7893"


load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

def get_llm_output(model_name, question_content, question_id, output, semaphore=None):
    try:
        retries = 5
        for attempt in range(retries):
            try:
                if isinstance(question_content, str):
                    messages = [{"role": "user", "content": question_content}]
                else:
                    messages = [{"role": "user", "content": q} for q in question_content]

                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0,
                )

                llm_answer = chat_response.choices[0].message.content.strip()
                output[question_id] = llm_answer
                break  # 成功就退出 retry 循环

            except RateLimitError as e:
                wait_time = random.uniform(1, 3) * (2 ** attempt)
                print(f"RateLimitError on Q{question_id}, retry {attempt+1}/{retries}, wait {wait_time:.1f}s")
                time.sleep(wait_time)

            except Exception as e:
                print(f"Error on Q{question_id}: {e}")
                break 

        else:
            output[question_id] = ""

    finally:
        if semaphore:
            semaphore.release()

def get_llm_output_parallel(model_name, question_contents, max_threads=5):
    # Create threads for each question
    output = {}
    threads = []
    semaphore = threading.Semaphore(max_threads)
    for question_id, question_content in tqdm(enumerate(question_contents)):
        semaphore.acquire() 
        thread = threading.Thread(target=get_llm_output, args=(model_name, question_content, question_id, output, semaphore))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    sorted_keys = sorted(output.keys())
    sorted_outputs = [output[key] for key in sorted_keys]        
    return sorted_outputs

from datasets import load_dataset, load_from_disk
d=load_dataset("nvidia/OpenMathReasoning")  # load from huggingface datasets
d.save_to_disk('/ssdshare/share/data/OpenMathReasoning/') # save to local disk
d = load_from_disk('/ssdshare/share/data/OpenMathReasoning/') # load from local disk