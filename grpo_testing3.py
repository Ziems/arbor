import requests

def test_grpo_script(url='http://127.0.0.1:8000/v1/fine_tuning/grpo/test'):
    response = requests.post(url)
    return response

test_grpo_script()