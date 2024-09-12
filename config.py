API_KEY = "ENTER-YOUR-API-KEY"


def assistant(content: str):
    return { "role": "assistant", "content": content }
def user(content: str):
    return { "role": "user", "content": content }
def system(content: str):
    return { "role": "system", "content": content }
