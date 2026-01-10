import asyncio
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, AsyncGenerator
import httpx


class AsyncExternalLLM:
    """
    Async class to handle OpenRouter LLM requests with streaming.
    Suitable for real-time WebSocket chat.
    """

    def __init__(self, api_key: str = None, model: str = "google/gemini-flash-1.5-exp:free"):
        # Load variables from .env into the environment
        load_dotenv()

        # Access the key using os.getenv
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENROUTER_API_KEY")
        self.model = model
        self.endpoint = "https://openrouter.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Required for some providers
            "X-Title": "Local Test Script"  # Helps with routing
        }

    async def stream_chat(self, messages: List[Dict], max_tokens: int = 500) -> AsyncGenerator[str, None]:
        """
        Sends a chat request to OpenRouter with streaming enabled.
        Yields content tokens asynchronously.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
            # "max_tokens": max_tokens
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.endpoint, headers=self.headers, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith("event: ping"):
                        continue
                    if line.startswith("data: "):
                        line_content = line[len("data: "):]
                        if line_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line_content)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue


async def llm_stream():
    # 1. Initialize the class
    # Make sure OPENROUTER_API_KEY is in your environment or pass it here
    llm = AsyncExternalLLM()



    # 2. Define a simple prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short 2-sentence poem about coding."}
    ]

    print(f"--- Starting Stream (Model: {llm.model}) ---")

    # 3. Iterate through the async generator
    full_response = []
    try:
        async for token in llm.stream_chat(messages):
            print(token, end="", flush=True)
            full_response.append(token)
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")

    print("\n\n--- Stream Complete ---")
    print(f"Total tokens received: {len(full_response)}")


import httpx
import asyncio
import json
import os


async def direct_debug_test():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = "google/gemma-3n-e4b-it:free"

    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Local Debug Script"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a short 4-line poem about coding."}],
        "stream": True
    }

    print(f"--- 📡 Connecting to OpenRouter ({model}) ---")

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                print(f"STATUS CODE: {response.status_code}")

                if response.status_code != 200:
                    error_body = await response.aread()
                    print(f"❌ SERVER ERROR: {error_body.decode()}")
                    return

                print("--- 📥 Stream Started ---")
                print("\n✨ POEM START:\n" + "-" * 20)

                async for line in response.aiter_lines():
                    # 1. Skip empty lines or keep-alive pings (starting with :)
                    if not line or line.startswith(":"):
                        continue

                    # 2. Process data lines
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]

                        # Check for the end of the stream
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            # Extract content from the delta object
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                # Print content in GREEN (\033[92m) then reset (\033[0m)
                                print(f"\033[92m{content}\033[0m", end="", flush=True)

                        except json.JSONDecodeError:
                            # Occasionally OpenRouter sends partial JSON or malformed lines
                            continue
                        except Exception as e:
                            print(f"\n[Unexpected Error: {e}]")

                print("\n" + "-" * 20 + "\n--- ✅ Stream Finished ---")

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(direct_debug_test())