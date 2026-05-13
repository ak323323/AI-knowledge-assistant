import requests


class OllamaClient:
    def __init__(self, model: str = "llama3"):
        """
        Initialize Ollama client.

        Args:
            model (str): Name of the local Ollama model (e.g., llama3)
        """
        self.model = model
        self.url = "http://localhost:11434/api/generate"  # Ollama local API endpoint

    def generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama and get generated response.

        Args:
            prompt (str): Final prompt string (includes question + context)

        Returns:
            str: Generated response from LLM
        """

        try:
            #  Make POST request to Ollama API
            response = requests.post(
                self.url,
                json={
                    "model": self.model,   # model name (llama3)
                    "prompt": prompt,     # full prompt from RAG pipeline
                    "stream": False       # disable streaming (simple response)
                },
                timeout=60  # avoid hanging forever
            )

            #  Raise error if HTTP request failed (4xx/5xx)
            response.raise_for_status()

            #  Parse JSON response
            data = response.json()

            #  Extract generated text safely
            result = data.get("response", "").strip()

            #  Debug logs (safe, no undefined variables)
            print(f"[LLM] Prompt length: {len(prompt)} chars")
            print(f"[LLM] Response length: {len(result)} chars")

            return result if result else "No response generated"

        except requests.exceptions.Timeout:
            print("[OLLAMA ERROR] Request timed out")
            return "LLM timeout error"

        except requests.exceptions.RequestException as e:
            # Handles network errors, connection issues, etc.
            print("[OLLAMA ERROR] Request failed:", str(e))
            return "LLM request failed"

        except Exception as e:
            # Catch any unexpected errors
            print("[OLLAMA ERROR] Unexpected error:", str(e))
            return "LLM failed to generate response"