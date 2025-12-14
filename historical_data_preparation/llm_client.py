"""
LLM Client abstraction supporting multiple API providers.
Handles both OpenAI-compatible APIs and custom corporate APIs.
"""

import json
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import requests
import urllib3
from openai import OpenAI
from loguru import logger

# Suppress SSL warnings for corporate self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        response_format: Optional[str] = None
    ) -> Optional[str]:
        """
        Make a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            response_format: Optional format ("json_object" for structured output)

        Returns:
            Response content as string, or None on error
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass


class OpenAIClient(BaseLLMClient):
    """Standard OpenAI-compatible client."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            model: Model name to use
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"Initialized OpenAI client (model={model}, base_url={base_url})")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        response_format: Optional[str] = None
    ) -> Optional[str]:
        """Make chat completion request using OpenAI client."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages
            }

            # Try to add temperature, but catch if not supported
            try:
                kwargs["temperature"] = temperature

                # Add response format if specified
                if response_format == "json_object":
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)

                if response and response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content
                else:
                    logger.error(f"Empty or invalid response: {response}")
                    return None

            except Exception as temp_error:
                # If temperature not supported, retry without it
                if "temperature" in str(temp_error).lower() or "unsupported" in str(temp_error).lower():
                    logger.warning(f"Temperature not supported, retrying without it")
                    del kwargs["temperature"]

                    if response_format == "json_object":
                        kwargs["response_format"] = {"type": "json_object"}

                    response = self.client.chat.completions.create(**kwargs)

                    if response and response.choices and len(response.choices) > 0:
                        return response.choices[0].message.content
                    else:
                        logger.error(f"Empty or invalid response on retry: {response}")
                        return None
                else:
                    raise

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model


class CustomCorporateClient(BaseLLMClient):
    """Custom client for corporate LLM API with OAuth authentication."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str
    ):
        """
        Initialize custom corporate client.

        Args:
            api_url: Full API endpoint URL
            api_token: OAuth token for authentication
            model: Model name to use
        """
        self.api_url = base_url
        self.api_token = api_key
        self.model = model
        logger.info(f"Initialized custom corporate client (model={model})")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        response_format: Optional[str] = None
    ) -> Optional[str]:
        """Make chat completion request using custom corporate API."""
        try:
            # Build payload
            payload = {
                "model": self.model,
                "messages": messages
            }

            # Add response format if JSON is requested
            if response_format == "json_object":
                payload["response_format"] = {"type": "json_object"}

            # Set headers with OAuth authentication
            headers = {
                "authorization": f"OAuth {self.api_token}",
                "content-type": "application/json"
            }

            # Make API request (verify=False for self-signed certs)
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                verify=False,
                timeout=60
            )
            response.raise_for_status()

            # Extract content from custom response format
            # Format: response.json()["response"]["choices"][0]["message"]["content"]
            response_data = response.json()
            content = response_data["response"]["choices"][0]["message"]["content"]

            return content

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = f" Details: {e.response.text[:200]}"
            except Exception:
                pass
            logger.error(f"Corporate API request failed: {e}{error_detail}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Corporate API request failed: {e}")
            return None

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            return None

    def get_model_name(self) -> str:
        return self.model


def create_llm_client(
    use_custom: Optional[bool] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client.
    Automatically loads environment variables from .env in project root.

    Args:
        use_custom: If True, use custom client, else OpenAI client (if None - reads from USE_CUSTOM_CLIENT env)
        api_key: API key for OpenAI client (if None - reads from LLM_API_KEY env)
        base_url: Base URL for OpenAI client (if None - reads from BASE_URL env)
        model: Model name (if None - reads from LLM_MODEL env)

    Returns:
        Configured LLM client instance

    Raises:
        ValueError: If required configuration is missing
    """
    from dotenv import load_dotenv

    # Load .env from project root (parent of historical_data_preparation)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, '.env'))

    use_custom = use_custom or os.getenv("USE_CUSTOM_CLIENT", "false").lower() == "true"
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("BASE_URL")
    model = model or os.getenv("LLM_MODEL")

    if not model:
        raise ValueError("Model name not provided (set LLM_MODEL in env)")
    if not base_url:
        raise ValueError("Base url not provided (set BASE_URL in env)")
    if not api_key:
        raise ValueError("API key not provided (set LLM_API_KEY in env)")

    if use_custom:
        logger.info("Using custom corporate LLM client")
        return CustomCorporateClient(
            base_url=base_url,
            api_key=api_key,
            model=model
        )
    else:
        logger.info("Using OpenAI-compatible LLM client")
        return OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=model
        )


if __name__ == "__main__":
    # Test the clients
    import sys
    import os
    from dotenv import load_dotenv

    # Load .env from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, '.env'))

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n" + "="*60)
    print("TESTING LLM CLIENTS")
    print("="*60)

    # Check which client to use
    use_custom = os.getenv("USE_CUSTOM_CLIENT", "false").lower() == "true"

    print(f"\nUSE_CUSTOM_CLIENT: {use_custom}")

    # Create client
    try:
        client = create_llm_client(use_custom=use_custom)
        print(f"Client created: {client.__class__.__name__}")
        print(f"Model: {client.get_model_name()}")

        # Test simple message
        print("\nTesting simple message...")
        messages = [
            {"role": "user", "content": "Say 'hello' in one word"}
        ]

        response = client.chat_completion(messages, temperature=0.1)

        if response:
            print(f"Response: {response}")
        else:
            print("Failed to get response")

        # Test JSON response
        print("\nTesting JSON response...")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format."
            },
            {
                "role": "user",
                "content": 'Return JSON: {"status": "ok", "message": "test successful"}'
            }
        ]

        response = client.chat_completion(
            messages,
            temperature=0.1,
            response_format="json_object"
        )

        if response:
            print(f"JSON Response: {response}")
            try:
                parsed = json.loads(response)
                print(f"Parsed: {parsed}")
            except json.JSONDecodeError:
                print("Failed to parse as JSON")
        else:
            print("Failed to get JSON response")

        print("\n✓ Tests complete")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
