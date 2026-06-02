"""
Unified LLM client supporting multiple inference engines:
- Together AI
- vLLM 
- OpenAI
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

try:
    import openai
except ImportError:
    openai = None

try:
    import requests
except ImportError:
    requests = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

# Model-specific max_model_len mapping (experiment values with actual max in comments)
MODEL_MAX_LENGTHS = {
    # Gemma models
    "google/gemma-2-2b-it": 8000,  # actual max: 8192
    "google/gemma-3-1b-it": 8000,  # actual max: 8192
    "google/gemma-3-4b-it": 8000,  # actual max: 8192
    
    # Llama models
    "meta-llama/Llama-3.1-70B-Instruct": 10000,  # actual max: 131072
    "meta-llama/Llama-3.3-70B-Instruct": 10000,  # actual max: 131072
    "meta-llama/Llama-3.2-1B-Instruct": 10000,  # actual max: 131072
    "meta-llama/Llama-3.2-3B-Instruct": 10000,  # actual max: 131072
    
    # Qwen models
    "Qwen/Qwen2.5-72B-Instruct": 10000,  # actual max: 32768
    "Qwen/Qwen3-4B-Instruct-2507": 10000,  # actual max: 32768
    
    # Mistral models
    "mistralai/Mistral-Small-Instruct-2409": 10000,  # actual max: 32768
    "mistralai/Mistral-Large-Instruct-2411": 10000,  # actual max: 32768
    # Default fallback
    "default": 8000  # conservative default
}

def get_model_max_length(model_path: str) -> int:
    """Get the appropriate max_model_len for a given model."""
    # Try exact match first
    if model_path in MODEL_MAX_LENGTHS:
        return MODEL_MAX_LENGTHS[model_path]
    
    # Try partial matches for custom model paths
    for model_key in MODEL_MAX_LENGTHS:
        if model_key in model_path:
            return MODEL_MAX_LENGTHS[model_key]
    
    # Default fallback
    logging.warning(f"Unknown model {model_path}, using default max_model_len={MODEL_MAX_LENGTHS['default']}")
    return MODEL_MAX_LENGTHS["default"]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt asynchronously."""
        pass


class TogetherAIClient(BaseLLMClient):
    """Together AI client."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')
        
        if not self.api_key:
            raise ValueError("Together AI API key not found. Set TOGETHER_API_KEY environment variable.")
        
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        
        logging.info(f"Initialized Together AI client with model: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using Together AI."""
        if not requests:
            raise ImportError("requests library required for Together AI client")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logging.error(f"Together AI API error: {e}")
            raise
    
    async def generate_async(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using Together AI asynchronously."""
        # For now, use synchronous version wrapped in asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, max_tokens, temperature)


class vLLMClient(BaseLLMClient):
    """vLLM local inference client."""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9, download_dir: str = None):
        if LLM is None or SamplingParams is None:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.download_dir = download_dir
        
        # Set Hugging Face cache directory if specified
        if download_dir:
            os.environ['HF_HOME'] = download_dir
            logging.info(f"Set Hugging Face cache directory to: {download_dir}")
        
        # Initialize vLLM model
        logging.info(f"Loading vLLM model from: {model_path}")
        
        # Get appropriate max_model_len for this model
        max_model_len = get_model_max_length(model_path)
        logging.info(f"Using max_model_len={max_model_len} for model {model_path}")
        
        # Prepare LLM arguments
        llm_kwargs = {
            'model': model_path,
            'tensor_parallel_size': tensor_parallel_size,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'trust_remote_code': True
        }
        
        # Note: download_dir removed to avoid duplicate caching
        # HF_HOME environment variable (set above) is sufficient
            
        self.llm = LLM(**llm_kwargs)
        logging.info("vLLM model loaded successfully")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using vLLM."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
            
        except Exception as e:
            logging.error(f"vLLM generation error: {e}")
            raise
    
    async def generate_async(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using vLLM asynchronously."""
        # vLLM doesn't support async directly, so wrap in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, max_tokens, temperature)


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        if openai is None:
            raise ImportError("openai library required for OpenAI client")
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logging.info(f"Initialized OpenAI client with model: {model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_async(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using OpenAI asynchronously."""
        try:
            # Use async OpenAI client if available
            if hasattr(self.client, 'achat'):
                response = await self.client.achat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            else:
                raise AttributeError("OpenAI async client not available - upgrade openai library or use sync version")
                
        except Exception as e:
            logging.error(f"OpenAI async API error: {e}")
            raise


class UnifiedLLMClient:
    """
    Unified LLM client that supports multiple inference engines.
    """
    
    SUPPORTED_ENGINES = ["together", "vllm", "openai"]
    
    def __init__(self, 
                 engine: str,
                 model_name: Optional[str] = None,
                 model_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 **engine_kwargs):
        """
        Initialize unified LLM client.
        
        Args:
            engine: LLM engine to use ("together", "vllm", "openai")
            model_name: Model name for API-based engines
            model_path: Model path for vLLM
            api_key: API key for API-based engines
            **engine_kwargs: Additional engine-specific arguments
        """
        self.engine = engine.lower()
        
        if self.engine not in self.SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported engine: {engine}. Choose from {self.SUPPORTED_ENGINES}")
        
        # Initialize the appropriate client
        if self.engine == "together":
            if not model_name:
                model_name = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            self.client = TogetherAIClient(model_name, api_key)
            
        elif self.engine == "vllm":
            if not model_path:
                raise ValueError("model_path required for vLLM engine")
            
            # Filter kwargs for vLLM - only pass supported parameters
            vllm_kwargs = {
                k: v for k, v in engine_kwargs.items() 
                if k in ['tensor_parallel_size', 'gpu_memory_utilization', 'download_dir', 'hf_cache_dir']
            }
            # Handle hf_cache_dir -> download_dir mapping
            if 'hf_cache_dir' in vllm_kwargs:
                vllm_kwargs['download_dir'] = vllm_kwargs.pop('hf_cache_dir')
                
            self.client = vLLMClient(model_path, **vllm_kwargs)
            
        elif self.engine == "openai":
            if not model_name:
                model_name = "gpt-3.5-turbo"
            self.client = OpenAIClient(model_name, api_key)
        
        logging.info(f"Unified LLM client initialized with engine: {self.engine}")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        return self.client.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
    
    async def generate_async(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text from prompt asynchronously.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        return await self.client.generate_async(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.7, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                results.append(result)
            except Exception as e:
                logging.error(f"Error generating for prompt: {e}")
                raise  # Don't add empty string, just fail
        
        return results
    
    async def generate_batch_async(self, prompts: List[str], max_tokens: int = 500, temperature: float = 0.7, max_concurrent: int = 10, **kwargs) -> List[str]:
        """
        Generate text for multiple prompts asynchronously.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt):
            async with semaphore:
                try:
                    return await self.generate_async(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
                except Exception as e:
                    logging.error(f"Error generating for prompt: {e}")
                    raise  # Don't return empty string, just fail
        
        tasks = [generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current engine."""
        info = {"engine": self.engine}
        
        if hasattr(self.client, 'model_name'):
            info["model_name"] = self.client.model_name
        if hasattr(self.client, 'model_path'):
            info["model_path"] = self.client.model_path
            
        return info


# Factory function for easy client creation
def create_llm_client(engine: str, **kwargs) -> UnifiedLLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        engine: LLM engine ("together", "vllm", "openai")
        **kwargs: Engine-specific arguments
        
    Returns:
        UnifiedLLMClient instance
    """
    return UnifiedLLMClient(engine, **kwargs)


# Mock client for testing
class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing purposes."""
    
    def __init__(self, mock_responses: Optional[List[str]] = None):
        self.mock_responses = mock_responses or ["Mock response"]
        self.call_count = 0
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        response = self.mock_responses[self.call_count % len(self.mock_responses)]
        self.call_count += 1
        return response
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Return mock response asynchronously."""
        return self.generate(prompt, **kwargs)
