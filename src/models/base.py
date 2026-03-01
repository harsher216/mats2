"""
Model interfaces for running evaluations.

Supports multiple backends:
- OpenAI (o3, o4-mini, GPT-4o, etc.)
- Anthropic (Claude)
- HuggingFace / local models (for open-source models like Llama, Qwen)

Each model interface handles:
- Sending the scenario to the model
- Extracting both the final output and chain-of-thought (if available)
- Returning a standardized ModelResponse

Design note: We separate CoT extraction because different providers
expose it differently (OpenAI reasoning tokens, Anthropic thinking blocks,
open models via full generation).
"""

from abc import ABC, abstractmethod
from typing import Optional
import uuid
import json

from src.tasks.base import Scenario, ModelResponse, Message, Role


class BaseModel(ABC):
    """Abstract interface for running a model on a scenario."""
    
    model_id: str = "base"
    
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
    
    @abstractmethod
    def run(self, scenario: Scenario) -> ModelResponse:
        """Run the model on a scenario and return a response."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"


class OpenAIModel(BaseModel):
    """
    OpenAI API model (GPT-4o, o3, o4-mini, etc.)
    
    For reasoning models (o3, o4-mini), attempts to extract
    chain-of-thought from reasoning tokens if available.
    """
    
    def __init__(self, model_id: str = "gpt-4o", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    
    def run(self, scenario: Scenario) -> ModelResponse:
        client = self._get_client()
        
        messages = scenario.get_conversation()
        
        # For reasoning models, request reasoning tokens
        kwargs = {}
        if self.model_id.startswith(("o3", "o4")):
            kwargs["reasoning"] = {"effort": "medium"}
        
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
            **self.config,
        )
        
        choice = response.choices[0]
        final_output = choice.message.content or ""
        
        # Extract CoT from reasoning models
        chain_of_thought = None
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            chain_of_thought = choice.message.reasoning_content
        # Some API versions use 'reasoning' field
        elif hasattr(choice.message, "reasoning") and choice.message.reasoning:
            chain_of_thought = choice.message.reasoning
        
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]
        
        return ModelResponse(
            response_id=str(uuid.uuid4()),
            scenario=scenario,
            model_id=self.model_id,
            final_output=final_output,
            chain_of_thought=chain_of_thought,
            tool_calls=tool_calls,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )


class AnthropicModel(BaseModel):
    """
    Anthropic API model (Claude).
    
    Extracts chain-of-thought from <thinking> blocks if the model
    uses extended thinking.
    """
    
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        return self._client
    
    def run(self, scenario: Scenario) -> ModelResponse:
        client = self._get_client()
        
        # Separate system prompt from conversation
        system_prompt = scenario.get_system_prompt()
        messages = [
            msg.to_dict() for msg in scenario.messages 
            if msg.role != Role.SYSTEM
        ]
        
        kwargs = {"max_tokens": self.config.get("max_tokens", 4096)}
        
        # Enable extended thinking for CoT extraction
        if self.config.get("extended_thinking", False):
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.get("thinking_budget", 10000)
            }
        
        response = client.messages.create(
            model=self.model_id,
            system=system_prompt or "",
            messages=messages,
            **kwargs,
        )
        
        # Extract content and thinking blocks
        final_output = ""
        chain_of_thought = ""
        
        for block in response.content:
            if block.type == "text":
                final_output += block.text
            elif block.type == "thinking":
                chain_of_thought += block.thinking
        
        return ModelResponse(
            response_id=str(uuid.uuid4()),
            scenario=scenario,
            model_id=self.model_id,
            final_output=final_output,
            chain_of_thought=chain_of_thought if chain_of_thought else None,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )


class HuggingFaceModel(BaseModel):
    """
    Local/HuggingFace model for open-source models.
    
    For the CASI project, this is the primary model interface since
    we're reproducing results on open models (Llama, Qwen, etc.)
    
    Supports both HuggingFace transformers and vLLM for faster inference.
    """
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        backend: str = "transformers",  # "transformers" or "vllm"
        device: str = "auto",
        **kwargs
    ):
        super().__init__(model_id, **kwargs)
        self.backend = backend
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is not None:
            return
            
        if self.backend == "vllm":
            try:
                from vllm import LLM
                self._model = LLM(model=self.model_id)
            except ImportError:
                raise ImportError("vllm package required. Install with: pip install vllm")
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map=self.device,
                )
            except ImportError:
                raise ImportError("transformers package required. Install with: pip install transformers torch")
    
    def run(self, scenario: Scenario) -> ModelResponse:
        self._load_model()
        
        messages = scenario.get_conversation()
        
        if self.backend == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(
                max_tokens=self.config.get("max_tokens", 4096),
                temperature=self.config.get("temperature", 0.7),
            )
            outputs = self._model.chat(messages, sampling_params=params)
            final_output = outputs[0].outputs[0].text
        else:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_tokens", 4096),
                    temperature=self.config.get("temperature", 0.7),
                    do_sample=True,
                )
            
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            final_output = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # For open models, the full generation IS the chain of thought
        # We can try to separate thinking from output if the model uses
        # special tokens (e.g., <think>...</think> for DeepSeek-R1)
        chain_of_thought = None
        if "<think>" in final_output and "</think>" in final_output:
            think_start = final_output.index("<think>") + len("<think>")
            think_end = final_output.index("</think>")
            chain_of_thought = final_output[think_start:think_end].strip()
            final_output = final_output[think_end + len("</think>"):].strip()
        
        return ModelResponse(
            response_id=str(uuid.uuid4()),
            scenario=scenario,
            model_id=self.model_id,
            final_output=final_output,
            chain_of_thought=chain_of_thought,
        )


def create_model(provider: str, model_id: str, **kwargs) -> BaseModel:
    """Factory function to create a model by provider name."""
    providers = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "huggingface": HuggingFaceModel,
        "local": HuggingFaceModel,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(providers.keys())}")
    return providers[provider](model_id=model_id, **kwargs)
