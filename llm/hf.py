from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.api import get_gpt_response


MODEL_LIST = {
    'qwen': ['your_model_path'],
}


class LLMModel(object):
    def __init__(self, model_name: str, device: str = "cuda:0", dtype="auto"):
        """
        Initialize the LLMModel instance.

        Args:
            model_name (str): The name of the model to use.
            device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').
            dtype (str): Data type for tensors (default is "auto").
        """
        self.dtype = dtype  # Set data type
        self.device = device  # Set device
        self.model_name = model_name  # Set model name
        if self.model_name in ['qwen', 'glm']:
            self.model_path = MODEL_LIST[self.model_name][0]  # Get model path from MODEL_LIST
            self.tokenizer, self.model = self._load_model_and_tokenizer()  # Load model and tokenizer

    def _load_model_and_tokenizer(self):
        """
        Load the tokenizer and model based on the model path.

        Returns:
            tuple: A tuple containing the tokenizer and model.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=self.device, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=self.dtype, device_map=self.device, trust_remote_code=True)
        return tokenizer, model

    def generate_response(self, prompt):
        """
        Generate a response based on the provided prompt.

        Args:
            prompt (str): The input text for which the response is to be generated.

        Returns:
            str: The generated response.
        """
        if self.model_name == "openai":
            response = get_gpt_response(prompt)
            return response
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[0].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
            return response