from typing import Any, Dict

from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import LoRATM, TemplateType, get_model_tokenizer, register_model
from swift.utils import get_logger

logger = get_logger()


class CustomModelType:
    qwen_32b = 'qwen-32b-instruct'
    qwen_14b = 'qwen-14b-instruct'


class CustomTemplateType:
    qwen = 'qwen'


@register_model(CustomModelType.qwen_32b,
                '/root/autodl-tmp/qwen2_5-32b-instruct', LoRATM.llama,
                TemplateType.default_generation)
                

@register_model(CustomModelType.qwen_14b,
                '/root/autodl-tmp/qwen2_5-14b-instruct', LoRATM.llama,
                TemplateType.default_generation)


def get_gemma_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True')
        model_kwargs['use_flash_attention_2'] = True
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs)
    return model, tokenizer