from infra.salign.util.get_base_model import get_base_model

class PromptTemplate:
    _instance = None

    def __new__(cls, *args, **kwargs):

        if cls._instance is not None:
            return cls._instance
        
        base_model = get_base_model()
        if base_model == "llama":
            cls._instance = LlamaPromptTemplate(*args, **kwargs)
        elif base_model == "qwen":
            cls._instance = QwenPromptTemplate(*args, **kwargs)
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        return cls._instance

class LlamaPromptTemplate:
    def user(self):
        return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    def assistant(self):
        return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

class QwenPromptTemplate:
    def user(self):
        return "<|im_start|>user\n"
    def assistant(self):
        return "<|im_end|><|im_start|>assistant\n<thinking>\n"
