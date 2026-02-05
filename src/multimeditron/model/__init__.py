from transformers import AutoConfig, AutoModel
from multimeditron.model.model import MultimodalConfig, MultiModalModelForCausalLM

AutoConfig.register("multimeditron", MultiModalModelForCausalLM)
AutoModel.register(MultimodalConfig, MultiModalModelForCausalLM)

