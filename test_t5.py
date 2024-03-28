import os
import sys

import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter

from transformers import AutoTokenizer

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, ".")))
# from mindone.transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(sys.path)
from mindone.transformers.models import T5Model

ms.set_context(pynative_synchronize=True)

tokenizer = AutoTokenizer.from_pretrained("./t5-small", local_files_only=True)
model = T5Model.from_pretrained("./t5-small", local_files_only=True, use_safetensors=True)

# print("************",model,"*************")

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

input_ids = Tensor(input_ids.numpy(), ms.int64)
decoder_input_ids = Tensor(decoder_input_ids.numpy(), ms.int64)
# print(input_ids.shape)

# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states_mindspore = outputs[0]

print(last_hidden_states_mindspore.shape)

from transformers import T5Model

model = T5Model.from_pretrained("./t5-small", local_files_only=True, use_safetensors=True)

# print("************",model,"*************")

# for name, _ in model.named_parameters():
#     print(name)

input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# print(input_ids.shape)

# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states_torch = outputs.last_hidden_state

print(last_hidden_states_torch.shape)

import numpy as np
print("max-mean diff", np.mean(np.abs(last_hidden_states_mindspore.asnumpy()-last_hidden_states_torch.detach().numpy())))
print("max-var diff", np.var(np.abs(last_hidden_states_mindspore.asnumpy()-last_hidden_states_torch.detach().numpy())))

