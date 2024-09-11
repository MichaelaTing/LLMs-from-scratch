## Chapter 2 Encoding

- GPT-2 使用字节对编码（BytePair encoding，简称BPE）作为其分词器。这使得模型能够将其预定义词汇表中没有的单词分解成更小的子词单元甚至单个字符，从而处理不在词汇表中的单词。这里使用的是 OpenAI 开源的 [tiktoken](https://github.com/openai/tiktoken) 库中的 BPE 分词器，该库用 Rust 实现了其核心算法，以提高计算性能。BPE from tiktoken is faster than the original BPE used in GPT-2 and BPE via Hugging Face transformers.

- embedding层只是一种更高效的实现方式，它等同于独热编码和矩阵乘法方法，所以它可以被视为一个可以通过反向传播优化的神经网络层。

```python
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs) # (batchsize, max_length, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length)) # (max_length, output_dim)
```

- 由于独热编码行中除了一个索引外，其他索引都是0，linear层与独热编码相乘本质上等同于对

- 独热元素的查找。这种在独热编码上使用矩阵乘法的方式等同于嵌入层查找，但如果处理大型嵌入矩阵，可能会效率不高，因为有很多乘以零的无用乘法。

```python
embedding = torch.nn.Embedding(vocab_size, output_dim)
linear = torch.nn.Linear(vocab_size, output_dim, bias=False)
linear.weight = torch.nn.Parameter(embedding.weight.T)
```

## Chapter 3 Attention Mechanism

- 在初始化attention机制中的q,k,v时，可以使用 PyTorch 的 `Linear` 层来简化实现，如果禁用偏置单元，它们就等同于矩阵乘法。使用 `nn.Linear` 而不是手动的 `nn.Parameter(torch.rand(...)` 方法的一个优势是，`nn.Linear` 有一个首选的权重初始化方案，这有助于模型训练更加稳定。

- 我们将之前的自注意力机制转换为因果自注意力机制。因果自注意力确保模型对序列中某个位置的预测仅依赖于之前位置已知的输出，而不依赖于未来位置。用更简单的话说，这确保了每个下一个词的预测应该只依赖于前面的词。为了实现这一点，对于每个给定的标记，我们屏蔽掉未来标记。

```python
context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
```

- 此外，还可以应用dropout来减少训练过程中的过拟合。Dropout可以在几个地方应用：例如，在计算注意力权重之后；或者在用注意力权重乘以值向量之后。这里我们在计算注意力权重之后应用dropout掩码，因为这是更常见的做法。

- 当处理 GPU 计算时，PyTorch 中的缓冲区特别有用，因为它们需要与模型的参数一起在设备之间传输（例如从 CPU 传输到 GPU）。与参数不同，缓冲区不需要梯度计算，但它们仍然需要在正确的设备上，以确保所有计算都正确执行。如果不用buffer，即实现方式为 `self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)`，那么 `module.to("cuda")`并不能将 `mask`也移动到cuda，因为它不像权重（例如 `W_query.weight`）那样是一个 PyTorch 参数。这意味着必须通过 `module.mask.to("cuda")` 手动将其移动到 GPU 上。PyTorch 缓冲区相对于普通张量的另一个优势是，它们会被包含在模型的 `state_dict` 中。

```python
self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
```

- 在PyTorch的2.2版本及以后，可以使用 `torch.nn.functional.scaled_dot_product_attention`来实现scaled dot-product attention，并支持Flash Attention。Flash Attention实现了对scaled dot product attention的加速，能够提供大约2倍的性能提升。使用时，可以结合sdpa_kernel上下文管理器，以确保在GPU上使用特定的实现，从而提高运行效率。

- 在MultiheadAttention中，设置need_weights=False，将使用优化过的 `scaled_dot_product_attention`并为 MHA（多头注意力机制）实现最佳性能。

- 下面几种MHA的实现速度最快：

  - MHA with pytorch scaled_dot_product_attention
  - pytorch MHA with need_weights=False
  - pytorch MHA defaults

## Chapter 4 LLM Architecture

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

- 在LayerNorm模块，除了通过减去均值和除以方差来进行归一化之外，还添加了两个可训练参数： `scale`（缩放）和 `shift`（平移）参数。初始的 `scale`（乘以1）和 `shift`（加上0）值没有任何效果；然而，`scale` 和 `shift` 是可训练参数，如果确定这样做可以提高模型在训练任务上的表现，大型语言模型（LLM）会在训练过程中自动调整这些参数。这允许模型学习最适合其处理数据的适当缩放和平移。

- 在方差计算中，设置 `unbiased=False` 意味着使用公式 $\frac{\sum_i (x_i - \bar{x})^2}{n}$ 来计算方差，其中 n 是样本大小；这个公式不包括贝塞尔校正（在分母中使用 `n-1`），因此提供了方差的有偏估计。对于 LLMs 来说，当嵌入维度 `n` 非常大时，使用 n 和 `n-1` 之间的差异可以忽略不计。

- ReLU 是一个分段线性函数，如果输入为正，则直接输出输入；否则，输出零。GELU 是一个平滑的非线性函数，它近似于 ReLU，但对于负值（除了大约 -0.75 之外）具有非零梯度。

- 添加如下所示的shortcut连接来缓解梯度消失

```python
if self.use_shortcut and x.shape == layer_output.shape:
    x = x + layer_output
else:
    x = layer_output
```

- 计算参数量

```python
total_params = sum(p.numel() for p in model.parameters())
total_size_bytes = total_params * 4 # assuming float32, 4 bytes per parameter
total_size_mb = total_size_bytes / (1024 * 1024) # Convert to megabytes
print(f"Total size of the model: {total_size_mb:.2f} MB")
```

- FLOPs（每秒浮点运算次数）通过计算执行的浮点运算数量来衡量神经网络模型的计算复杂性。高FLOPs表明更密集的计算和能源消耗。可以用thop库分析FLOPs。

## Chapter 5 Pretraining

- 一个与交叉熵损失相关的概念是大型语言模型（LLM）的困惑度perplexity。困惑度简单地说就是交叉熵损失的指数函数计算结果。困惑度通常被认为更具可解释性，因为它可以被理解为模型在每一步中对下一标记所不确定的词表大小。换句话说，困惑度提供了一种衡量模型预测的概率分布与数据集中单词实际分布匹配程度的方法。与损失类似，较低的困惑度表明模型预测更接近实际分布。

- 通常使用两个解码策略来修改 `generate_text_simple`：*温度缩放*和*top-k*采样。这将允许模型控制生成文本的随机性和多样性。

  * 可以通过一个称为温度缩放的概念来控制分布和选择过程。“温度缩放”只是将logits除以一个大于0的数字的高级说法。大于1的温度值将在应用softmax后导致更均匀分布的标记概率；小于1的温度值将在应用softmax后导致更自信（更尖锐）的分布。为了说明，我们不通过 `torch.argmax`来确定最可能的标记，而是使用 `torch.multinomial(probas, num_samples=1)`从softmax分布中采样来确定最可能的标记。通过 `temperature<1`进行重新缩放会得到一个更尖锐的分布，接近于 `torch.argmax`，以至于最可能的单词几乎总是被选中。通过 `temperature>1`重新缩放的概率更加均匀，在LLM中，使用该方法有时会产生无意义的文本。

  ```python
  def print_sampled_tokens(probas):
      torch.manual_seed(123) # Manual seed for reproducibility
      sample = [torch.multinomial(probas, num_samples=1).item() for _ in range(1_000)]
      sampled_ids = torch.bincount(torch.tensor(sample))
      for i, freq in enumerate(sampled_ids):
          print(f"{freq} x {inverse_vocab[i]}")
  ```

  * 为了能够使用更高的温度来增加输出的多样性，并降低无意义句子出现的概率，可以将采样的标记限制在最可能的前k个标记中

  ```python
  top_k = 3
  top_logits, top_pos = torch.topk(next_token_logits, top_k)
  new_logits = torch.full_like(next_token_logits, -torch.inf)
  new_logits[top_pos] = next_token_logits[top_pos] 
  ```

- 提供了在Project Gutenberg Dataset数据集上预训练GPT的代码，一些改进策略如下：

  * 修改 `prepare_dataset.py`脚本，从每个书籍文件中剥离古腾堡的样板文本。
  
  * 更新数据准备和加载工具，预先分词数据集并以分词形式保存，这样在每次调用预训练脚本时就不必重新分词。
  
  * 通过添加在附录D：为训练循环添加额外功能中介绍的特性来更新 `train_model_simple`脚本，即余弦衰减、线性预热和梯度裁剪。

    - 在训练像LLM这样的复杂模型时，实施学习率预热可以帮助稳定训练。在学习率预热中，逐渐将学习率从一个非常低的值（`initial_lr`）增加到用户指定的最大值（`peak_lr`）。这样，模型将以较小的权重更新开始训练，这有助于降低训练过程中出现大幅度破坏性更新的风险。
    
    - 另一种流行技术是余弦衰减，它也会在训练时期中调整学习率。在余弦衰减中，学习率遵循一个余弦曲线，从一个初始值减少到接近零，遵循半个余弦周期。这种逐渐减少的设计是为了在模型开始改善其权重时减缓学习的速度；它减少了在训练进展中过度达到最小值的风险，这对于在训练的后期阶段稳定训练至关重要。余弦衰减因其在学习率调整中的更平滑过渡而常被优先选择，但线性衰减在实践中也被使用。
    
    - 梯度裁剪是另一种在训练LLM时用来稳定训练的技术通过设置一个阈值，超出此限制的梯度会被缩小到最大幅度，以确保在反向传播期间对模型参数的更新保持在可控范围内。

    ```python
    if step < warmup_steps:
        # Linear warmup
    else:
        # Cosine annealing after warmup

    if global_step > warmup_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ```
  
  * 更新预训练脚本以保存优化器状态（见第5章 *5.4在PyTorch中加载和保存权重* ；ch05.ipynb），并添加选项以加载现有的模型和优化器检查点，如果训练运行被中断，则继续训练。

  ```python
  torch.save({
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      }, 
      "model_and_optimizer.pth"
  )
  ```
  
  * 添加更高级的日志记录器（例如，Weights and Biases）以实时查看损失和验证曲线。
  
  * 添加分布式数据并行性（DDP）并在多个GPU上训练模型（见附录A中的 *A.9.3使用多个GPU进行训练* ；DDP-script.py）。
  
  * 将 `previous_chapter.py`脚本中的从头开始的 `MultiheadAttention`类替换为在高效的多头注意力实现奖励部分实现的高效的 `MHAPyTorchScaledDotProduct`类，该类通过PyTorch的 `nn.functional.scaled_dot_product_attention`函数使用Flash Attention。
  
  * 通过[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）或[thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加速训练。

  ```python
  compiled_model = torch.compile(model, mode='training', backend='nvfuser')
  ```
  
  * 实现梯度低秩投影（GaLore）以进一步加速预训练过程。这可以通过简单地将 `AdamW`优化器替换为[GaLore Python库](https://github.com/jiaweizzhao/GaLore)中提供的 `GaLoreAdamW`来实现。

- 网格搜索，选取最优超参数：

  ```python
  HPARAM_GRID = {
      "batch_size": [2, 4, 8, 16],
      "drop_rate": [0.0, 0.1, 0.2],
      "warmup_iters": [10, 20, 30],
      "weight_decay": [0.1, 0.01, 0.0],
      "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
      "initial_lr": [0.00005, 0.0001],
      "min_lr": [0.00005, 0.00001, 0.0001],
      "n_epochs": [5, 10, 15, 20, 25],
  }
  hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
  for combination in hyperparameter_combinations:
      HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))
  ```

## Chapter 6 Finetuning for Classification

* 注意文本序列有不同的长度。如果想在一批中组合多个训练示例，必须将所有消息截断为数据集或批次中最短文本序列的长度，或者将所有消息填充到数据集或批次中最长文本序列的长度。

* 首先冻结模型，这意味着使所有层都不可训练；然后替换输出层（`model.out_head`）；最后使最后一个transformer块和将最后一个transformer块连接到输出层的最终 `LayerNorm`模块可训练。

```python
for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
```

* 如果我们向模型提供具有 4 个token的text，那么输出由 4 个二维向量组成。基于因果注意力机制，最后一个标记包含所有标记中最多的信息，因为它是唯一包含所有其他标记信息的标记。因此，我们对最后一个标记特别感兴趣。

* 微调阶段一些策略及其效果解读：

  * **训练最后一个与第一个输出Token位置**：训练最后一个输出Token位置的结果比训练第一个Token位置的性能要好得多。这种改进是由于因果自注意力掩码的使用。
  * **训练最后一个Transformer Block与最后一层**：训练整个最后一个Transformer Block的结果比仅训练最后一层要好得多。
  * **训练最后一个与最后两个Transformer Blocks**：与仅训练最后一个Block相比，训练最后两个Transformer Blocks可以使准确率显著提高。
  * **训练最后一个Transformer Block与所有层** ：训练所有层比仅训练最后一个Transformer Block性能提高，但训练时间几乎长了三倍。此外，它的性能并不比仅训练最后两个Transformer Block更好。
  * **使用更大的预训练模型**：使用一个大三倍的预训练模型会导致结果变差。然而，使用一个五倍大的模型比初始模型提高了性能。同样，12倍大的模型进一步提高了预测性能。（中等大小的模型可能预训练得不够好，或者特定的微调配置对这个模型的效果不佳。）
  * **使用具有随机权重与预训练权重的模型**：使用具有随机权重的模型得到的结果仅比使用预训练权重的结果稍差。
  * **使用LoRA（低秩适应）与训练所有层**：保持模型冻结并添加可训练的LoRA层是训练所有模型参数的一个可行的替代方案，甚至可以提高性能，这可能是由于过拟合较少。此外，使用LoRA也更节省内存，因为需要更新的参数较少。
  * **将输入填充到完整的上下文长度与最长训练示例**：将输入填充到完整的支持上下文长度的结果明显更差。
  * **填充与不填充** ：`--no_padding`选项禁用了数据集中的填充，这要求以批量大小1训练模型，因为输入的长度是可变的。这导致测试准确率更好，但训练时间更长。这里我们额外启用了8步梯度累积，以实现与其他实验相同的批量大小，这有助于减少过拟合并轻微提高测试集准确率。
  * **禁用因果注意力掩码**：禁用了多头注意力模块中使用的因果注意力掩码，这意味着所有Token都可以关注所有其他Token。与使用因果掩码的GPT模型相比，模型准确率略有提高。
  * **在损失和反向传播中忽略填充索引**：设置 `--ignore_index 50256`在PyTorch的 `cross_entropy`损失函数中排除了 `|endoftext|`填充Token。在这种情况下，它没有任何效果，因为我们替换了输出层，使得Token ID对于二元分类示例来说要么是0要么是1。然而，当在第7章中对模型进行指令微调时，此设置是有用的。

## Chapter 7 Finetuning To Follow Instructions

- 在第5章中看到预训练LLM涉及一个训练过程，该过程学习一次生成一个单词。因此，预训练的LLM擅长文本补全，但不擅长遵循指令。指令微调通常被称为“监督指令微调”，因为它涉及在数据集上训练模型，其中输入-输出对是明确提供的。

- 引入一个 `ignore_index`值来将所有填充令牌ID替换为一个新值，这个 `ignore_index`的目的是可以在损失函数中忽略填充值。默认情况下，PyTorch具有cross_entropy(..., ignore_index=-100)设置，用于忽略与标签-100相对应的样本。使用这个-100 ignore_index，可以忽略批次中用于将训练样本填充到相等长度的额外文本结束（填充）标记。但是，我们不想忽略文本填充标记（50256）的第一个实例，因为它可以帮助LLM判断响应何时完成。

```python
mask = targets == pad_token_id
indices = torch.nonzero(mask).squeeze()
if indices.numel() > 1:
    targets[indices[1:]] = ignore_index
```

- PyTorch 的 `DataLoader` 允许通过 `collate_fn` 参数传入一个自定义的函数来决定如何将多个数据样本组合成一个批次。我们使用Python的functools标准库中的partial函数，通过预先填充原始函数的device参数来创建一个新函数。从输出可以看到所有批次的大小均为8，但长度不同，符合预期。

```python
from functools import partial

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]

        # Pad sequences to max_length
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
```

- 在实践中，像聊天机器人这样的指令微调大型语言模型（LLMs）通过多种方法进行评估：

  - 简短回答和多项选择基准测试，如MMLU（“测量大规模多任务语言理解”，[https://arxiv.org/abs/2009.03300](https://arxiv.org/abs/2009.03300)），它测试模型的知识
  
  - 与其它LLMs的人类偏好比较，例如LMSYS聊天机器人竞技场([https://arena.lmsys.org](https://arena.lmsys.org/))
  
  - 自动化对话基准测试，其中使用另一个LLM（如GPT-4）来评估回应，例如AlpacaEval([https://tatsu-lab.github.io/alpaca_eval/](https://tatsu-lab.github.io/alpaca_eval/))

- 本节我们使用另一个更大的LLM自动评估微调后的LLM的响应，即使用Meta AI的经过指令微调的80亿参数的Llama 3模型，该模型可以通过ollama在本地运行 ([https://ollama.com](https://ollama.com/)) （如果更喜欢通过OpenAI API使用功能更强大的LLM，如GPT-4，请参见 [llm-instruction-eval-openai.ipynb](https://github.com/datawhalechina/llms-from-scratch-cn/blob/65cc17a68c4cfab395dc7b39017f89bb953ddb1a/Codes/ch07/03_model-evaluation/llm-instruction-eval-openai.ipynb))。Ollama是一个用于高效运行LLM的应用程序，它是llama.cpp的一个包装器 ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp))，使用纯C/C++实现LLM以最大化效率。注意，这是一个使用LLMs生成文本（推理）的工具，而不是用于训练或微调LLMs的工具。

- 使用ollama与“llama3”模型（即80亿参数的模型）需要16GB的RAM；如果机器不支持，可以尝试较小的模型，比如通过将 `model = "phi-3"`设置为38亿参数的phi-3模型，这只需要8GB的RAM。或者，如果机器支持，也可以使用更大的700亿参数的Llama 3模型，只需将“llama3”替换为“llama3:70b”即可。

- 提供一个空的提示模板 `"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"`，这将导致经过指令微调的LLaMA 3模型生成一个指令。

```python
query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
result = query_model(query, role="assistant")
instruction = extract_instruction(result)
response = query_model(instruction, role="user")
```

- 低秩适应（LoRA）是一种机器学习技术，它通过仅调整模型参数中的一小部分低秩子集，来修改预训练模型，使其更好地适应特定、通常是更小的数据集。这种方法很重要，因为它允许在特定任务数据上高效地微调大型模型，显著降低了微调所需的计算成本和时间。将每个线性层替换成线性层+LoRA（初始化B=0，因此模型输出应该与预训练模型一致)：

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)
```

## Appendix PyTorch & DDP

- torch.tensor()  返回NumPy array的复制；torch.from_numpy()  和NumPy array共享内存

- 在PyTorch中，`view`和 `reshape`是用于改变张量形状的两个常用方法。差异：要使用 `view`，原始张量必须在内存中是连续的。如果不连续，需要先调用 `.contiguous()`方法；`reshape`方法更灵活，因为它会自动处理非连续张量，如果需要，它会在内部调用 `.contiguous()`。如果确定张量是连续的，并且想要一个非常轻量级的操作，`view`是一个好选择。如果不确定张量是否连续，或者想要一个更健壮的方法，`reshape`是更好的选择。

- tensor.matmul(tensor) 等同于 tensor @ tensor，矩阵乘法。

- 允许Apple Silicon Chips的写法（比Apple CPU大约快两倍）：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")
```

- 分布式数据并行写法：

  ```python
  import os
  import torch.multiprocessing as mp
  from torch.utils.data.distributed import DistributedSampler
  from torch.nn.parallel import DistributedDataParallel as DDP
  from torch.distributed import init_process_group, destroy_process_group

  def ddp_setup(rank, world_size):
      """
      Arguments:
          rank: a unique process ID
          world_size: total number of processes in the group
      """
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ["MASTER_PORT"] = "12345"
      init_process_group(backend="nccl", rank=rank, world_size=world_size)
      torch.cuda.set_device(rank) # 将当前进程绑定到对应的GPU上，确保每个进程只会使用分配给它的GPU资源。


  train_loader = DataLoader(
          dataset=train_ds,
          batch_size=2,
          shuffle=False,  # False because of DistributedSampler below
          pin_memory=True,
          drop_last=True,
          sampler=DistributedSampler(train_ds) # DistributedSampler确保在多个GPU上训练时，每个GPU获得的数据样本不重叠，即每个样本仅被一个GPU处理，这对于确保模型训练的有效性和公平性至关重要。
  )


  def main(rank, world_size, num_epochs):

      ddp_setup(rank, world_size)  # initialize process groups
      train_loader, test_loader = prepare_dataset()
      model = NeuralNetwork(num_inputs=2, num_outputs=2)
      model.to(rank)
      optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
      model = DDP(model, device_ids=[rank])  # 使用 DDP 包装模型
      # the core model is now accessible as model.module

      for epoch in range(num_epochs):
          model.train()
          for features, labels in train_loader:
              features, labels = features.to(rank), labels.to(rank)  # use rank
              logits = model(features)
              loss = F.cross_entropy(logits, labels)  # Loss function
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              # LOGGING
              print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batchsize {labels.shape[0]:03d}"
                    f" | Train/Val Loss: {loss:.2f}")

      model.eval()
      print(f"[GPU{rank}] Training accuracy", compute_accuracy(model, train_loader, device=rank))
      print(f"[GPU{rank}] Test accuracy", compute_accuracy(model, test_loader, device=rank))
      destroy_process_group()  # 清理分布式进程组，确保资源被正确释放。


  if __name__ == "__main__":
      # spawn new processes 生成新的进程
      # note that spawn will automatically pass the rank
      num_epochs = 3
      world_size = torch.cuda.device_count()
      mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) # nprocs=world_size 指定了要生成的进程数，这里设置为与可用 GPU 数量相同，意味着每个 GPU 将由一个单独的进程控制。
  ```

- 当使用 `torch.nn.parallel.DistributedDataParallel` (DDP) 包装一个模型时，原始模型被包装在一个 DDP 对象中。为了访问原始模型，需要通过 `model.module` 属性来获取。比如打印模型架构：`print(model.module)`。在保存模型时，应该保存 `model.module.state_dict()`，在加载模型时，应该先创建模型实例，然后加载状态字典到 `model.module.load_state_dict()`。
