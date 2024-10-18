# pytorch Tutorial

## torch

> 创建操作 Creation Operations

- torch.is_tensor(obj)
- torch.numel(input)->int  返回张量中的元素个数
- torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None)  设置打印选项
- torch.from_numpy(ndarray) → Tensor
- torch.eye(n, m=None, out=None) → Tensor
- torch.zeros(*sizes, out=None) → Tensor
- torch.ones(*sizes, out=None) → Tensor
- torch.rand(*sizes, out=None) → Tensor  包含了从[0,1)的均匀分布中抽取的一组随机数
- torch.randn(*sizes, out=None) → Tensor  包含了从标准正态分布中抽取一组随机数
- torch.linspace(start, end, steps=100, out=None) → Tensor  返回一个1维张量，包含在区间start和end上均匀间隔的steps个点，输出张量的长度为steps。
- torch.randperm(n, out=None) → LongTensor  返回一个从0到n-1的随机整数排列。
- torch.arange(start, end, step=1, out=None) → Tensor

> 索引,切片,连接,换位 Indexing, Slicing, Joining, Mutating Ops

- torch.cat(inputs, dimension=0) → Tensor
- torch.stack(sequence, dim=0)
- torch.chunk(tensor, chunks, dim=0)  在给定维度(轴)上将输入张量进行分块，chunks (int) – 分块的个数
- torch.split(tensor, split_size, dim=0)  将输入张量分割成相等形状的chunks。如果沿指定维的张量形状大小不能被split_size 整分， 则最后一个分块会小于其它分块。
- torch.index_select(input, dim, index, out=None) → Tensor  沿着指定维度对输入进行切片，取index中指定的相应项(index为一个LongTensor)，然后返回到一个新的张量， 返回的张量与原始张量有相同的维度(在指定轴上)。
- torch.masked_select(input, mask, out=None) → Tensor  根据掩码张量mask中的二元值，取输入张量中的指定项(mask为一个ByteTensor)，将取值返回到一个新的1D张量，mask须跟input有相同数量的元素数目，但形状或维度不需要相同。返回的张量不与原始张量共享内存空间。
- torch.nonzero(input, out=None) → LongTensor  输出张量中的每行包含输入中非零元素的索引。
- torch.squeeze(input, dim=None, out=None)  将输入张量形状中的1去除并返回。
- torch.transpose(input, dim0, dim1, out=None) → Tensor  交换维度dim0和dim1。输出张量与输入张量共享内存。
- torch.unsqueeze(input, dim, out=None)  返回一个新的张量，对输入的制定位置插入维度1。返回张量与输入张量共享内存。

> 随机抽样 Random sampling

- torch.manual_seed(seed)
- torch.normal(means, std, out=None)
- torch.bernoulli(input, out=None) → Tensor

> 序列化 Serialization

- torch.save(obj, f)
- torch.load(f, map_location=None)

> 并行化 Parallelism

- torch.get_num_threads() → int  获得用于并行化CPU操作的OpenMP线程数
- torch.set_num_threads(int)  设定用于并行化CPU操作的OpenMP线程数

> 数学操作 Math operations

- torch.abs(input, out=None) → Tensor
- torch.add(input, value, out=None)
- torch.mul(input, value, out=None)  按元素乘
- torch.div(input, value, out=None)  按元素除
- torch.ceil(input, out=None) → Tensor
- torch.floor(input, out=None) → Tensor
- torch.round(input, out=None) → Tensor
- torch.clamp(input, min, max, out=None) → Tensor  将input张量每个元素的夹紧到区间[min, max]
- torch.exp(tensor, out=None) → Tensor
- torch.log(input, out=None) → Tensor
- torch.pow(input, exponent, out=None)
- torch.sqrt(input, out=None) → Tensor
- torch.sign(input, out=None) → Tensor
- torch.sigmoid(input, out=None) → Tensor
- torch.sum(input, dim, out=None) → Tensor
- torch.prod(input, dim, out=None) → Tensor
- torch.cumsum(input, dim, out=None) → Tensor  返回输入沿指定维度的累积和。
- torch.cumprod(input, dim, out=None) → Tensor  返回输入沿指定维度的累积积。
- torch.mean(input, dim, out=None) → Tensor
- torch.var(input, dim, out=None) → Tensor
- torch.std(input, dim, out=None) → Tensor
- torch.norm(input, p, dim, out=None) → Tensor
- torch.eq(input, other, out=None) → Tensor
- torch.ne(input, other, out=None) → Tensor
- torch.max(input, dim, max=None, max_indices=None) -> (Tensor, LongTensor)
- torch.min(input, dim, min=None, min_indices=None) -> (Tensor, LongTensor)
- torch.sort(input, dim=None, descending=False, out=None) -> (Tensor, LongTensor)
- torch.kthvalue(input, k, dim=None, out=None) -> (Tensor, LongTensor)  取输入张量input指定维上第k 个最小值。如果不指定dim，则默认为input的最后一维。
- torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
- torch.diag(input, diagonal=0, out=None) → Tensor  diagonal=0, 主对角线；diagonal>0 主对角线之上；diagonal<0, 主对角线之下
- torch.trace(input) → float
- torch.tril(input, k=0, out=None) → Tensor  包含输入矩阵(2D张量)的下三角部分。参数k控制对角线：k =0, 主对角线；k>0, 主对角线之上；k<0, 主对角线之下
- torch.triu(input, k=0, out=None) → Tensor  包含输入矩阵(2D张量)的上三角部分
- torch.dot(tensor1, tensor2) → float  计算两个张量的点乘(内乘),两个张量都为1-D 向量
- torch.mm(mat1, mat2, out=None) → Tensor  对矩阵mat1和mat2进行相乘
- torch.bmm(batch1, batch2, out=None) → Tensor  对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操作。
- torch.eig(a, eigenvectors=False, out=None) -> (Tensor, Tensor)  计算实方阵a的特征值和特征向量
- torch.inverse(input, out=None) → Tensor  对方阵输入input 取逆。
- torch.svd(input, some=True, out=None) -> (Tensor, Tensor, Tensor)

## torch.Tensor

Torch定义了七种CPU tensor类型和八种GPU tensor类型：

| data type                | cpu tensor         | gpu tensor              |
| ------------------------ | ------------------ | ----------------------- |
| 32-bit floating point    | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point    | N/A                | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.LongTensor   | torch.cuda.LongTensor   |

注意： 会改变tensor的函数操作会用一个下划线后缀来标示。比如，torch.FloatTensor.abs_()会在原地计算绝对值，并返回改变后的tensor，而tensor.FloatTensor.abs()将会在一个新的tensor中计算结果。

- copy_(src, async=False) → Tensor  将src中的元素复制到tensor中并返回这个tensor。 可以是不同的数据类型或存储在不同的设备上。
- cpu() → Tensor  如果在CPU上没有该tensor，则会返回一个CPU的副本
- cuda(device=None, async=False)  返回此对象在CPU内存中的一个副本 如果对象已存在CUDA存储中并且在正确的设备上，则不会进行复制并返回原始对象。
- element_size() → int  返回单个元素的字节大小。
- is_contiguous() → bool  如果该tensor在内存中是连续的则返回True。
- is_cuda
- masked_fill_(mask, value)  在mask值为1的位置处用value填充。
- numpy() → ndarray  将该tensor以NumPy的形式返回ndarray，两者共享相同的底层内存。
- pin_memory()  如果原来没有在固定内存中，则将tensor复制到固定内存中。

## torch.nn

> Parameters

- class torch.nn.Parameter()  Parameters 是 Variable 的子类。Paramenters和Modules一起使用的时候会有一些特殊的属性，当Paramenters赋值给Module的属性的时候，他会自动的被加到 Module的参数列表中(会出现在 parameters() 迭代器中)。将Varibale赋值给Module属性则不会有这样的影响。 这样做的原因是：我们有时候会需要缓存一些临时的状态(state), 比如：模型中RNN的最后一个隐状态。如果没有Parameter这个类的话，那么这些临时变量也会注册成为模型变量。Variable 与 Parameter的另一个不同之处在于，Parameter不能被 volatile(无法设置volatile=True)而且默认requires_grad=True。Variable默认requires_grad=False。

> Containers

- class torch.nn.Module  所有网络的基类。你的模型也应该继承这个类。

  - cpu(device_id=None)  将所有的模型参数(parameters)和buffers复制到CPU
  - cuda(device_id=None) 将所有的模型参数(parameters)和buffers赋值GPU
  - train(mode=True)  将module设置为 training mode。仅仅当模型中有Dropout和BatchNorm是才会有影响。
  - eval()  将模型设置成evaluation模式，仅仅当模型中有Dropout和BatchNorm是才会有影响。
  - zero_grad()  将module中的所有模型参数的梯度设置为0
  - forward(*input)  定义了每次执行的计算步骤。在所有的子类中都需要重写这个函数。
  - children()  返回当前模型子模块的迭代器。
  - modules() 返回一个包含当前模型所有模块的迭代器。
  - parameters()  返回一个包含模型所有参数的迭代器。一般用来当作optimizer的参数。
  - state_dict()  返回一个字典，保存着module的所有状态。
  - load_state_dict(state_dict)  用来加载模型参数。
- class torch.nn.Sequential(*args)  一个时序容器。Modules会以他们传入的顺序被添加到容器中。也可以传入一个OrderedDict。
- class torch.nn.ModuleList(modules=None)  将submodules保存在一个list中。ModuleList可以像一般的list一样被索引。而且ModuleList中包含的modules已经被正确的注册，对所有的module method可见。

  - append(module)  等价于 list 的 append()
  - extend(modules)  等价于 list 的 extend()
- class torch.nn.ParameterList(parameters=None)  将submodules保存在一个list中。ParameterList可以像一般的list一样被索引。而且ParameterList中包含的parameters已经被正确的注册，对所有的module method可见。

> Convolution layers

- class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

> Pooling layers

- class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
- class torch.nn.MaxUnpool2d(kernel_size, stride=None, padding=0)  Maxpool2d的逆过程
- class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

> Activation layers

- class torch.nn.ReLU(inplace=False)
- class torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
- class torch.nn.Sigmoid
- class torch.nn.Tanh
- class torch.nn.Softplus(beta=1, threshold=20)
- class torch.nn.Softmax
- class torch.nn.LogSoftmax

> Normalization layers

- class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)  对小批量3d数据组成的4d输入进行批标准化操作，输入：(N, C，H, W) 输出：(N, C, H, W）

> Recurrent layers

- class torch.nn.RNN(*args, **kwargs)
- class torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
- class torch.nn.LSTM(*args, **kwargs)
- class torch.nn.LSTMCell(input_size, hidden_size, bias=True)
- class torch.nn.GRU(*args, **kwargs)
- class torch.nn.GRUCell(input_size, hidden_size, bias=True)

> Linear layers

- class torch.nn.Linear(in_features, out_features, bias=True)

> Dropout layers

- class torch.nn.Dropout(p=0.5, inplace=False)

> Sparse layers

- class torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)

> Loss functions

- class torch.nn.L1Loss(size_average=True)
- class torch.nn.MSELoss(size_average=True)
- class torch.nn.CrossEntropyLoss(weight=None, size_average=True)  此标准将LogSoftMax和NLLLoss集成到一个类中。当训练一个多类分类器的时候，这个方法是十分有用的。
- class torch.nn.NLLLoss(weight=None, size_average=True)
- class torch.nn.KLDivLoss(weight=None, size_average=True)

> Multi-GPU layers

- class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)  在模块级别上实现数据并行。

> Utilities

- torch.nn.utils.clip_grad_norm(parameters, max_norm, norm_type=2)

> Transformer layers

- torch.nn.MultiheadAttention
- torch.nn.Transformer

## torch.nn.functional

> Convolution functions

- torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
- torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)  在由几个输入平面组成的输入图像上应用二维转置卷积，有时也称为“去卷积”。

> Pooling functions

- torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
- torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

> Activation functions

- torch.nn.functional.relu(input, inplace=False)
- torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
- torch.nn.functional.softmax(input)
- torch.nn.functional.tanh(input)
- torch.nn.functional.sigmoid(input)

> Normalization functions

- torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)

> Linear functions

- torch.nn.functional.linear(input, weight, bias=None)

> Dropout functions

- torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)

> Loss functions

- torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)
- torch.nn.functional.kl_div(input, target, size_average=True)
- torch.nn.functional.cross_entropy(input, target, weight=None, size_average=True)

## torch.utils.data

- class torch.utils.data.Dataset  所有数据集的基类。所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)。
- class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=`<function default_collate>`, pin_memory=False, drop_last=False)  数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
- class torch.utils.data.sampler.RandomSampler(data_source)
- class torch.utils.data.sampler.SubsetRandomSampler(indices)
- class torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples, replacement=True)

## torch.optim

- class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
- class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
- class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
- class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
- class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)  将每个参数组的学习速率设置为给定函数的初始lr倍
- class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)  将每个参数组的学习速率设置为每个step_size epochs由gamma衰减的初始lr
- class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
- class torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
- class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

## torchvision.datasets

- MNIST
- COCO
- Imagenet-12
- CIFAR10 and CIFAR100

## torchvision.models

- AlexNet
- VGG
- ResNet
- DenseNet

## torchvision transform

- class torchvision.transforms.Compose(transforms)
- class torchvision.transforms.CenterCrop(size)
- class torchvision.transforms.RandomCrop(size, padding=0)
- class torchvision.transforms.RandomHorizontalFlip
- class torchvision.transforms.RandomSizedCrop(size, interpolation=2)
- class torchvision.transforms.Normalize(mean, std)
- class torchvision.transforms.ToTensor
