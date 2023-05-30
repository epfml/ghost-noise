# Ghost Noise for Regularizing Deep Neural Networks
This is the official code for our manuscript https://arxiv.org/abs/2305.17205 where we investigate the regularization effects of using finite batch sizes with batch normalization. The abstract is repeated below:

Batch Normalization (BN) is widely used to stabilize the optimization process and improve the test performance of deep neural networks. The regularization effect of BN depends on the batch size and explicitly using smaller batch sizes with Batch Normalization, a method known as Ghost Batch Normalization (GBN), has been found to improve generalization in many settings. We investigate the effectiveness of GBN by disentangling the induced "Ghost Noise" from normalization and quantitatively analyzing the distribution of noise as well as its impact on model performance. Inspired by our analysis, we propose a new regularization technique called Ghost Noise Injection (GNI) that imitates the noise in GBN without incurring the detrimental train-test discrepancy effects of small batch training. We experimentally show that GNI can provide a greater generalization benefit than GBN. Ghost Noise Injection can also be beneficial in otherwise non-noisy settings such as layer-normalized networks, providing additional evidence of the usefulness of Ghost Noise in Batch Normalization as a regularizer.

# Code Organization
Our core implementations are in the nodo subdirectory, in particular [nodo/ghost_noise_injector_replacement.py](nodo/ghost_noise_injector_replacement.py) which implements Ghost Noise Injection using sampling with replacement.

We use a slightly modified version of the [TIMM library](https://huggingface.co/docs/timm/index) by Ross Wightman and HuggingFace in our experiments.
Our version is in [submodules/timm/](submodules/timm/). 
The modifications primarily focus on CIFAR compatibility and more flexibility in the construction of the models.

The [shared/](shared/) subdirectory contains tools to integrate our code with TIMM.
Our experiments expect the root directory to be in the PYTHONPATH enviroment variable and looks for the shared directory.
This can be done by adding `PYTHONPATH=$PYTHONPATH:/path/to/current/dir` in front of your python launch command.

Examples of launch commands can be found in the [experiment_scripts folder](experiment_scripts).

We an exported conda environment in [environment.yml](environment.yml) that we used in our runs. Our code only relies on the core PyTorch library but TIMM has additional dependencies.
We use [wandb](https://wandb.ai/) for experiment logging through TIMM.
