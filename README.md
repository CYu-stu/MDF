# MDF-Net:Multi-layer Feature Dynamic Interactive Fusion Network for Few-Shot Fine-Grained Image Classification

Code relaese for [MDF-Net:Multi-layer Feature Dynamic Interactive Fusion Network for Few-Shot Fine-Grained Image Classification). 

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate MDF
  ```

## Dataset

Datasets refer to the introduction in the paper. The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN).

## Train

* For example, to train MDF on `CUB_fewshot_cropped` with ResNet-12 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/ResNet-12
    python ./test.py
```

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- rational_cy@163.com


