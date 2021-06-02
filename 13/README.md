# Hi

MNIST 평균을 구하기 위해 진행된 프로젝트 입니다.

# package

```shell
conda install scikit-learn

conda install pydot
conda install graphviz

conda install matplotlib
```

# 이슈

1. 누락된 package의 경우 위의 cli를 통해 설치하도록 합니다.
1. metrics='accuracy' 에러
   1. `keras.metrics.Accuracy()` 으로 변경하였습니다.
   1. [레퍼런스](https://github.com/tensorflow/tensorflow/issues/34088)
1. ValueError: Shapes (32, 10) and (32, 1) are incompatible
   1. `keras.metrics.SparseCategoricalAccuracy()`
   1. [레퍼런스](https://somjang.tistory.com/entry/TF20-MNIST-ValueError-Shapes-32-10-and-32-1-are-incompatible-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95)

# 레퍼런스

1. [8-2.ipynb - Colaboratory (google.com)](https://colab.research.google.com/github/rickiepark/hg-mldl/blob/master/8-2.ipynb#scrollTo=VsZV03UD5Qb5)
