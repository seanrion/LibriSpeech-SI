# 项目目录说明
```
│  audio.py
│  Audio_preprocessing.py
│  batcher.py
│  Build_keras_inputs.py
│  Constant.py
│  conv_models.py
│  eval_metrics.py
│  Mix_train_set_and_noise.py
│  requirements.txt
│  silence_threshold.txt
│  test.py
│  train.py
│  triplet_loss.py
│  utils.py
|  predict.py
|  eval_models.py
|  trainlog.csv
|  pretrainlog.csv
|  pretrain_models_eval.csv
|  train_models_eval.csv
├─audio_dir
│  ├─noise
│  ├─test
|  ├─test-noisy
│  └─train       
├─cache_dir
│  └─train   
├─checkpoints-softmax
├─checkpoints-triplets
```
`audio_dir` 存放音频文件数据，将解压的音频数据放在这里

`cache_dir` 存放音频文件切除静音片段的numpy数组文件

`checkpoints-triplets`,`checkpoints-softmax` 存放模型的checkpoints

`audio.py` 处理音频相关的类和函数

`Audio_preprocessing.py` 建立音频数组数据目录

`batcher.py` 存放批采样相关的类

`Build_keras_inputs.py` 进行数据采样，存放采样结果

`Constant.py` 存放各个常数

`conv_models.py` 模型

`eval_metrics.py` 存放评价函数

`Mix_train_set_and_noise.py` 将不带噪音的音频文件和噪音文件进行随机混合生成训练数据

`silence_threshold.txt` 存放静音门槛数据

`test.py` 测试集测试

`train.py` 训练

`triplet_loss.py` 存放损失函数

`utils.py` 存放常用函数

`eval_models.py` 评估模型

`predict.py` 预测

`trainlog.csv`,`pretrainlog.csv` 训练模型和预训练模型的训练log

`pretrain_models_eval.csv`,`train_models_eval.csv` 模型的评估数据
# 使用顺序

```shell
python Mix_train_set_and_noise.py
```
生成带噪音训练数据
```shell
python Audio_preprocessing.py
```
音频文件预处理生成数组数据
```shell
python Build_keras_inputs.py
```
生成批采样结果
```shell
python train.py
```
训练
```shell
python eval_models.py
```
评估模型
```shell
python test.py
```
测试
```shell
python predict.py
```
预测

# 进度

- [x] 生成带噪音训练数据
- [x] 音频文件预处理生成数组数据
- [x] 生成批采样结果
- [x] 训练
- [x] 评估模型
- [x] 测试
- [x] 预测