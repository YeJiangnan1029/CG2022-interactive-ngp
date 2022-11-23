# CG2022-interactive-ngp
### 2022秋冬学期计图课程大作业

**使用前先配置好JNeRF环境！**



### inference 模块使用

因为JNeRF模块化之间过于耦合，很难提取出纯推理的接口，还是调用了`runner`里面的接口

所以调用前需要加载一次训练的数据集，用来获取数据集参数

在`data`文件夹下放好数据集，在`models`文件夹下放好训练的模型，就可以使用`inference`了

数据集和模型比较大，放在网盘上了，链接在对应的readme文件里，也可以直接在网盘文件夹里找



### 项目结构

- data

  存放训练的数据集

- data_process

  数据处理的脚本

- inference

  模型推理接口

- models

  训练好的模型

- ui

  主程序，交互界面



### 大文件资源

https://cowtransfer.com/s/23d7c51f3b3a4c

3e7gtc
