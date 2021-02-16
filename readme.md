目录结构

```
scala
├─feature                           模型处理
│    ├─extractor                         模型的提取
│    │      CountVectotor.scala              文本出现数量统计
│    │      Tf_idf_learn.scala               文本在文章中的重要度
│    │      Work2VectorL.scala               文本转化为空间向量(神经网络分析)
│    │      
│    ├─selector                          特征的选择器
│    │      ChiSqSelectorL.scala              卡方选择器，维度与结果关联度选择
│    │      PCASelectorL.scala                 PCA主要成分分析、降维   
│    ├─Transformer                       模型的预处理，缩放器
│    │       MinMaxScalerL.scala              归一化处理，用于处理小而标准的数据
│    │       StandardScalerL.scala            标准化处理，将数据转为均值为0，标准差为1的正态分布模型
│    │       StringIndexerL.scala             字符穿转化为索引
│    │       IndexToStringL.scala             索引数据转回成字符串
│    │       OneHotEncoderL.scala             独热码化，将各异的数据，化为稀疏矩阵
│
├─classification                      分类算法
│    ├─DecisionTreeL                      决策树，基于信息熵id3（信息增益最大）
│    ├─RandomForestL                      随机森林
├─regression                          回归算法               
│    ├─LinearRegressionL                   线性回归，基于随机梯度下降SGD
│    ├─LogisticRegressionL                 逻辑回归
├─recommend                           推荐系统
│    ├─DataReader.scala                    数据读取帮助类
│    ├─ContentRecommender.scala            基于余弦相似的商品内容推荐 
│    ├─LFMRecommender.scala                基于LFM（矩阵分解）实现基于模型的协调过滤推荐（cf）;通过ALS（交替最小二乘法）实现

```
    
