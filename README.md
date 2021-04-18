# Channel-Estimation-and-Hybrid-Precoding-for-Millimeter-Wave-Systems-Based-on-Deep-Learning

1. Put the _"matlab"_ and _"python"_ folders in a root directory

2. The _"matlab"_ folder contains traditional HBF algorithm, channel estimation algorithm and data generation code. The _"python"_ file contains the defined neural network models and the trained models.

3. If you want to test the HBF-Net and CE-HBF-Net directly, you can
* Run _"matlab/channel_gen.m"_ to generate test channel. Run _"matlab/gen_testdata.m"_ to generate test dataset
* You can also click [here](https://pan.baidu.com/s/1y6R4lY5XtMC_8MapTThHYQ) (Extraction code: om9r) to download the data set without generating a new test data set.
* The trained models are saved in _"python/model"_. Run "python/main.py" in test mode (train_flag=0), you can test the performance of HBF-Net and CE-HBF-Net.
  
4. If you want to retrain the HBF-Net and CE-HBF-Net, you can
* Run _"matlab/gen_traindata.m"_ to generate training data set
* Set _"python/main.py"_ to training mode (train_flag=1), you can retrain the corresponding neural network model
  
**Pay attention to the correspondence between the _saved path_ and the _load path_**
