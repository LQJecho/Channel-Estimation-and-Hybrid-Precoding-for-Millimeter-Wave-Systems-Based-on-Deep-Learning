# Channel-Estimation-and-Hybrid-Precoding-for-Millimeter-Wave-Systems-Based-on-Deep-Learning

1. Put the "matlab" and "python" folders in a root directory
2. The "matlab" folder contains traditional HBF algorithm, channel estimation algorithm and test data set. The "python" file contains the defined neural network models and the trained models
3. The trained models are saved in "python/model". Run "python/main.py" in test mode (train_flag=0), you can test the performance of HBF-Net and CE-HBF-Net.
4. If you want to retrain the HBF-Net and CE-HBF-Net, you can
  a) Run "matlab/gen_traindata.m" to generate training data set
  b) Set "python/main.py" to training mode (train_flag=1), you can retrain the corresponding neural network model
