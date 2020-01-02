# Deep_SBIR_tf
Implementation of **Sketch Me That Shoe** (http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html) and **Deep Spatial-Semantic Attention for Fine-grained Sketch-based Image Retrieval**

Modified from **Sketch Me That Shoe** (https://github.com/seuliufeng/DeepSBIR) and its **TensorFlow implementation**:  https://github.com/yuchuochuo1023/Deep_SBIR_tf/tree/master

All dataset and models are included

**Steps**:

(1) Download code and install dependencies, now only run on shoes prediction and training

(2) Run triplet_sbir_train.py which will start training process, the default epoch is 200, the output is the trained model: model-iter***.npy together with the model-iter0.npy

(3) Run triplet_sbir_test.py which will start the prediction process, store the test pic for sketches in the source folder and type in the name in the source code. The prediction will run on all models in the model/shoes/deep_sbir. The top 10 predictions will be shown.


**Requirements**


python: 3.6.5;
pip: 19.3.1;
tensorflow: 2.0.0;
numpy: 1.16.1;
scikit-learn: 0.22;
keras: 2.3.1;
cv2: 4.1.2;



**Note**: 

(1) Please download the **pre-trained** model and **well-trained** models via https://drive.google.com/open?id=19LSQ5fbVVr3bPvE5Xuxd1zeMy1Y9ozC7;

(2) The dataset for shoes can be found here: http://www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/sbir_cvpr2016.tar

(3) Implementation by **Caffe** can be found here: https://github.com/seuliufeng/DeepSBIR.



And if you use the code for your research, please cite these papers:

    @inproceedings{yu2016sketch,
            title={Sketch me that shoe},
            author={Yu, Qian and Liu, Feng and Song, Yi-Zhe and Xiang, Tao and Hospedales, Timothy M and Loy, Chen Change},
            booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
            pages={799--807},
            year={2016},
            organization={IEEE}
    }

    @article{yu2017sketch,
            title={Sketch-a-net: A deep neural network that beats humans},
            author={Yu, Qian and Yang, Yongxin and Liu, Feng and Song, Yi-Zhe and Xiang, Tao and Hospedales, Timothy M},
            journal={International Journal of Computer Vision},
            volume={122},
            number={3},
            pages={411--425},
            year={2017},
            publisher={Springer}
    }
    
