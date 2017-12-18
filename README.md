# cnn_alexnet
Experiment on AlexNet (Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.)

Dataset:
	wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	tar zxvf cifar-10-python.tar.gz
	wget http://www.cmlab.csie.ntu.edu.tw/~jacky82226/research/alexnet_imagenet.npy

		Download and decompress cifar 10 dataset (163MB)
		Download pre-trained AlexNet by imagenet (233MB)

		10 classes: airplance, automobile,
		bird, cat, deer, dog, frog, horse,
		ship, truck

		50000 traning images: 40000 for training and 10000 for validation
		10000 testing images


Training:
	CUDA_VISIBLE_DEVICES=0 python train.py

		(0 means the GPU's id, since tensorflow use all gpu by default)
		--model: use which model to test (deafult model/model_best)
		--batch_size: batch size for training (default 64)
		--lr: learning rate for loss (default 0.00001)
		--test: test or train (default train)

		time: 3mins / epo

Testing:
	CUDA_VISIBLE_DEVICES=0 python train.py --test
	
		--model means use which model to test (deafult model_best)

		time: 45 secs

Environment:
	Software:
	on Debian GNU/Linux testing (stretch)
	1. Python 2.7
	2. tensorflow 0.10.0
	3. numpy 1.12.0
	4. cv2, IPython, cPickle

	Hardware:
	NVIDIA Tesla GPU K80 (4 core and 10GB memory)

	tensorflow installment:
	# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
	pip install --upgrade $TF_BINARY_URL

Description:
  I have tried on oxford 17 first, but I think it is too small to prove the robustness.
  Therefore, for quick demo, I choice cifar10 dataset eventually. 
  It's an appropiate size for proving the alexnet.
  And I use alexnet model which is pretrained by Imagenet for faster converaging and easy fine-tuning. 
  Training code would produce model after epo ending, then we can use testing code to check the accuracy.
  Training code would print epo and loss message on the screen. 
  
  Epo Accuracy
	1 0.21
	2 0.533738057325
	3 0.662320859873
	4 0.711186305732
	5 0.728503184713 
	6 0.748606687898 
	7 0.765525477707
	8 0.765824044586
	9 0.777368630573
 11 0.79090366242
 12 0.796476910828 
 14 0.797969745223 
 16 0.799064490446
 17 0.801652070064

   In my environment, this model achieves accuracy 0.8 in an hour. (3min/epo)
   
   code:
   	train.py: training and testing
   	model.py: model structure of AlexNet
   	data.py: Dataset for random sampling and data format
