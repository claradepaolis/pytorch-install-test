# pytorch-install-test
Scripts to test installation of GPU-enabled Pytorch and tensorboard

These files are meant to test a pytorch installation, for example when creating a new environment or 
after updating firmware etc. 

`pytorch_test.py` test a neural network training on CPU only.  
`pytorch_test_gpu.py` test a neural network training on GPU.  
`pytorch_test_tensorboard.py` test a tensorboard functionality in pytorch. 
Make sure to launch the tensorboard server before executing. More details here: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html 
