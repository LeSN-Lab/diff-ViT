CUDA_VISIBLE_DEVICES=0 python test_quant.py deit_base /home/shared_data/imagenet --quant --ptf --lis --quant-method minmax --mode 0
# CUDA_VISIBLE_DEVICES=0 python test_quant.py vit_base /home/shared_data/imagenet --quant --quant-method minmax --mode 2