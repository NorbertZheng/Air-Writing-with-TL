clean:
	rm -rf ./__pycache__
	cd cnnblstm && make clean
	cd cnnblstm_with_adabn && make clean
	cd finetune && make clean
	cd network && make clean


