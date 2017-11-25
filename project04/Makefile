preprocess: util/preprocess.py
	python $<

train: train_model.py
	CUDA_VISIBLE_DEVICES=0 python $<

predict: predict.py
	CUDA_VISIBLE_DEVICES=0 python $<

clean:
	rm -rf .model_checkpts
	mkdir -p .model_checkpts
