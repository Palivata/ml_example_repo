initial:
	pip install pre-commit
	pre-commit install
install:
	docker compose -f ./docker/docker-compose.yaml build
train:
	docker compose -f ./docker/docker-compose.yaml run trainer scripts/train.sh configs/config_ocr.yaml

convert_to_ov:
	docker compose -f ./docker/docker-compose.yaml run trainer scripts/convert.sh configs/config_ocr.yaml experiments/ocr_baseline/epoch_epoch=08-valid_ctc_loss=0.275.ckpt models/

infer_current_model:
	docker compose -f ./docker/docker-compose.yaml run trainer scripts/infer.sh examples/test.jpg configs/config_segmentation.yaml
