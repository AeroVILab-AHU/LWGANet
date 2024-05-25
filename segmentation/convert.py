
# LoveDA dataset
# python ./tools/loveda_mask_convert.py --mask-dir /dataset/seg/LoveDA/Train/Rural/masks_png --output-mask-dir /dataset/seg/LoveDA/Train/Rural/masks_png_convert
# python ./tools/loveda_mask_convert.py --mask-dir /dataset/seg/LoveDA/Val/Rural/masks_png --output-mask-dir /dataset/seg/LoveDA/Val/Rural/masks_png_convert
# python ./tools/loveda_mask_convert.py --mask-dir /dataset/seg/LoveDA/Train/Urban/masks_png --output-mask-dir /dataset/seg/LoveDA/Train/Urban/masks_png_convert
# python ./tools/loveda_mask_convert.py --mask-dir /dataset/seg/LoveDA/Val/Urban/masks_png --output-mask-dir /dataset/seg/LoveDA/Val/Urban/masks_png_convert

# UAVid dataset
# train_val
# python tools/uavid_patch_split.py --input-dir "/dataset/seg/uavid/uavid_train_val" --output-img-dir "/dataset/seg/uavid/train_val/images" --output-mask-dir "/dataset/seg/uavid/train_val/masks" --mode 'train' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024
# val
# python tools/uavid_patch_split.py --input-dir "/dataset/seg/uavid/uavid_val" --output-img-dir "/dataset/seg/uavid/val/images" --output-mask-dir "/dataset/seg/uavid/val/masks" --mode 'val' --split-size-h 1024 --split-size-w 1024 --stride-h 1024 --stride-w 1024


# Potsdam
# train
# python tools/potsdam_patch_split.py --img-dir "/dataset/seg/Potsdam/images/train" --mask-dir "/dataset/seg/Potsdam/labels/train" --output-img-dir "/dataset/seg/potsdam/train/images_1024" --output-mask-dir "/dataset/seg/potsdam/train/masks_1024" --mode "train" --split-size 1024 --stride 1024 --rgb-image
# val
# python tools/potsdam_patch_split.py --img-dir "/dataset/seg/Potsdam/images/val" --mask-dir "/dataset/seg/Potsdam/5_Labels_all_noBoundary" --output-img-dir "/dataset/seg/potsdam/val/images_1024" --output-mask-dir "/dataset/seg/potsdam/val/masks_1024" --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
# test
# python tools/potsdam_patch_split.py --img-dir "/dataset/seg/Potsdam/images/test"
# --mask-dir "/dataset/seg/Potsdam/labels/test_eroded"
# --output-img-dir "/dataset/seg/potsdam/test/images_1024"
# --output-mask-dir "/dataset/seg/potsdam/test/masks_1024"
# --mode "val" --split-size 1024 --stride 1024 --eroded --rgb-image
# RGB
# python tools/potsdam_patch_split.py --img-dir "/dataset/seg/Potsdam/images/test"
# --mask-dir "/dataset/seg/Potsdam/labels/test"
# --output-img-dir "/dataset/seg/potsdam/test/images_1024"
# --output-mask-dir "/dataset/seg/potsdam/test/masks_1024_rgb"
# --mode "val" --split-size 1024 --stride 1024 --gt --rgb-image
