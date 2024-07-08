1.Generating image :

CD stylegan3
python3 gen_images.py --outdir=out_test --trunc=1 --seeds=0-4 \
    --network=./stylegan3-r-ffhq-1024x1024.pkl
    
2. Evaluate data set:

CD ..
python3 code/get_csv.py

CD to FairFace
python3 predict.py --csv ../result/test_imgs.csv

CD ..
python3 code/plots_attrib.py

3.dataset editing:

## ALIGNING IMAGE##
CD stylegan3-editing-main
python3 prepare_data/preparing_faces_parallel.py --mode align --root_path ../stylegan3/out_test

##CROPPING IMAGE##
python3 prepare_data/preparing_faces_parallel.py --mode crop --root_path ../stylegan3/out_test --random_shift 0.05

##SAVE FACIAL FEATURE LOCALIZATION##
python3 prepare_data/compute_landmarks_transforms.py --raw_root ../stylegan3/out_test --aligned_root ../stylegan3/out_test_aligned  --cropped_root ../stylegan3/out_test_croped/ --output_root ../result

##GENERATING IMAGE FOR DIFFERENT AGE, POSE AND SMILE##
python3 inversion/scripts/inference_editing.py \
--output_path experiment/output_test \
--checkpoint_path restyle_pSp_ffhq.pt \
--data_path ../stylegan3/out_test_aligned \
--test_batch_size 4 \
--test_workers 4 \
--n_iters_per_batch 3 \
--edit_directions "[age,pose,smile]" \
--factor_ranges "[(-5_5),(-5_5),(-5_5)]" \
--landmarks_transforms_path ../result/landmarks_transforms.npy

#EDITING MORE FEATURE##
editing/interfacegan/edit_synthetic
https://github.com/genforce/interfacegan

4.evaluate model：## HERE YOU NEED TO MODIFY THE IMAGE PATH THAT YOU WANT TO FEED TO THE MODEL TO TRAIN ##

python3 code/finetune.py ##SAVED AT /CHECKPOINTS/##
python3 code/predict.py  ##CHANGE THE MODEL PATH TO DO A PREDICTION WITH THIRD PARTY DATA SET AFTER FINISH IT WILL GENERATED PRED.CSV## 
python3 code/evaluate.py ## CHANGE THE PATH TO THE PRED.CSV FILE ACCORDING WHICH PRED.CSV YOU WANT TO EVALUATE IT WILL GENERATE KPL FILE##
Run plot_result.ipynb  ## CHANGE PATH AND PLOT THE KPL FILE YOU WAN TO PLOT ##

5.Visualize：
python3 anycost-gan-master/demo.py
