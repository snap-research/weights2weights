python inversion/invert.py \
    --device="cuda:0"  \
    --mean_path="files/mean.pt" \
    --std_path="files/std.pt" \
    --v_path="files/V.pt" \
    --std_path="files/std.pt" \
    --dim_path="files/weight_dimensions.pt" \
    --imfolder="inversion/images/real_image/real/" \
    --mask_path="inversion/images/real_image/mask.png" \
    --epochs=400 \
    --lr=1e-1 \
    --weight_decay=1e-10 \
    --dim=10000 \
    --save_name="files/real_inversion1.pt"




            
            
            
          