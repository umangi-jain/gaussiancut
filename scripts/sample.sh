python3 gaussian-splatting/segment_render.py -m 'path/to/optimized/3dgs/model' \
--scene_path='/path/to/dataset/and/masks' \
--identifier='identifier' \
--mask_type='multiview' \
--foreground_threshold=0.9 \
--select_images='image(s)/name/to/evaluate'