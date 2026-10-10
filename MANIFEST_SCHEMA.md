# Manifest schema

CSV files are UTF-8 and contain one image per row. All paths in the `image` and `mask` columns should resolve on the machine that runs the script.

```csv
sample_id,image,mask
sample_0001,/data/images/sample_0001.png,/data/masks/sample_0001.png
```

For unlabeled target-calibration images, omit the `mask` column or leave its values empty. A labeled source manifest must provide valid masks. Target-validation masks are read only during the evaluation stage described by each runner.

Semantic masks are single-channel integer images with `0=background`, `1=crop`, `2=weed`, and `255=ignore`. Do not remap palette indices unless they already match these IDs.
