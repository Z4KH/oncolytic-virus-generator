# Loss analysis

Run `python analysis/distribution.py` from the project root to extract loss values
from `out.txt`. The script writes `losses.csv` and `loss_distribution_cdf.png`
into this directory and displays the plot. Its default input and output paths
also work when run from another directory.

- `losses.csv`: 1,000 loss values from the saved run.
- `loss_distribution_cdf.png`: cumulative distribution with annotated comparisons.
- `loss_distribution.png`: earlier overview with histogram, box, cumulative,
  and violin plots; the current script does not regenerate this image.

The Influenza A label and generated-sequence comparison value (5882) are
hardcoded annotations in the plotting script.
