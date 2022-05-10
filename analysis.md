# Analysis
Forecasting day-ahead realized variance:
1. HAR Model.
2. HARQ Model
3. (V)FARIMA
4. Attention-based stacked model. Attention-based autoencoder and HAR model.
5. Transformer-based end-to-end model, where each day is encoded as a word and the prediction is the realized variance.

If we resample the ticks to 5-minutes, see the article in the gh repo, we have 288 daily observations, which fits neatly with the architecture of a transformer model.
