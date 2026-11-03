"""StockSense: an LSTM stock market prediction dashboard."""
import os

# Quieten TensorFlow before it is (lazily) imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
