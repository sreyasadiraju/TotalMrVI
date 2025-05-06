# TOTALMRVI Project

This project contains the PyTorch module `TOTALMRVAE` for modeling CITE-seq data,
inspired by MrVI and TOTALVI.

## Package Structure

- `totalmrvi/`: The main Python package.
  - `components.py`: Core neural network building blocks (ConditionalLayerNorm, MLP, AttentionBlock).
  - `encoders.py`: Encoder modules (EncoderXYU, EncoderUZ, BackgroundProteinEncoder).
  - `decoders.py`: Decoder modules (DecoderZXAttention, ProteinDecoderZYAttention).
  - `module.py`: The main `TOTALMRVAE` nn.Module class.
  - `model.py`: (Placeholder for the scvi-tools `TOTALMRVI` model class).
- `tests/`: Unit and integration tests for the package.

## Installation

To install the `totalmrvi` for development:

```bash
cd totalmrvi
pip install -e .