"""Explicit model adapters: an architecture must be registered before launch."""
from types import SimpleNamespace

IMPLEMENTED = ('FITS',)


def build_model(name, context, horizon, cutoff):
    if name != 'FITS':
        raise ValueError(f'{name} has no implemented foundation pretraining adapter')
    from models.FITS import Model
    if (context + horizon) % 2:
        raise ValueError('current FITS irFFT implementation requires even output length')
    if not 1 <= cutoff <= context // 2 + 1:
        raise ValueError('cutoff exceeds context rFFT bins')
    if int(cutoff * (context + horizon) / context) > (context + horizon) // 2 + 1:
        raise ValueError('interpolated cutoff exceeds output rFFT bins')
    return Model(SimpleNamespace(seq_len=context, pred_len=horizon, enc_in=1,
                                 individual=False, cut_freq=cutoff))


def forecast(model, context, horizon):
    output, _ = model(context)
    return output[:, -horizon:]
