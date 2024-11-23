from yacs import CfgNode as CN


__C = CN()

# Model Config
__C.MODEL = CN()
__C.MODEL.dropout = 0.1
__C.MODEL.d_model = 512
__C.MODEL.d_ff = 2048
__C.MODEL.heads = 8
__C.MODEL.flash_attention = False
__C.MODEL.encoder_layers = 6
__C.MODEL.decoder_layers = 6

# Data
__C.DATA = CN()
__C.DATA.source_vocab_size = 32000
__C.DATA.target_vocab_size = 32000
__C.DATA.source_seq_len = 512
__C.DATA.target_seq_len = 512


def get_config():
    return __C.clone()
