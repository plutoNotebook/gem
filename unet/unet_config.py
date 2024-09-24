from unet.unet import BeatGANsUNetConfig, BeatGANsEncoderConfig

unet_config = BeatGANsUNetConfig(
    image_size = 84, 
    in_channels = 3,
    model_channels = 192,
    out_channels = 3,
    num_res_blocks = 3,
    num_input_res_blocks = None, # downsampling num_resblock
    embed_channels = 512, # embedding's channel
    attention_resolutions = (42, 21, 10),
    time_embed_channels = None, # time_embedding's output_channel -> embedding_channel projected
    dropout = 0.1,
    channel_mult = [1, 2, 3, 4],
    input_channel_mult = None, # downsampling's channel mult
    conv_resample = True, # ud sample by convolution
    dims = 2,
    num_classes = None,
    use_checkpoint = True, 
    num_heads = 1, # num attention head
    num_head_channels = 64, # num_channel per head
    num_heads_upsample = -1, # ?? not used 
    resblock_updown = True,
    use_new_attention_order = False, 
    resnet_two_cond = False, 
    resnet_cond_channels = None, # condition emb channel
    resnet_use_zero_module = True, 
    attn_checkpoint = True 
)

encoder_config = BeatGANsEncoderConfig(
    image_size = 84,
    in_channels = 3,
    model_channels = 192,
    out_hid_channels = 768,
    out_channels = 768,
    num_res_blocks = 3,
    attention_resolutions = (42, 21, 10),
    dropout = 0.1,
    channel_mult = [1, 2, 3, 4],
    use_time_condition = False, 
    conv_resample = True,
    dims = 2,
    use_checkpoint = True,
    num_heads = 1,
    num_head_channels = 64,
    resblock_updown = True,
    use_new_attention_order = False,
    pool = 'adaptivenonzero'
)


