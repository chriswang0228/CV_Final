arch   = 'UnetPlusPlus'  #arch_list=['Unet','MAnet','UnetPlusPlus','DeepLabV3Plus'] if arch   = 'DeepLabV3Plus': 
                            #train.py line99 model = smp.create_model(arch=cfg.arch,encoder_name=cfg.encoder, encoder_weights='imagenet', classes=n_classes, activation=None, encoder_depth=5)
epochs     = 50           
batch_size = 3         
encoder = 'se_resnet101'
