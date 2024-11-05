with tf.device(devicename):
    '''
    From https://arxiv.org/pdf/1809.11096.pdf:
    BigGAN-deep is based on residual blocks with bottlenecks (He et al., 2016), which incorporate
    two additional 1 × 1 convolutions: the first reduces the number of channels by a factor of 4 before
    the more expensive 3 × 3 convolutions; the second produces the required number of output chan-
    nels. While BigGAN relies on 1 × 1 convolutions in the skip connections whenever the number of
    channels needs to change, in BigGAN-deep we use a different strategy aimed at preserving identity
    throughout the skip connections. In G, where the number of channels needs to be reduced, we sim-
    ply retain the first group of channels and drop the rest to produce the required number of channels.
    In D, where the number of channels should be increased, we pass the input channels unperturbed,
    and concatenate them with the remaining channels produced by a 1 × 1 convolution. As far as the
    network configuration is concerned, the discriminator is an exact reflection of the generator. There
    are two blocks at each resolution (BigGAN uses one), and as a result BigGAN-deep is four times
    deeper than BigGAN. Despite their increased depth, the BigGAN-deep models have significantly
    fewer parameters mainly due to the bottleneck structure of their residual blocks. For example, the
    128 × 128 BigGAN-deep G and D have 50.4M and 34.6M parameters respectively, while the corre-
    sponding original BigGAN models have 70.4M and 88.0M parameters. All BigGAN-deep models
    use attention at 64 × 64 resolution, channel width multiplier ch = 128, and z ∈ R128.
    ... A ResBlock (without up or down) in BigGAN-deep does not include
    the Upsample or Average Pooling layers, and has identity skip connections.
    ... Embed(y) ∈ R128
    
    Comments:
    It is not clear how one would reduce the number of channels by a factor of 4 from the initial input,
    which only has 3 channels.  I instead interpret this to mean that the first 1x1 convolution in a
    residual block reduces compared to the output depth.

    We scale feature depths linearly rather than exponentially and reduce the total number of layers to
    fit on the GPU with our other models.

    I switched to leaky relus to mitigate node atrophy.  
    '''
    class BigGANDeepResnetBlock(tf.keras.Model):
        def __init__(self, indepth, outdepth, pooling=False):
            super(BigGANDeepResnetBlock, self).__init__(name='')
            self.pooling = pooling
            self.relu = tf.keras.layers.ReLU(negative_slope=0.01)
            #self.conv1a = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(1,1),activation="relu",bias_initializer=tf.keras.initializers.Constant(0.000001)))
            #self.conv3a = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(3,3),padding='same',activation="relu",bias_initializer=tf.keras.initializers.Constant(0.000001)))
            #self.conv3b = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(3,3),padding='same',activation="relu",bias_initializer=tf.keras.initializers.Constant(0.000001)))
            # switch to leaky relus
            self.conv1a = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(1,1),activation=self.relu,bias_initializer=tf.keras.initializers.Constant(0.000001)))
            self.conv3a = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(3,3),activation=self.relu,padding='same',bias_initializer=tf.keras.initializers.Constant(0.000001)))
            self.conv3b = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(indepth//4,(3,3),activation=self.relu,padding='same',bias_initializer=tf.keras.initializers.Constant(0.000001)))
            if pooling:
                self.apool2a = layers.AveragePooling2D((2,2))
            self.conv1b = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(outdepth,(1,1),bias_initializer=tf.keras.initializers.Constant(0.000001)))
            if pooling:
                self.apool2b = layers.AveragePooling2D((2,2))
                self.conv1c = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(outdepth-indepth,(1,1),bias_initializer=tf.keras.initializers.Constant(0.000001)))

        def call(self, input_tensor, training=False):
            x = self.relu(input_tensor)
            x = self.conv1a(x)
            x = self.conv3a(x)
            x = self.conv3b(x)
            if self.pooling:
                x = self.apool2a(x)
            x = self.conv1b(x)
            y = input_tensor
            if self.pooling:
                y = self.apool2b(y)
                y1 = y
                y = self.conv1c(y)
                y = tf.concat([y1,y],axis=-1)
            x += y
            return tf.nn.relu(x)

    
    #with strategy.scope():
    class Discriminator(tf.keras.Model):
        def __init__(self, name="discriminator", **kwargs):
            super(Discriminator, self).__init__(name=name, **kwargs)
            self.ch = 128
            self.edim = 128
            self.conv3 = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(self.ch,(3,3),padding='same',bias_initializer=tf.keras.initializers.Constant(0.000001),input_shape=[224,224,3]))
            self.drb1_2 = BigGANDeepResnetBlock(1*self.ch,2*self.ch,pooling=True)
            self.drb2_2 = BigGANDeepResnetBlock(2*self.ch,2*self.ch)
            self.drb2_3 = BigGANDeepResnetBlock(2*self.ch,3*self.ch,pooling=True)
            self.drb3_3 = BigGANDeepResnetBlock(3*self.ch,3*self.ch)
            self.aquery = layers.Conv1D(3*self.ch//8,1)
            self.avalue = layers.Conv1D(3*self.ch//8,1)
            self.akey = layers.Conv1D(3*self.ch//8,1)
            self.attend = layers.Attention()
            self.aexpand = layers.Conv1D(3*self.ch,1)
            self.ascale = tf.Variable(0.0)
            self.drb3_4 = BigGANDeepResnetBlock(3*self.ch,4*self.ch,pooling=True)
            self.drb4_4 = BigGANDeepResnetBlock(4*self.ch,4*self.ch)
            self.drb4_5 = BigGANDeepResnetBlock(4*self.ch,5*self.ch,pooling=True)
            self.drb5_5 = BigGANDeepResnetBlock(5*self.ch,5*self.ch)
            self.relu = tf.keras.layers.ReLU(negative_slope=0.01)
            self.dense128a = layers.Dense(self.edim) # not clear whether this should include bias
            self.dense128b = layers.Dense(self.edim) # class embedding
            
        def call(self, input_tensor, training=False):
            image = input_tensor["input"]
            anno = input_tensor["anno"]
            x = self.conv3(image)
            x = self.drb1_2(x)
            x = self.drb2_2(x)
            x = self.drb2_3(x)
            x = self.drb3_3(x)
            # attention mechanism with downscaled depth
            y = tf.reshape(x,[-1,56**2,3*self.ch])
            q = self.aquery(y)
            v = self.avalue(y)
            k = self.akey(y)
            a = self.attend([q,v,k])
            v = self.aexpand(a)
            y = v*self.ascale
            y = tf.reshape(x,[-1,56,56,3*self.ch])
            x += y
            #
            x = self.drb3_4(x)
            x = self.drb4_4(x)
            x = self.drb4_5(x)
            x = self.drb5_5(x)
            x = self.relu(x)
            x = tf.reduce_sum(x,axis=[-3,-2]) # over height and width
            x = self.dense128a(x)
            y = self.dense128b(anno)
            x *= y
            x = tf.reduce_sum(x,axis=-1) # over depth
            #x = tf.math.sigmoid(x) # use assumes logit form
            return(x)
