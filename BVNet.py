from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, BatchNormalization



def BVNet(input_size = (256,256, 5)):
    # Build U-Net model
    inputs = Input((input_size))
   # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(64, (3, 3), padding='same') (inputs)
    #c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
   # c1 = Dropout(0.1) (c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(128, (3, 3), padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2,2)) (c1)

#Encode block 2
    c2 = Conv2D(128, (3, 3), padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    #c2 = Dropout(0.1) (c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(256, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2), strides = (2,2)) (c2)

#Encode block 3
    c3 = Conv2D(256, (3, 3), padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    #c3 = Dropout(0.2) (c3)
    c3 = Conv2D(256, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(512, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2,2)) (c3)

# Encode block 4
    c4 = Conv2D(512, (3, 3), padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    #c4 = Dropout(0.2) (c4)
    c4 = Conv2D(512, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(1024, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

#Decode block 1
    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c4)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(512, (3, 3), padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    #c7 = Dropout(0.2) (c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

#Decode block 2
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    #c8 = Dropout(0.1) (c8)
    c8 = Conv2D(256, (3, 3), padding='same') (c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

#Decode block 3
    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(128, (3, 3), padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

#c9 = Dropout(0.1) (c9)
    #c9 = Conv2D(64, (3, 3), padding='same') (c9)
    #c9 = BatchNormalization()(c9)
    #c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss=loss, metrics=[binary_accuracy, dice_coefficient, recall, precision])
    #model.summary()
    return model
