import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. 高级目标检测 - 使用预训练模型
def advanced_object_detection():
    """高级目标检测"""
    # 加载预训练的目标检测模型
    # 注意：这里使用OpenCV的DNN模块加载预训练模型
    # 实际使用时需要下载模型文件
    try:
        # 加载模型
        net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )
        
        # 读取图像
        img = cv2.imread('test_image.jpg')
        if img is None:
            raise FileNotFoundError("测试图像未找到，请替换为真实的图像路径")
    except:
        # 创建一个测试图像
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)
        return img, []
    
    # 预处理图像
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # 检测
    net.setInput(blob)
    detections = net.forward()
    
    # 绘制检测结果
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = f"{confidence:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    print("高级目标检测完成")
    return img, detections

# 2. 高级图像分割 - 使用U-Net
def advanced_image_segmentation():
    """高级图像分割"""
    # 定义U-Net模型
    def unet_model(input_size=(128, 128, 3)):
        inputs = layers.Input(input_size)
        
        # 编码器
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        
        # 瓶颈
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
        
        # 解码器
        u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
        
        u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
        
        u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
        
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
        
        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model
    
    # 创建U-Net模型
    model = unet_model()
    model.summary()
    
    print("U-Net模型创建完成")
    return model

# 3. 视觉Transformer (ViT)
def vision_transformer():
    """视觉Transformer"""
    # 定义Vision Transformer模型
    def vit_model(input_shape=(224, 224, 3), num_classes=1000, patch_size=16, embed_dim=768, num_heads=12, num_layers=12):
        # 输入层
        inputs = layers.Input(shape=input_shape)
        
        # 分块和嵌入
        patches = layers.Reshape((input_shape[0]//patch_size, input_shape[1]//patch_size, patch_size*patch_size*input_shape[2]))(inputs)
        patch_embeddings = layers.Dense(embed_dim)(patches)
        
        # 添加位置编码
        positions = tf.range(start=0, limit=patch_embeddings.shape[1]*patch_embeddings.shape[2], delta=1)
        position_embeddings = layers.Embedding(input_dim=patch_embeddings.shape[1]*patch_embeddings.shape[2], output_dim=embed_dim)(positions)
        position_embeddings = tf.reshape(position_embeddings, (1, patch_embeddings.shape[1], patch_embeddings.shape[2], embed_dim))
        patch_embeddings = patch_embeddings + position_embeddings
        
        # 展平补丁
        patch_embeddings = tf.reshape(patch_embeddings, (tf.shape(patch_embeddings)[0], -1, embed_dim))
        
        # 添加分类标记
        cls_token = layers.Dense(embed_dim)(tf.ones((tf.shape(patch_embeddings)[0], 1, 1)))
        patch_embeddings = tf.concat([cls_token, patch_embeddings], axis=1)
        
        # Transformer编码器
        for _ in range(num_layers):
            # 多头自注意力
            x = layers.LayerNormalization(epsilon=1e-6)(patch_embeddings)
            x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
            x = layers.Add()([x, patch_embeddings])
            
            # 前馈网络
            patch_embeddings = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dense(embed_dim * 4, activation='relu')(patch_embeddings)
            x = layers.Dense(embed_dim)(x)
            patch_embeddings = layers.Add()([x, patch_embeddings])
        
        # 分类头
        x = layers.LayerNormalization(epsilon=1e-6)(patch_embeddings)
        x = x[:, 0, :]  # 取分类标记
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    # 创建ViT模型
    model = vit_model(input_shape=(224, 224, 3), num_classes=10)
    model.summary()
    
    print("Vision Transformer模型创建完成")
    return model

# 4. 注意力机制 - SENet
def senet_block():
    """SENet注意力机制"""
    def se_block(input_tensor, ratio=16):
        channels = input_tensor.shape[-1]
        
        # 挤压操作
        x = layers.GlobalAveragePooling2D()(input_tensor)
        x = layers.Reshape((1, 1, channels))(x)
        
        # 激励操作
        x = layers.Dense(channels // ratio, activation='relu')(x)
        x = layers.Dense(channels, activation='sigmoid')(x)
        
        # 缩放操作
        output = layers.Multiply()([input_tensor, x])
        return output
    
    # 创建包含SE块的模型
    inputs = layers.Input(shape=(224, 224, 3))
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = se_block(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    
    print("SENet模型创建完成")
    return model

# 5. 生成对抗网络 (GAN)
def gan_model():
    """生成对抗网络"""
    # 生成器
    def generator(latent_dim):
        model = models.Sequential()
        model.add(layers.Dense(7*7*128, input_dim=latent_dim))
        model.add(layers.Reshape((7, 7, 128)))
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu'))
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu'))
        model.add(layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        return model
    
    # 判别器
    def discriminator(input_shape=(28, 28, 1)):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model
    
    # 构建GAN
    latent_dim = 100
    
    # 创建判别器
    disc = discriminator()
    disc.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 创建生成器
    gen = generator(latent_dim)
    
    # 创建GAN模型
    disc.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = disc(gen(gan_input))
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    print("GAN模型创建完成")
    return gen, disc, gan

# 6. 迁移学习
def transfer_learning():
    """迁移学习"""
    # 加载预训练的ResNet50模型
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # 冻结基础模型
    for layer in base_model.layers:
        layer.trainable = False
    
    # 添加自定义分类头
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    predictions = layers.Dense(10, activation='softmax')(x)
    
    # 创建完整模型
    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    
    print("迁移学习模型创建完成")
    return model

# 7. 视频处理基础
def video_processing_basics():
    """视频处理基础"""
    # 打开视频文件或摄像头
    try:
        cap = cv2.VideoCapture('test_video.mp4')
        if not cap.isOpened():
            raise FileNotFoundError("测试视频未找到，请替换为真实的视频路径")
    except:
        # 使用摄像头
        cap = cv2.VideoCapture(0)
    
    # 读取视频帧
    ret, frame = cap.read()
    if ret:
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 100, 200)
        
        # 释放资源
        cap.release()
        
        print("视频处理基础操作完成")
        return frame, gray, edges
    else:
        # 创建测试图像
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.putText(frame, "Video Processing", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cap.release()
        print("视频处理基础操作完成")
        return frame, frame, frame

# 8. 多模态学习 - 图像描述生成
def image_captioning():
    """图像描述生成"""
    # 定义图像编码器
    def image_encoder(input_shape=(224, 224, 3)):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        model = models.Model(inputs=base_model.input, outputs=x)
        return model
    
    # 定义文本解码器
    def text_decoder(vocab_size, max_length):
        image_features = layers.Input(shape=(256,))
        sequence_input = layers.Input(shape=(max_length,))
        
        # 词嵌入
        embedding = layers.Embedding(vocab_size, 256, mask_zero=True)(sequence_input)
        
        # LSTM
        lstm = layers.LSTM(256, return_sequences=True)(embedding)
        
        # 注意力机制
        attention = layers.dot([lstm, image_features], axes=[2, 1])
        attention = layers.Activation('softmax')(attention)
        context = layers.dot([attention, lstm], axes=[1, 1])
        
        # 融合特征
        combined = layers.concatenate([context, image_features])
        outputs = layers.Dense(vocab_size, activation='softmax')(combined)
        
        model = models.Model(inputs=[sequence_input, image_features], outputs=outputs)
        return model
    
    # 创建模型
    vocab_size = 10000
    max_length = 30
    
    encoder = image_encoder()
    decoder = text_decoder(vocab_size, max_length)
    
    print("图像描述生成模型创建完成")
    return encoder, decoder

# 主函数
if __name__ == "__main__":
    print("=== 计算机视觉进阶示例 ===")
    
    # 1. 高级目标检测
    print("\n1. 高级目标检测")
    try:
        detection_img, detections = advanced_object_detection()
        print("高级目标检测操作成功")
    except Exception as e:
        print(f"高级目标检测操作失败: {e}")
    
    # 2. 高级图像分割
    print("\n2. 高级图像分割")
    try:
        unet_model = advanced_image_segmentation()
        print("U-Net模型创建成功")
    except Exception as e:
        print(f"U-Net模型创建失败: {e}")
    
    # 3. 视觉Transformer
    print("\n3. 视觉Transformer")
    try:
        vit_model = vision_transformer()
        print("Vision Transformer模型创建成功")
    except Exception as e:
        print(f"Vision Transformer模型创建失败: {e}")
    
    # 4. SENet注意力机制
    print("\n4. SENet注意力机制")
    try:
        senet_model = senet_block()
        print("SENet模型创建成功")
    except Exception as e:
        print(f"SENet模型创建失败: {e}")
    
    # 5. GAN模型
    print("\n5. GAN模型")
    try:
        gen, disc, gan = gan_model()
        print("GAN模型创建成功")
    except Exception as e:
        print(f"GAN模型创建失败: {e}")
    
    # 6. 迁移学习
    print("\n6. 迁移学习")
    try:
        transfer_model = transfer_learning()
        print("迁移学习模型创建成功")
    except Exception as e:
        print(f"迁移学习模型创建失败: {e}")
    
    # 7. 视频处理基础
    print("\n7. 视频处理基础")
    try:
        frame, gray, edges = video_processing_basics()
        print("视频处理操作成功")
    except Exception as e:
        print(f"视频处理操作失败: {e}")
    
    # 8. 图像描述生成
    print("\n8. 图像描述生成")
    try:
        encoder, decoder = image_captioning()
        print("图像描述生成模型创建成功")
    except Exception as e:
        print(f"图像描述生成模型创建失败: {e}")
    
    print("\n=== 计算机视觉进阶示例完成 ===")
