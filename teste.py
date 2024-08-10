from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

# Carregar o modelo salvo
model = load_model('model.keras')

# Caminhos para as imagens que você deseja testar
image_paths = [f'teste{i}.jpeg' for i in range(1, 7)]  # Gera os nomes 'teste1.jpeg' até 'teste6.jpeg'

for image_path in image_paths:
    # Carregar e processar a imagem
    img = image.load_img(image_path, target_size=(64, 64))  # Certifique-se de que o target_size corresponde ao im_shape usado no treinamento
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para o batch
    img_array /= 255.0  # Normaliza a imagem

    # Fazer a predição
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = classes[predicted_class[0]]
    
    # Mostrar o resultado
    print(f'Imagem: {image_path}. Classe prevista: {predicted_label}')
    print('-------------------------------------')


