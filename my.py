from tensorflow.keras.models import load_model

model = load_model(r"C:\Users\91772\plani_pro\potatoes_v3.h5")
print(model.summary())
print("Expected input shape:", model.input_shape)
