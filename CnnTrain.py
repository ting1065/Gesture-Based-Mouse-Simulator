import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
df = pd.read_csv('keypoints.csv')
X = df.iloc[:, :-1].values  # Keypoint data
y = df.iloc[:, -1].values  # Class labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape the data to fit the model
X_train = X_train.reshape(-1, 21, 2, 1)   # Assuming there are 21 keypoints, each with x and y coordinates
X_test = X_test.reshape(-1, 21, 2, 1)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(1, 2), activation='relu', input_shape=(21, 2, 1)),  # Adjusted kernel size
    MaxPooling2D(pool_size=(2, 1)),  # 调整了池化层的尺寸
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax')
])


model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

# Save the model
model.save('hand_gesture_model.h5')
