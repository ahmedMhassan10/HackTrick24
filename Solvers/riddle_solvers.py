# Add the necessary imports here

import pandas as pd
import torch
from utils import *
import numpy as np
from SteganoGAN.utils import *
from torchvision import transforms
from PIL import Image
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.metrics import mean_squared_error


def solve_cv_easy(test_case: tuple) -> list:
    shredded_image, shred_width = test_case
    shredded_image = np.array(shredded_image)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    return []


def solve_cv_medium(test_case: tuple) -> list:
    combined_image_array, patch_image_array = test_case
    combined_image = np.array(combined_image_array, dtype=np.uint8)
    patch_image = np.array(patch_image_array, dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    return []


def solve_cv_hard(test_case: tuple) -> int:
    extracted_question, image = test_case
    image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    return 0


# def preprocess_data(data):
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
#     return scaled_data, scaler
#
#
# def create_dataset(data, look_back=1):
#     X, Y = [], []
#     for i in range(len(data) - look_back):
#         X.append(data[i:(i + look_back), 0])
#         Y.append(data[i + look_back, 0])
#     return np.array(X), np.array(Y)
#
#
# def build_lstm_model(look_back):
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(look_back, 1)))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model


# def solve_ml_easy(data: pd.DataFrame) -> list:
#     # Preprocess data
#     scaled_data, scaler = preprocess_data(data)
#
#     # Split data into train and test sets
#     train_size = int(len(scaled_data) * 0.8)
#     test_size = len(scaled_data) - train_size
#     train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]
#
#     # Reshape into X=t and Y=t+1
#     look_back = 1
#     X_train, Y_train = create_dataset(train, look_back)
#     X_test, Y_test = create_dataset(test, look_back)
#
#     # Reshape input to be [samples, time steps, features]
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
#     # Build LSTM model
#     model = build_lstm_model(look_back)
#
#     # Fit the model
#     model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)
#
#     # Make predictions
#     train_predict = model.predict(X_train)
#     test_predict = model.predict(X_test)
#
#     # Invert predictions
#     train_predict = scaler.inverse_transform(train_predict)
#     Y_train = scaler.inverse_transform([Y_train])
#     test_predict = scaler.inverse_transform(test_predict)
#     Y_test = scaler.inverse_transform([Y_test])
#
#     # Calculate root mean squared error
#     train_score = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
#     test_score = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
#     print('Train RMSE: %.2f' % (train_score))
#     print('Test RMSE: %.2f' % (test_score))
#
#     # Forecast future attacks
#     last_sequence = scaled_data[-look_back:]
#     future_attacks = []
#     for i in range(50):
#         prediction = model.predict(np.reshape(last_sequence, (1, look_back, 1)))
#         future_attacks.append(prediction[0][0])
#         last_sequence = np.append(last_sequence[1:], prediction[0][0])
#
#     # Invert scaling for forecasted data
#     future_attacks = np.array(future_attacks).reshape(-1, 1)
#     future_attacks = scaler.inverse_transform(future_attacks)
#
#     return future_attacks.flatten().tolist()

def solve_ml_easy(input: pd.DataFrame) -> list:
    data = pd.DataFrame(input)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    return []


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    return 0


def solve_sec_medium(input) -> str:
    """
    This function takes an Image as input and returns a string as output.

    Parameters:
    input : An Image

    Returns:
    str: A string representing the decoded message from the image.
    """
    image_array = np.array(input, dtype=np.uint8)
    pil_image = Image.fromarray(image_array, mode='RGB')
    image_tensor = image_to_tensor(pil_image)
    return decode(image_tensor)


def solve_sec_hard(input: tuple) -> str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """

    return ''


def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    my_dict = {}
    x = 0
    for item in input[0]:
        if item not in my_dict:
            my_dict[item] = 1
        else:
            my_dict[item] += 1
    my_list = []
    for key, value in my_dict.items():
        my_list.append([-value, key])
    my_list.sort()
    result = []
    x = int(input[1])
    for item in my_list:
        result.append(item[1])
    return result[:x]


def flatten(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten(item))
        else:
            flattened_list.append(item)
    return flattened_list


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    stack = []
    current_string = ''
    for char in input:
        if char.isdigit():
            current_string += char
        elif char == '[':
            stack.append(current_string)
            stack.append(char)
            current_string = ''
        elif char == ']':

            inner_list = []
            while stack[-1] != '[':
                inner_list.insert(0, stack.pop())
            stack.pop()
            repeat_count = int(stack.pop())
            stack.append(inner_list * repeat_count)
        else:
            stack.append(char)
    res = ''.join(flatten(stack))
    return res


def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    x = int(input[0])
    y = int(input[1])
    dp = np.zeros((x, y))
    dp[0][0] = 1
    for i in range(x):
        for j in range(y):
            if i:
                dp[i][j] += dp[i - 1][j]
            if j:
                dp[i][j] += dp[i][j - 1]
    return int(dp[x - 1][y - 1])


def image_to_tensor(image):
    """
    Converts an image to a PyTorch tensor with batch dimension.
    """
    # Define transformation to apply to the image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Apply the transformation
    tensor = transform(image)

    # Add batch dimension
    tensor_with_batch = tensor.unsqueeze(0)

    return tensor_with_batch


riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard': solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}

# sample test case
path = "D://hackTrick//HackTrick24//SteganoGAN//sample_example//encoded.png"
img = Image.open(path)
print("Decoded Message:", solve_sec_medium(img))

# # Load the CSV file
# data = pd.read_csv("D://hackTrick//HackTrick24//Riddles//ml_easy_sample_example//series_data.csv")
#
# # Print the forecasted attacks
# forecast = solve_ml_easy(data)
# print(forecast)