# Add the necessary imports here

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering
from statsmodels.tsa.arima.model import ARIMA
from torchvision import transforms
from SteganoGAN.utils import *

init_p = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

pc1_p = [
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
]

shift_table = [
    1, 1, 2, 2,
    2, 2, 2, 2,
    1, 2, 2, 2,
    2, 2, 2, 1
]

pc2_p = [
    14, 17, 11, 24, 1, 5,
    3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8,
    16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]

exp_d = [
    32, 1, 2, 3, 4, 5, 4, 5,
    6, 7, 8, 9, 8, 9, 10, 11,
    12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21,
    22, 23, 24, 25, 24, 25, 26, 27,
    28, 29, 28, 29, 30, 31, 32, 1
]

# S-box Table
sbox = [
    # S-box 1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
    ],
    # S-box 2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
    ],
    # S-box 3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
    ],
    # S-box 4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
    ],
    # S-box 5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
    ],
    # S-box 6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
    ],
    # S-box 7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
    ],
    # S-box 8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
    ]
]

per = [16, 7, 20, 21,
       29, 12, 28, 17,
       1, 15, 23, 26,
       5, 18, 31, 10,
       2, 8, 24, 14,
       32, 27, 3, 9,
       19, 13, 30, 6,
       22, 11, 4, 25]

ip_inverse = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
]


def hex2bin(s):
    m = {'0': "0000", '1': "0001", '2': "0010", '3': "0011", '4': "0100", '5': "0101", '6': "0110",
         '7': "0111", '8': "1000", '9': "1001", 'A': "1010", 'B': "1011", 'C': "1100", 'D': "1101", 'E': "1110",
         'F': "1111"}
    b = ""
    for i in range(len(s)):
        b = b + m[s[i]]
    return b


def bin2hex(s):
    m = {"0000": '0', "0001": '1', "0010": '2', "0011": '3', "0100": '4', "0101": '5',
         "0110": '6', "0111": '7', "1000": '8', "1001": '9', "1010": 'A', "1011": 'B', "1100": 'C', "1101": 'D',
         "1110": 'E', "1111": 'F'}
    h = ""
    for i in range(0, len(s), 4):
        c = ""
        for j in range(4):
            c = c + s[i + j]
        h = h + m[c]

    return h


def bin2dec(binary):
    decimal, i = 0, 0
    binary_str = str(binary)

    for bit in reversed(binary_str):
        decimal += int(bit) * (1 << i)
        i += 1

    return decimal


def dec2bin(num):
    binary_representation = bin(num)[2:]
    binary_length = len(binary_representation)

    padding_zeros = (4 - (binary_length % 4)) % 4
    binary_result = '0' * padding_zeros + binary_representation
    return binary_result


def permute(binary_rep, permutation_f, n):
    permutation = ""
    for i in range(n):
        permutation = permutation + binary_rep[permutation_f[i] - 1]
    return permutation


def shift_left(k, shamt):
    return k[shamt:] + k[:shamt]


def xor(a, b):
    return ''.join('0' if x == y else '1' for x, y in zip(a, b))


def get_round_key(k):
    k = hex2bin(k)
    k = permute(k, pc1_p, 56)
    left = k[0:28]
    right = k[28:56]
    round_key_binary = []
    round_key = []

    for i in range(0, 16):
        left = shift_left(left, shift_table[i])
        right = shift_left(right, shift_table[i])
        combine_str = left + right

        r_k = permute(combine_str, pc2_p, 48)
        round_key_binary.append(r_k)
        round_key.append(bin2hex(r_k))

    return round_key_binary, round_key


def encrypt(plain_text, key):
    plain_text = hex2bin(plain_text)
    plain_text = permute(plain_text, init_p, 64)

    round_key_binary, round_key = get_round_key(key)

    left = plain_text[:32]
    right = plain_text[32:]

    for i in range(16):

        expanded = permute(right, exp_d, 48)
        xx = xor(expanded, round_key_binary[i])

        sbox_sub = ""
        for j in range(0, 48, 6):
            row_bits = xx[j] + xx[j + 5]
            col_bits = xx[j + 1: j + 5]
            row = bin2dec(int(row_bits))
            col = bin2dec(int(col_bits))
            val = sbox[j // 6][row][col]
            sbox_sub += dec2bin(val)

        sbox_sub = permute(sbox_sub, per, 32)

        result = xor(left, sbox_sub)
        left = result

        left, right = (right, left) if i < 15 else (left, right)

    combine = left + right
    return permute(combine, ip_inverse, 64)


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

    # print(data.head())
    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    train_data = data['visits'].values
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=50).tolist()


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    print(input)
    filename = "D://hackTrick//HackTrick24//Riddles//ml_medium_dataset//MlMediumTrainingData.csv"
    data = pd.read_csv(filename)
    X = data[['x_', 'y_']]  # Features
    y = data['class']  # Labels# Labels

    X._append(input)
    spectral_model = SpectralClustering(n_clusters=2,
                                        affinity='nearest_neighbors')  # Adjust the number of clusters as needed
    labels = spectral_model.fit_predict(X)

    # print("Predicted label:", predicted_label)
    predicted_label = labels[len(labels) - 1]

    print(predicted_label)
    return int(predicted_label) - 1


def solve_sec_medium(input) -> str:
    """
    This function takes an Image as input and returns a string as output.

    Parameters:
    input : An Image

    Returns:
    str: A string representing the decoded message from the image.
    """
    # print(len(input))
    input = np.array(input)
    # print(input.shape)
    image_tensor = torch.tensor(input)  # Convert list to tensor
    image_tensor = image_tensor.float()
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
    return bin2hex(encrypt(input[1], input[0]))


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

# Given input with one channel
# input_tensor = np.array([[
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ],
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]
#     ]
# ]])


# # # # # Load the CSV file
# filename = "D://hackTrick//HackTrick24//Riddles//ml_medium_dataset//MlMediumTrainingData.csv"
# # # data = pd.read_csv(filename, parse_dates=['timestamp'], index_col='timestamp')
# # data = pd.read_csv(filename)
# # #
# # # # Print the forecasted attacks
# # forecast = solve_ml_easy(data)
# # print("forecast")
# # print(forecast)
# # # Calculate RMSE
# # true_output = [2.0, 12.0, 13.0, 1.0, 10.0, 8.0, 6.0, 6.0, 6.0, 8.0, 9.0, 6.0, 1.0, 12.0, 11.0, 7.0, 5.0, 12.0, 8.0, 7.0, 15.0, 11.0, 9.0, 7.0, 9.0, 6.0, 8.0, 13.0, 8.0, 5.0, 12.0, 10.0, 7.0, 7.0, 11.0, 7.0, 14.0, 11.0, 3.0, 10.0, 5.0, 14.0, 8.0, 11.0, 8.0, 24.0, 10.0, 15.0, 12.0, 11.0]
# # rmse = sqrt(mean_squared_error(true_output, forecast))
# # print(f"RMSE: {rmse:.2f}")
#
# # Load the data
# data = pd.read_csv(filename)
# #
# # # Separate features and labels
# X = data[['x_', 'y_']]
# y = data['class']
#
# # # Instantiate DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# #
# # # Fit DBSCAN to the data
# dbscan.fit(X)
# #
# # # Function to classify a new sample point
# def classify_new_point(new_point):
#     # Reshape the new point into a 2D array
#     new_point_reshaped = new_point.reshape(1, -1)
#     # Predict the label of the new point
#     predicted_label = dbscan.predict(new_point_reshaped)
#     return predicted_label[0]
#
# # # Example usage:
# sample_point = (0,0)  # Example sample point
# predicted_class = classify_new_point(sample_point)

# print(solve_ml_medium([0, 0]))
# sample = [7.504178776, 10.53795112]
# print(solve_ml_medium(sample))


# # Assuming X and Y are arrays of x and y coordinate points
# X = np.random.uniform(low=-1, high=1, size=(1000, 100))
# Y = np.random.uniform(low=-1, high=1, size=(1000, 100))
#
# # Combine X and Y coordinates into a single feature vector
# # Each row represents a single point with x and y coordinates
# features = np.column_stack((X.flatten(), Y.flatten()))
#
# # Labels for the points (replace with your actual labels)
# labels = np.random.randint(0, 2, size=len(features))
#
# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
#
# # Define CNN model architecture
# model = models.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(2,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test, y_test))
#
# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print("Test Accuracy:", test_accuracy)
#
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, models
#
# # Load the data
# filename = "/content/MlMediumTrainingData.csv"
# data = pd.read_csv(filename)
#
# # Extract features and labels
# X = data[['x_', 'y_']].values
# labels = data['class'].values
#
# # Map -1 labels to +1
# labels[labels == -1] = 1
#
# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
#
# # Define a fully connected model
# model = models.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(2,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
#
# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print("Test Accuracy:", test_accuracy)

# Example usage
# new_point = [0, 0]  # New point to classify
# print(solve_ml_medium(new_point))
# predicted_label = predict_label(new_point)
