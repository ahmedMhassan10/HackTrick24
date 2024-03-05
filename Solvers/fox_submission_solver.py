import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers
import random

# team_id = "Jt4hTHH"
team_id = "a3333333"
# 3.70.97.142
api_base_url = "http://3.70.97.142:5000/fox"


def init_fox(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    payload = {
        "teamId": team_id
    }

    response = requests.post(api_base_url + f'/start', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        response_data = response.json()
        message = response_data['msg']
        image_carrier = response_data['carrier_image']
        return message, image_carrier
    pass


def generate_message_array(message, image_carrier):
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''
    real_message_chunks = []
    length = int(len(message) / 3)
    for i in range(2):
        real_message_chunks.append(message[i * length: (i + 1) * length])
    real_message_chunks.append(message[2 * length:])
    fake_message_chunks = []
    for i in range(3):
        temp_fake_array = []
        for j in range(2):
            temp_fake_array.append("dell is back")
        fake_message_chunks.append(temp_fake_array)
    list_of_chunks = []
    for i in range(3):
        chunk_set = [
            [real_message_chunks[i], "R"],
            [fake_message_chunks[i][0], "F"],
            [fake_message_chunks[i][1], "F"]
        ]
        random.shuffle(chunk_set)
        list_of_chunks.append(chunk_set)
    images = []
    images_entities = []
    for i in range(len(list_of_chunks)):
        temp = []
        image_entity = []
        for j in range(len(list_of_chunks[i])):
            mat = image_carrier.copy()
            temp.append(encode(mat, list_of_chunks[i][j][0]))
            image_entity.append(list_of_chunks[i][j][1])
        images.append(temp)
        images_entities.append(image_entity)
    return images, images_entities


def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that:
        1. Once you requested a riddle you cannot request it again per game.
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle.
    '''
    payload = {
        "teamId": team_id,
        "riddleId": riddle_id
    }

    response = requests.post(api_base_url + f'/get-riddle', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        response_data = response.json()
        test_case = response_data['test_case']
        return test_case
    pass
    # if riddle_id == "problem_solving_easy":
    #     return ("faris", "faris", "ahmed", 1)
    # elif riddle_id == "problem_solving_medium":
    #     return "2[ab]2[cd]"
    # elif riddle_id == "problem_solving_hard":
    #     return (2, 2)


def riddle_runner(riddle_id, test_case):
    func = riddle_solvers[riddle_id]
    return func(test_case)


def solve_riddle(team_id, solution):
    '''
    In this function you will solve the riddle that you have requested.
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    payload = {
        "teamId": team_id,
        "solution": solution
    }
    response = requests.post(api_base_url + f'/solve-riddle', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        response_data = response.json()
        budget_increase = response_data['budget_increase']
        total_budget = response_data['total_budget']
        status = response_data['status']
        return budget_increase, total_budget, status
    # return 1, 1, 1
    pass


def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message.
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call.
    '''
    payload = {
        "teamId": team_id,
        "messages": messages,
        "message_entities": message_entities
    }
    response = requests.post(api_base_url + f'/send-message', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        response_data = response.json()
        status = response_data['status']
        return status
    pass


def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    payload = {
        "teamId": team_id,
    }
    response = requests.post(api_base_url + f'/end-game', json=payload)
    print(response.text)


def submit_fox_attempt(team_id):
    '''
     Call this function to start playing as a fox.
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox
        2. Solve riddles
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling
    '''
    message, image_carrier = init_fox(team_id)
    image_carrier = np.array(image_carrier)
    message_array, message_entities = generate_message_array(message, image_carrier)
    riddles_list = ["problem_solving_easy", "problem_solving_medium", "problem_solving_hard", 'sec_medium_stegano', 'ml_easy', 'sec_hard', 'ml_medium']
    total = 0
    for riddle_id in riddles_list:
        test_case = get_riddle(team_id, riddle_id)
        solution = riddle_runner(riddle_id, test_case)
        budget_increase, total, status = solve_riddle(team_id, solution)

    for i in range(len(message_array)):
        for j in range(len(message_array[i])):
            message_array[i][j] = message_array[i][j].tolist()
        status = send_message(team_id, message_array[i], message_entities[i])
    end_fox(team_id)


submit_fox_attempt(team_id)
