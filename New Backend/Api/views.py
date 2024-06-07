from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians
from django.http import request
from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import permissions
from rest_framework import mixins
from rest_framework import generics
from .serializers import Contactserializer,Carrypicserializer ,HClientcarpoolserializer,HDrivercarpoolserializer, Voucherserializer, Carpoolreqserializer, Finalcarpoolserializer, PrivateRoomserializer
from .serializers import Customerserializer, Driverserializer, Clientserializer, Carpoolserializer, DCarpoolreqserializer
from .models import Contact, Carpool, PrivateRoom, Voucher,History_Driver,History_Client,Carrypic
from .models import Customer, Client_location, Driver_location, DCarpool_request, Carpool_request, Final_Carpool
from twilio.rest import Client
from rest_framework.renderers import JSONRenderer
import random
from geopy import distance, Point
from django.core.management.base import BaseCommand
import math
import datetime
import qrcode
from PIL import Image
from pathlib import Path
import os
from io import BytesIO


# import nltk
# from nltk.stem import WordNetLemmatizer
# import json
# import pickle
# import numpy as np
# import random

# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
# from keras.models import load_model
# #--------------Chatbot-------------
# lemmatizer = WordNetLemmatizer()
# words=[]
# classes = []
# documents = []
# ignore_words = ['?', '!']
# data_file = open(r"D:\Carpool App\Backend\Api\intents.json").read()
# intents = json.loads(data_file)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# for intent in intents['intents']:
#     for pattern in intent['patterns']:

#         #tokenize each word
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         #add documents in the corpus
#         documents.append((w, intent['tag']))

#         # add to our classes list
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))

# # sort classes
# classes = sorted(list(set(classes)))

# # documents = combination between patterns and intents
# print (len(documents), "documents")

# # classes = intents
# print (len(classes), "classes", classes)

# # words = all words, vocabulary
# print (len(words), "unique lemmatized words", words)

# # creating a pickle file to store the Python objects which we will use while predicting
# pickle.dump(words,open('words.pkl','wb'))
# pickle.dump(classes,open('classes.pkl','wb'))
# # create our training data
# training = []

# # create an empty array for our output
# output_empty = [0] * len(classes)

# # training set, bag of words for each sentence
# for doc in documents:
#     # initialize our bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = doc[0]

#     # lemmatize each word - create base word, in attempt to represent related words
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

#     # create our bag of words array with 1, if word match found in current pattern
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
#     # output is a '0' for each tag and '1' for current tag (for each pattern)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])

# # shuffle features and converting it into numpy arrays
# random.shuffle(training)
# training = np.array(training,dtype=object)

# # create train and test lists

# train_x = list(training[:,0])
# train_y = list(training[:,1])

# # Create NN model to predict the responses

# print("Training data created"+f"{training}")

# model = Sequential()

# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# #fitting and saving the model
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# model.save('chatbot.h5', hist)
# # we will pickle this model to use in the future

# print("\n")
# print("*"*50)
# print("\nModel Created Successfully!")
# # load the saved model file
# model = load_model('chatbot.h5')
# intents = json.loads(open(r"D:\Carpool App\Backend\Api\intents.json").read())
# words = pickle.load(open('words.pkl','rb'))
# classes = pickle.load(open('classes.pkl','rb'))

# def clean_up_sentence(sentence):

#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)

#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def bow(sentence, words, show_details=True):

#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)

#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s:

#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#                 if show_details:
#                     print ("found in bag: %s" % w)
#     return(np.array(bag))

# def predict_class(sentence, model):

#     # filter out predictions below a threshold
#     p = bow(sentence, words,show_details=False)
#     res = model.predict(np.array([p]))[0]
#     error = 0.25
#     results = [[i,r] for i,r in enumerate(res) if r>error]

#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []

#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list
#     # function to get the response from the model

# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result

# def chatbot_response(text):
#     ints = predict_class(text, model)
#     res = getResponse(ints, intents)
#     return res

# chatbot_response(data)

class ChatCheck(APIView):
    def get(self, request, data):
        return Response({"Replay": data, "status": status.HTTP_200_OK})


class Contactviewset(viewsets.ModelViewSet):
    queryset = Contact.objects.all()
    serializer_class = Contactserializer


class FinalCarpoolviewset(viewsets.ModelViewSet):
    queryset = Final_Carpool.objects.all()
    serializer_class = Finalcarpoolserializer


class Clientviewset(viewsets.ModelViewSet):
    queryset = Client_location.objects.all()
    serializer_class = Clientserializer


class Driverviewset(viewsets.ModelViewSet):
    queryset = Driver_location.objects.all()
    serializer_class = Driverserializer


class Customerviewset(viewsets.ModelViewSet):
    queryset = Customer.objects.all()
    serializer_class = Customerserializer


class Carpoolviewset(viewsets.ModelViewSet):
    queryset = Carpool.objects.all()
    serializer_class = Carpoolserializer


class Voucherviewset(viewsets.ModelViewSet):
    queryset = Voucher.objects.all()
    serializer_class = Voucherserializer


class DCarpoolviewset(viewsets.ModelViewSet):
    queryset = DCarpool_request.objects.all()
    serializer_class = DCarpoolreqserializer
class Carrypic_requestviewset(viewsets.ModelViewSet):
    queryset = Carrypic.objects.all()
    serializer_class = Carrypicserializer

class Carpool_requestviewset(viewsets.ModelViewSet):
    queryset = Carpool_request.objects.all()
    serializer_class = Carpoolreqserializer


class PrivateRoomviewset(viewsets.ModelViewSet):
    queryset = PrivateRoom.objects.all()
    serializer_class = PrivateRoomserializer
class HClientCarpoolviewset(viewsets.ModelViewSet):
    queryset = History_Client.objects.all()
    serializer_class = HClientcarpoolserializer
class HDriverCarpoolviewset(viewsets.ModelViewSet):
    queryset = History_Driver.objects.all()
    serializer_class = HDrivercarpoolserializer


class Login(APIView):
    def get(self, request, data, password):
        try:

            if Customer.objects.filter(email=data).exists():

                try:
                    queryset = Customer.objects.filter(email=data).values()
                    lastSourceId = queryset[0]
                    response = {
                        "data": lastSourceId["id"], "statusCode": status.HTTP_200_OK}
                    json = JSONRenderer().render(response)
                except Exception as e:
                    print(e)
                if lastSourceId['password'] == password:
                    return Response(response)
                else:
                    return Response({"status": status.HTTP_400_BAD_REQUEST})
            else:
                return Response({"status": status.HTTP_400_BAD_REQUEST})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class Otp(APIView):
    def get(self, request, data, otp):
        try:
            if Customer.objects.filter(phone=data).exists():
                return Response({"status": status.HTTP_401_UNAUTHORIZED})
            else:
                try:
                    account_sid = ''
                    auth_token = ''
                    client = Client(account_sid, auth_token)
                    message = client.messages.create(
                        from_='',
                        to=f'{data}',
                        body=f'Carpool App Your Verication Pin Is {otp} This and For Any Problem Contact With Us On This Number +1839423 ')

                    print(message.sid)
                    response = {"statusCode": status.HTTP_200_OK}
                    json = JSONRenderer().render(response)
                except Exception as e:
                    print(e)
                return Response(response)
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class Check(APIView):
    def get(self, request, data):
        try:
            if Customer.objects.filter(phone=data).exists():
                return Response({"status": status.HTTP_401_UNAUTHORIZED})
            else:
                response = {"statusCode": status.HTTP_200_OK}
                json = JSONRenderer().render(response)
                return Response(response)

        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class Checkemail(APIView):
    def get(self, request, data):
        try:
            if Customer.objects.filter(email=data).exists():
                return Response({"status": status.HTTP_401_UNAUTHORIZED})
            else:
                response = {"statusCode": status.HTTP_200_OK}
                json = JSONRenderer().render(response)
                return Response(response)

        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class Checkusername(APIView):
    def get(self, request, data):
        try:
            if Customer.objects.filter(username=data).exists():
                return Response({"status": status.HTTP_401_UNAUTHORIZED})
            else:
                response = {"statusCode": status.HTTP_200_OK}
                json = JSONRenderer().render(response)
                return Response(response)

        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


POPULATION_SIZE = 100
MUTATION_RATE = 0.01
MAX_GENERATIONS = 100


def calculate_distance(latitude1, longitude1, latitude2, longitude2):
    # Calculate the distance between two points using a formula (e.g., Haversine formula)
    # You can use libraries like geopy for more accurate calculations
    # This implementation assumes Earth as a perfect sphere
    # You can replace this with a more accurate implementation if needed
    radius = 6371  # Earth's radius in kilometers
    dlat = math.radians(latitude2 - latitude1)
    dlon = math.radians(longitude2 - longitude1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(latitude1)) * math.cos(math.radians(latitude2)) * math.sin(
        dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c
    return distance


def time_difference(pickup_datetime1, pickup_datetime2):
    # Calculate the time difference in minutes between two pickup datetimes
    format_str = "%Y-%m-%d %H:%M:%S"
    pickup_time1 = datetime.datetime.strptime(pickup_datetime1, format_str)
    pickup_time2 = datetime.datetime.strptime(pickup_datetime2, format_str)
    time_diff = (pickup_time1 - pickup_time2).total_seconds() / 60
    return abs(time_diff)


def fitness(driver, passenger):
    # Calculate the fitness score for a driver based on passenger's parameters
    distance = calculate_distance(
        passenger['latitude'], passenger['longitude'], driver['latitude'], driver['longitude'])
    pickup_distance = calculate_distance(passenger['latitude'], passenger['longitude'], driver['latitude'],
                                         driver['longitude'])
    destination_distance = calculate_distance(passenger['destination_latitude'], passenger['destination_longitude'],
                                              driver['latitude'], driver['longitude'])
    time_diff = time_difference(
        passenger['pickup_datetime'], driver['pickup_datetime'])
    seat_difference = driver['seats_available'] - passenger['required_seats']
    distance_penalty = max(
        0, pickup_distance - passenger['max_pickup_distance']) / passenger['max_pickup_distance']
    destination_penalty = max(0,
                              destination_distance - passenger['max_destination_distance']) / passenger[
        'max_destination_distance']
    time_penalty = max(
        0, time_diff - passenger['max_time_difference']) / passenger['max_time_difference']
    seat_penalty = max(0, seat_difference) / driver['seats_available']
    penalty = distance_penalty + destination_penalty + time_penalty + seat_penalty
    return 1 / (1 + penalty)


def create_individual(drivers):
    # Create a random individual (driver)
    return random.choice(drivers)


def create_population(drivers, population_size):
    # Create a population of individuals (drivers)
    return [create_individual(drivers) for _ in range(population_size)]


def mutate(individual, min_seats, max_seats):
    # Perform mutation on an individual (driver)
    if random.random() < MUTATION_RATE:
        individual['seats_available'] = random.randint(
            min_seats, max_seats)  # Change the number of available seats
    return individual


def crossover(parent1, parent2):
    # Perform crossover between two parents (drivers)
    print(parent1,parent2)
    child = {}
    child['id'] = random.choice([parent1['id'], parent2['id']])
    child['longitude'] = random.choice(
        [parent1['longitude'], parent2['longitude']])
    child['latitude'] = random.choice(
        [parent1['latitude'], parent2['latitude']])
    child['seats_available'] = random.choice(
        [parent1['seats_available'], parent2['seats_available']])
    child['pickup_datetime'] = random.choice(
        [parent1['pickup_datetime'], parent2['pickup_datetime']])
    return child


def selection(population, passenger):
    # Perform selection (tournament selection in this example)
    tournament_size = 5
    selected_parents = []
    for _ in range(2):  # Select 2 parents
        tournament = random.sample(population, tournament_size)
        eligible_drivers = [
            driver for driver in tournament if driver['seats_available'] >= passenger['required_seats']]
        if eligible_drivers:
            winner = max(eligible_drivers, key=lambda x: fitness(x, passenger))
        else:
            winner = max(tournament, key=lambda x: fitness(x, passenger))
        selected_parents.append(winner)
    return selected_parents


def perform_assignment(passengers, drivers, population_size=100, mutation_rate=0.01, max_generations=100, filename="assignment_results.csv"):
    # Constants
    POPULATION_SIZE = population_size
    MUTATION_RATE = mutation_rate
    MAX_GENERATIONS = max_generations

    for passenger in passengers:
        passenger['assigned'] = False

    population = create_population(drivers, POPULATION_SIZE)
    assignments = []
    for passenger in passengers:
        for generation in range(MAX_GENERATIONS):
            new_population = []
            for _ in range(POPULATION_SIZE // 2):
                parents = selection(population, passenger)
                offspring = crossover(parents[0], parents[1])
                offspring = mutate(offspring, 1, 5)
                new_population.extend([parents[0], parents[1], offspring])
            population = new_population
            if not passenger['assigned']:
                best_driver = max(
                    population, key=lambda x: fitness(x, passenger))
                if best_driver['seats_available'] >= passenger['required_seats']:
                    assignments.append(
                        {"passenger_id": passenger['id'], "driver_id": best_driver['id']})
                    passenger['assigned'] = True
                    break  # Exit the current generation loop once a passenger is assigned

    return assignments


def Select_of_client_interface(client_id, formatted_date):
    
    passengers = [
        {
            "id": client_id['id'],
            "longitude": client_id['long'],
            "latitude": client_id['lat'],
            "destination_longitude": client_id['des_long'],
            "destination_latitude": client_id['des_lat'],
            "max_pickup_distance": 5,
            "max_destination_distance": 5,
            "max_time_difference": 30,  # in minutes
            "required_seats": int(client_id['seat']),
            "pickup_datetime": f"{client_id['date']} {client_id['time']}"
        }
    ]
    queryset = DCarpool_request.objects.values()
    drivers = []
    for item in queryset:
        if item['date'] == formatted_date and item['seat'] != 0.0:
            lat1 = item['lat']
            lon1 = item['long']
            # Client
            lat2 = client_id['lat']
            lon2 = client_id['long']
            # instance.des_lat, instance.des_long, client_id['des_lat'], client_id['des_long']
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            time1 = datetime.datetime.strptime(item['time'], "%H:%M:%S")
            time2 = datetime.datetime.strptime(client_id['time'], "%H:%M:%S")
            time_diff = time1 - time2
            minutes_diff = time_diff.total_seconds() // 60
            print(minutes_diff)
            print(f"Pickup Lat long {lat1, lon1, lat2, lon2} {distance}")
            if abs(minutes_diff) <= 30:

                if distance < item['range']:
                    drivers.append({
                        "id": item['id'],
                        "longitude": item['long'],
                        "latitude": item['lat'],
                        "seats_available": int(item['seat']),
                        "pickup_datetime": f"{item['date']} {item['time']}"
                    },
                    )
            else:              
                pass
            
                
    try:
        assignments = perform_assignment(passengers, drivers)
        i=0
        for assignment in assignments:
            print(
                f"Passenger ID {i+1} : {assignment['passenger_id']}, Driver ID: {assignment['driver_id']}")

        instance = DCarpool_request.objects.get(pk=assignments[0]['driver_id'])
        print(
            f"This if final {calculate_distance(instance.des_lat, instance.des_long, client_id['des_lat'], client_id['des_long'])}")
        if calculate_distance(instance.des_lat, instance.des_long, client_id['des_lat'], client_id['des_long']) < item['range']:
            instance.lat = instance.lat
            instance.long = instance.long
            instance.des_lat = instance.des_lat
            instance.des_long = instance.des_long
            instance.client_id = instance.client_id
            instance.assien_driver = instance.assien_driver
            instance.date = instance.date
            instance.time = instance.time
            instance.price = instance.price
            instance.distance = instance.distance
            instance.seat = instance.seat-int(client_id['seat'])
            instance.status=instance.status
            instance.client_request_number=instance.client_request_number
            instance.driver_request_number=instance.driver_request_number
            instance.range=instance.range
            instance.save()
            serializer = DCarpool_request(instance)

            queryset = DCarpool_request.objects.filter(
                id=assignments[0]['driver_id']).values()
            lastSourceId = queryset[0]

            data = {
                "lat": client_id['lat'],
                "long": client_id['long'],
                "des_lat": client_id["des_lat"],
                "des_long": client_id["des_long"],
                "client_id": client_id['client_id'],
                "assien_driver": lastSourceId['assien_driver'],
                "date": client_id['date'],
                "time": client_id['time'],
                "price": client_id['price'],
                "distance": client_id['distance'],
                "seat": client_id['seat'],
                "sheuler_number": 0,
                "picksheuler_number": 0,
                "current_long": lastSourceId['long'],
                "current_lat": lastSourceId['lat'],
                "status": "Match",
                "client_request_number":client_id["client_request_number"],
                "driver_request_number":lastSourceId['driver_request_number']

            }
            print(client_id['id'])
            instance = Carpool_request.objects.get(id=client_id['id'])
            instance.lat = instance.lat
            instance.long = instance.long
            instance.des_lat = instance.des_lat
            instance.des_long = instance.des_long
            instance.client_id = instance.client_id
            instance.assien_driver = instance.assien_driver
            instance.date = instance.date
            instance.time = instance.time
            instance.price = instance.price
            instance.distance = instance.distance
            instance.seat = instance.seat
            instance.status = "Match"
            instance.client_request_number=instance.client_request_number
            instance.driver_request_number=instance.driver_request_number
                
            

            serializer = Finalcarpoolserializer(data=data)
            if serializer.is_valid():
                serializer.save()
        else:
            instance = Carpool_request.objects.get(id=client_id['id'])
            instance.lat = instance.lat
            instance.long = instance.long
            instance.des_lat = instance.des_lat
            instance.des_long = instance.des_long
            instance.client_id = instance.client_id
            instance.assien_driver = instance.assien_driver
            instance.date = instance.date
            instance.time = instance.time
            instance.price = instance.price
            instance.distance = instance.distance
            instance.seat = instance.seat
            instance.status = "Fail"
            
            instance.client_request_number=instance.client_request_number
            instance.driver_request_number=instance.driver_request_number

            serializer = Carpool_request(data=data)
            if serializer.is_valid():
                serializer.save()

    except Exception as e:
        print(e)


def generate_pickup_sequence(passengers, current_lat, current_lon):
    pickup_sequence = []
    remaining_passengers = passengers[:]
    current_location = (current_lat, current_lon)
    current_time = datetime.datetime.now()

    while remaining_passengers:
        closest_passenger = None
        min_distance = float('inf')
        min_pickup_time_diff = float('inf')

        for passenger in remaining_passengers:
            distance = geodesic(
                current_location, (passenger['lat'], passenger['lon'])).meters
            pickup_time_diff = (
                passenger['pickup_time'] - current_time).total_seconds()

            if pickup_time_diff < min_pickup_time_diff:
                min_pickup_time_diff = pickup_time_diff
                min_distance = distance
                closest_passenger = passenger

        if closest_passenger is None:
            break

        remaining_passengers.remove(closest_passenger)
        current_location = (closest_passenger['lat'], closest_passenger['lon'])
        pickup_sequence.append(
            (closest_passenger['name'], closest_passenger['lat'], closest_passenger['lon']))

    return pickup_sequence


def Make_client_sheduler(unique_drivers, data):
    Driver_data = []
    current_lat = 0
    current_long = 0
    for item in data:
        if item["assien_driver"] == unique_drivers:
            Driver_data.append(item)
    sorted_data = sorted(Driver_data, key=lambda x: x['driver_request_number'])
    ids=[]
    for item in sorted_data:
        ids.append(item['driver_request_number'])
    unique_elements = list(set(ids))
    print(unique_elements)
    # Final_data=[]
    print(unique_elements)
    for request_number in unique_elements:
        driver_request_number = request_number
        filtered_data = [entry for entry in data if entry["driver_request_number"] == driver_request_number]
            
            # Retrieve the IDs of the filtered entries
        compute = [entry["id"] for entry in filtered_data]
        Dataset_pickup = []
        for item in compute :
            queryset =Final_Carpool.objects.filter(id=item).values()
            lastSourceId = queryset[0]
            given_datetime = datetime.datetime.strptime(f"{lastSourceId['date']} {lastSourceId['time']}", "%Y-%m-%d %H:%M:%S")
            current_lat = lastSourceId['current_lat']
            current_long = lastSourceId["current_long"]
            Dataset_pickup.append({

                'name': lastSourceId['id'],
                'lat': lastSourceId['lat'],
                'lon': lastSourceId['long'],
                'pickup_time': given_datetime,
            })
        print(Dataset_pickup)
        pickup_sequence = generate_pickup_sequence(
        Dataset_pickup, current_lat, current_long)
        current = pickup_sequence[len(pickup_sequence)-1]
        current_lat = current[1]
        current_long = current[2]
        for i, (name, lat, lon) in enumerate(pickup_sequence, 1):
            instance = Final_Carpool.objects.get(pk=name)
            instance.lat = instance.lat
            instance.long = instance.long
            instance.des_lat = instance.des_lat
            instance.des_long = instance.des_long
            instance.client_id = instance.client_id
            instance.assien_driver = instance.assien_driver
            instance.date = instance.date
            instance.time = instance.time
            instance.price = instance.price
            instance.distance = instance.distance
            instance.seat = instance.seat
            instance.sheuler_number = instance.sheuler_number
            instance.picksheuler_number = i
            instance.current_long = current_long
            instance.current_lat = current_lat
            instance.status=instance.status
            instance.client_request_number=instance.client_request_number
            instance.driver_request_number=instance.driver_request_number
            instance.save()


def generate_pickup_sequence2(passengers, current_lat, current_lon):
    pickup_sequence = []
    remaining_passengers = passengers[:]
    current_location = (current_lat, current_lon)

    while remaining_passengers:
        closest_passenger = None
        min_distance = float('inf')

        for passenger in remaining_passengers:
            distance = geodesic(
                current_location, (passenger[1], passenger[2])).meters

            if distance < min_distance:
                min_distance = distance
                closest_passenger = passenger

        if closest_passenger is None:
            break

        remaining_passengers.remove(closest_passenger)
        current_location = (closest_passenger[1], closest_passenger[2])
        pickup_sequence.append(
            (closest_passenger[0], closest_passenger[1], closest_passenger[2]))

    return pickup_sequence


def Make_client_sheduler_dropoff(unique_drivers, data):
    Driver_data = []
    current_lat = 0
    current_long = 0
    for item in data:
        if item["assien_driver"] == unique_drivers:
            Driver_data.append(item)
    sorted_data = sorted(Driver_data, key=lambda x: x['driver_request_number'])
    ids=[]
    for item in sorted_data:
        ids.append(item['driver_request_number'])
    unique_elements = list(set(ids))
    print(unique_elements)
    # Final_data=[]
    print(unique_elements)
    for request_number in unique_elements:
        driver_request_number = request_number
        filtered_data = [entry for entry in data if entry["driver_request_number"] == driver_request_number]
            
            # Retrieve the IDs of the filtered entries
        compute = [entry["id"] for entry in filtered_data]
        Dataset_pickup = []
        for item in compute :
            queryset =Final_Carpool.objects.filter(id=item).values()
            lastSourceId = queryset[0]
            current_lat = lastSourceId['current_lat']
            current_long = lastSourceId["current_long"]
            Dataset_pickup.append((lastSourceId['id'],
                lastSourceId['lat'],
                lastSourceId['long'],
            ))
        print(Dataset_pickup)
        
        pickup_sequence = generate_pickup_sequence2(Dataset_pickup, current_lat, current_long)

        for i, (name, lat, lon) in enumerate(pickup_sequence, 1):
            print(f"Stop {i}: {name} (Latitude: {lat}, Longitude: {lon})")
            instance = Final_Carpool.objects.get(pk=name)
            instance.lat = instance.lat
            instance.long = instance.long
            instance.des_lat = instance.des_lat
            instance.des_long = instance.des_long
            instance.client_id = instance.client_id
            instance.assien_driver = instance.assien_driver
            instance.date = instance.date
            instance.time = instance.time
            instance.price = instance.price
            instance.distance = instance.distance
            instance.seat = instance.seat
            instance.sheuler_number = i
            instance.picksheuler_number = instance.picksheuler_number
            instance.current_long = instance.current_long
            instance.current_lat = instance.current_lat
            instance.client_request_number=instance.client_request_number
            instance.driver_request_number=instance.driver_request_number
            instance.save()


class GetDetailsofSheduler(APIView):
    def get(self, request, driver_id,id_of):
        try:
            current_date = datetime.date.today()
            formatted_date = current_date.strftime("%Y-%m-%d")
           
            queryset = Final_Carpool.objects.values()
            data = []
            for item in queryset:
                if item['date'] == formatted_date:
                    data.append(item)
            # data/driver_id
            Driver_data = []
            for item in data:
                if item["assien_driver"] == driver_id and item["driver_request_number"]==id_of:
                    Driver_data.append(item)
            Pick_up_id = []
            Drop_of_id = []
            sorted_data = sorted(Driver_data, key=lambda x: x['picksheuler_number'])
            for item in sorted_data:
                Pick_up_id.append(item['id'])
            sorted_data2 = sorted(Driver_data, key=lambda x: x['sheuler_number'])
            for item in sorted_data2:
                Drop_of_id.append(item['id'])

            return Response({"status": status.HTTP_200_OK, "Pickup": Pick_up_id, "Dropoff": Drop_of_id})
        except:
            Response({"status": status.HTTP_400_BAD_REQUEST})


def Updatedata(distance, driverid, clientid):

    addmoney = distance
    cutoff = distance/5
    try:
        user = Customer.objects.get(id=clientid)
        user.username = user.username
        user.email = user.email
        user.password = user.password
        user.phone = user.phone
        user.carplate = user.carplate
        user.carmodel = user.carmodel
        user.image1 = user.image1
        user.image2 = user.image2
        user.image3 = user.image3
        user.balance = user.balance-cutoff
        user.trips_as_client = user.trips_as_client+1
        user.trips_as_captain = user.trips_as_captain
        user.Profile = user.Profile
        user.expo_token = user.expo_token
        user.point = user.point
        user.save()




        # Driver Add Money
        user = Customer.objects.get(id=driverid)
        user.username = user.username
        user.email = user.email
        user.password = user.password
        user.phone = user.phone
        user.carplate = user.carplate
        user.carmodel = user.carmodel
        user.image1 = user.image1
        user.image2 = user.image2
        user.image3 = user.image3
        user.balance = user.balance
        user.trips_as_client = user.trips_as_client
        user.trips_as_captain = user.trips_as_captain+1
        user.Profile = user.Profile
        user.expo_token = user.expo_token
        user.point = user.point+addmoney
        user.save()
    except:
        pass


class DoneCarpool(APIView):
    def get(self, request, driver_id,id_of):
        try:
            current_date = datetime.date.today()
            formatted_date = current_date.strftime("%Y-%m-%d")
            queryset = Final_Carpool.objects.values()
            data = []
            for item in queryset:
                if item['date'] == formatted_date:
                    data.append(item)
            # data/driver_id
            Driver_data = []
            for item in data:
                if item["assien_driver"] == driver_id:
                    Driver_data.append(item)
            Clients = []

            for item in Driver_data:
                if item["assien_driver"] == driver_id and item["driver_request_number"]==id_of:
                    Updatedata(item['price'], driver_id, item['client_id'])
                    # item = Final_Carpool.objects.get(pk=item['id'])
                    # item.delete()
                    
            
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class Shedulder(APIView):
    def get(self, request):
        try:
            date = datetime.date.today()
            formatted_date = date.strftime('%Y-%m-%d')

            # Convert the next date to a string in the desired format
            print(formatted_date)
            # formatted_date= next_date.strftime('%Y-%m-%d')
            queryset = Final_Carpool.objects.values()
            data = []
            for item in queryset:
                if item['date'] == formatted_date:
                    data.append(item)
            unique_drivers = set(item["assien_driver"] for item in data)
            for item in unique_drivers:
                Make_client_sheduler(item, data)
            
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class DropofSheduler(APIView):
    def get(self, request):
        try:
            date = datetime.date.today()
            formatted_date = date.strftime('%Y-%m-%d')
            queryset = Final_Carpool.objects.values()
            data = []
            for item in queryset:
                if item['date'] == formatted_date:
                    data.append(item)
            unique_drivers = set(item["assien_driver"] for item in data)
            for item in unique_drivers:
                Make_client_sheduler_dropoff(item, data)

            return Response({"status": status.HTTP_200_OK})
        except:
            Response({"status": status.HTTP_400_BAD_REQUEST})


class GALog(APIView):
    def get(self, request):
        try:
            # formatted_date = current_date.strftime("%Y-%mm-%d")
            # date = str(formatted_date)
            # parts = date.split("-")
            # formatted_date = f"{parts[0]}-{int(parts[1]):d}-{int(parts[2]):d}"

            date = datetime.date.today()
            formatted_date = date.strftime('%Y-%m-%d')

            print(formatted_date)

            queryset = Carpool_request.objects.values()
            client = []
            print(date)
            for item in queryset:
                print(formatted_date,item['date'])
                if item['date'] == formatted_date:
                    
                    client.append(item)
                    Select_of_client_interface(item, formatted_date)
           
            return Response({"status": status.HTTP_200_OK, "Client": client})
        except:
            Response({"status": status.HTTP_400_BAD_REQUEST})


class UpdateBalance(APIView):
    def get(self, request, data, otp):
        try:
            user = Customer.objects.get(id=data)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance+otp
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.save()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


class CreateVoucher(APIView):
    def get(self, request, price, clientid):
        try:
            random_number = random.randint(1000000000, 9999999999)
            queryset = Customer.objects.filter(id=clientid).values()
            lastSourceId = queryset[0]
            number = 1234567890  # Replace with your desired number
            qr_code_image = generate_qr_code(number)
            if price > (lastSourceId['point']):
                return Response({"status": status.HTTP_400_BAD_REQUEST, "Messege": "You Can not Create Voucher"})
            else:
                BASE_DIR = Path(__file__).resolve().parent.parent
                MEDIA_ROOT = os.path.join(BASE_DIR, 'Media')
                MEDIA_URL = '/Media/'
                qr_code_image = generate_qr_code(random_number)
                qr_code_image.save(f"{MEDIA_ROOT}/Media/result-{random_number}.png") 
                
                data = {
                    "price": price,
                    "voucher": random_number,
                    "driver_id":clientid,

                }
                serializer = Voucherserializer(data=data)
                if serializer.is_valid():
                    instance = serializer.save()
                    saved_id = instance.id
                    
                user = Voucher.objects.get(id=saved_id)
                user.price=user.price
                user.voucher= user.voucher
                user.driver_id=user.driver_id
                user.Qr_code=f"{MEDIA_ROOT}/Media/result-{random_number}.png"
                user.product= user.product
                user.save()
            CutVouchermoney(price, clientid)
            send_email(lastSourceId['email'], random_number, price)
            return Response({"status": status.HTTP_200_OK, "Voucher Number": random_number})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})


def send_email(email, voucher, price):
    print(email, voucher, price)


def CutVouchermoney(price, clientid):
    cutoff = price/5
    try:
        user = Customer.objects.get(id=clientid)
        user.username = user.username
        user.email = user.email
        user.password = user.password
        user.phone = user.phone
        user.carplate = user.carplate
        user.carmodel = user.carmodel
        user.image1 = user.image1
        user.image2 = user.image2
        user.image3 = user.image3
        user.balance = user.balance
        user.trips_as_client = user.trips_as_client+1
        user.trips_as_captain = user.trips_as_captain
        user.Profile = user.Profile
        user.expo_token = user.expo_token
        user.point = user.point-cutoff
        user.save()
    except:
        pass
        # Driver Add Money


class VoucherCheck(APIView):
    def get(self, request, number, price):
        try:
            if Voucher.objects.filter(voucher=number).exists():
                queryset = Voucher.objects.filter(voucher=number).values()
                lastSourceId = queryset[0]
                if lastSourceId['price'] == price:
                    return Response({"statusCode": status.HTTP_200_OK})
                else:
                    return Response({"statusCode": status.HTTP_400_BAD_REQUEST, "Message": "This Voucher Is not valid Money"})
            else:
                return Response({"status": status.HTTP_400_BAD_REQUEST})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
# class ClearGarbage(APIView):
#     def get(self,request):
#         current_date = datetime.date.today()
#         formatted_date = current_date.strftime("%Y-%m-%d")
#         date = str(formatted_date)
#         parts = date.split("-")
#         formatted_date = f"{parts[0]}-{int(parts[1]):d}-{int(parts[2]):d}"
#         queryset = Carpool_request.objects.values()
#         client = []
#         for item in queryset:
#             if item['date']!= formatted_date:
#                 client.append(item)
#                 item = Carpool_request.objects.get(pk=item['id'])
#                 item.delete()

#         return Response({"status": status.HTTP_200_OK, "Client": client})
class DriverRequestnumber(APIView):
    def get(self,request,driver_id):
        # current_date = datetime.date.today()
        # formatted_date = current_date.strftime("%Y-%m-%d")
        # date = str(formatted_date)
        # parts = date.split("-")
        # formatted_date = f"{parts[0]}-{int(parts[1]):d}-{int(parts[2]):d}"
        current_date = datetime.date.today()

        # Calculate the next date
        next_date = current_date + datetime.timedelta(days=1)

        # Convert the next date to a string in the desired format
        formatted_date= next_date.strftime('%Y-%m-%d')
        queryset = DCarpool_request.objects.values()
        data = []
        for item in queryset:
            if item['date'] == formatted_date:
                data.append(item)
        Driver_data = []
        for item in data:
            if item["assien_driver"] == driver_id:
                Driver_data.append(item)
        return Response({"status": status.HTTP_200_OK, "Client": len(Driver_data)+1})
class ClientRequestnumber(APIView):
    def get(self,request,driver_id):
        current_date = datetime.date.today()

        # Calculate the next date
        next_date = current_date + datetime.timedelta(days=1)

        # Convert the next date to a string in the desired format
        formatted_date= next_date.strftime('%Y-%m-%d')
        
        queryset = Carpool_request.objects.values()
        data = []
        for item in queryset:
            if item['date'] == formatted_date:
                data.append(item)
        Driver_data = []
        for item in data:
            if item["assien_driver"] == driver_id:
                Driver_data.append(item)
        return Response({"status": status.HTTP_200_OK, "Client": len(Driver_data)+1})
class FinalCarpoolsdata(APIView):
    def get(self,request,driver_id):
        date = datetime.date.today()
        formatted_date = date.strftime('%Y-%m-%d')


        queryset = Final_Carpool.objects.values()
        data = []
        for item in queryset:
            if item['date'] == formatted_date:
                data.append(item)
        Driver_data = []
        for item in data:
            if item["assien_driver"] == driver_id:
                Driver_data.append(item)
        sorted_data = sorted(Driver_data, key=lambda x: x['driver_request_number'])
        ids=[]
        for item in sorted_data:
            ids.append(item['driver_request_number'])
        unique_elements = list(set(ids))
        Final_data=[]
        for request_number in unique_elements:
            driver_request_number = request_number
            filtered_data = [entry for entry in data if entry["driver_request_number"] == driver_request_number]
            queryset = DCarpool_request.objects.filter(driver_request_number=driver_request_number).values()
            lastSourceId = queryset[0]
            # Retrieve the IDs of the filtered entries
            ids = [entry["id"] for entry in filtered_data]
            Final_data.append({

                f"{driver_request_number}":ids,
                "Lenth":len(ids),
                "Data":lastSourceId
                

            })
            

        return Response({"status": status.HTTP_200_OK, "Final": Final_data})
def generate_qr_code(number):
    # Create a QR code object
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    # Add the number to the QR code
    qr.add_data(str(number))
    qr.make(fit=True)
    # Create an image from the QR code
    qr_image = qr.make_image(fill_color="black", back_color="white")
    # Return the QR code image
    return qr_image

class DeatilsofProdut(APIView):
    def get(self,request,driver_id):
        
       
        queryset = Voucher.objects.values()
        data = []
        for item in queryset:
            if item['driver_id'] == driver_id:
                data.append({"voucher":item['voucher'],"price": item['price'],"Qr_code":item['Qr_code']
                            })


        
        

        
        return Response({"status": status.HTTP_200_OK,"Data":data})

class ChangeUsername(APIView):
    def get(self,request,driver_id,name):
        try:
            user = Customer.objects.get(id=driver_id)
            user.username = name
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            return Response({"status": status.HTTP_200_OK})





      
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class ChangeUserPassword(APIView):
    def get(self,request,driver_id,password):
        try:
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class ChangeUsercarplate(APIView):
    def get(self,request,driver_id,carplate):
        try:
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class ChangeUsercarplate(APIView):
    def get(self,request,driver_id,carplate):
        try:
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})

class ChangeUsercarmodel(APIView):
    def get(self,request,driver_id,carmodel):
        try:
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})

class Changepicture(APIView):
    def get(self,request,driver_id,pic_id):
        try:
            picture = Carrypic.objects.get(id=pic_id)
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = picture.pic
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            picture.delete()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class Changepicture1(APIView):
    def get(self,request,driver_id,pic_id):
        try:
            picture = Carrypic.objects.get(id=pic_id)
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = picture.pic
            user.image2 = user.image2
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            picture.delete()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class Changepicture2(APIView):
    def get(self,request,driver_id,pic_id):
        try:
            picture = Carrypic.objects.get(id=pic_id)
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = picture.pic
            user.image3 = user.image3
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            picture.delete()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})
class Changepicture3(APIView):
    def get(self,request,driver_id,pic_id):
        try:
            picture = Carrypic.objects.get(id=pic_id)
            user = Customer.objects.get(id=driver_id)
            user.username = user.username
            user.email = user.email
            user.password = user.password
            user.phone = user.phone
            user.carplate = user.carplate
            user.carmodel = user.carmodel
            user.image1 = user.image1
            user.image2 = user.image2
            user.image3 = picture.pic
            user.balance = user.balance
            user.trips_as_client = user.trips_as_client
            user.trips_as_captain = user.trips_as_captain
            user.Profile = user.Profile
            user.expo_token = user.expo_token
            user.point = user.point
            user.save()
            picture.delete()
            return Response({"status": status.HTTP_200_OK})
        except:
            return Response({"status": status.HTTP_400_BAD_REQUEST})