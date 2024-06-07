import random
import math
import csv
import datetime

# Constants
POPULATION_SIZE = 50
MUTATION_RATE = 0.01
MAX_GENERATIONS = 50

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
    format_str = "%d-%m-%Y %H:%M:%S"
    pickup_time1 = datetime.datetime.strptime(pickup_datetime1, format_str)
    pickup_time2 = datetime.datetime.strptime(pickup_datetime2, format_str)
    time_diff = (pickup_time1 - pickup_time2).total_seconds() / 60
    return abs(time_diff)

def fitness(driver, passenger):
    # Calculate the fitness score for a driver based on passenger's parameters
    distance = calculate_distance(passenger['latitude'], passenger['longitude'], driver['latitude'], driver['longitude'])
    pickup_distance = calculate_distance(passenger['latitude'], passenger['longitude'], driver['latitude'],
                                         driver['longitude'])
    destination_distance = calculate_distance(passenger['destination_latitude'], passenger['destination_longitude'],
                                              driver['latitude'], driver['longitude'])
    time_diff = time_difference(passenger['pickup_datetime'], driver['pickup_datetime'])
    seat_difference = driver['seats_available'] - passenger['required_seats']
    distance_penalty = max(0, pickup_distance - passenger['max_pickup_distance']) / passenger['max_pickup_distance']
    destination_penalty = max(0,
                              destination_distance - passenger['max_destination_distance']) / passenger[
                               'max_destination_distance']
    time_penalty = max(0, time_diff - passenger['max_time_difference']) / passenger['max_time_difference']
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
        individual['seats_available'] = random.randint(min_seats, max_seats)  # Change the number of available seats
    return individual

def crossover(parent1, parent2):
    # Perform crossover between two parents (drivers)
    child = {}
    child['id'] = random.choice([parent1['id'], parent2['id']])
    child['longitude'] = random.choice([parent1['longitude'], parent2['longitude']])
    child['latitude'] = random.choice([parent1['latitude'], parent2['latitude']])
    child['seats_available'] = random.choice([parent1['seats_available'], parent2['seats_available']])
    child['pickup_datetime'] = random.choice([parent1['pickup_datetime'], parent2['pickup_datetime']])
    return child

def selection(population, passenger):
    # Perform selection (tournament selection in this example)
    tournament_size = 5
    selected_parents = []
    for _ in range(2):  # Select 2 parents
        tournament = random.sample(population, tournament_size)
        eligible_drivers = [driver for driver in tournament if driver['seats_available'] >= passenger['required_seats']]
        if eligible_drivers:
            winner = max(eligible_drivers, key=lambda x: fitness(x, passenger))
        else:
            winner = max(tournament, key=lambda x: fitness(x, passenger))
        selected_parents.append(winner)
    return selected_parents

def perform_assignment(passengers, drivers, population_size=100, mutation_rate=0.01, max_generations=100):
    # Create initial population
    population = create_population(drivers, population_size)
    
    # Assignments storage
    assignments = []
    
    for passenger in passengers:
        passenger['assigned'] = False
        
        # Printing population chromosomes before assignment
        print("\nInitial Population Chromosomes:")
        for i, chromosome in enumerate(population, 1):
            print(f"Chromosome {i}: {chromosome}")

        for generation in range(max_generations):
            new_population = []
            
            # Crossover and mutation
            for _ in range(population_size // 2):
                parents = selection(population, passenger)
                offspring = crossover(parents[0], parents[1])
                offspring = mutate(offspring, 1, 5)
                new_population.extend([parents[0], parents[1], offspring])
            
            # Update population
            population = new_population
            
            # Printing population chromosomes after each generation
            print(f"\nGeneration {generation + 1} Population Chromosomes:")
            for i, chromosome in enumerate(population, 1):
                print(f"Chromosome {i}: {chromosome}")
            
            if not passenger['assigned']:
                best_driver = max(population, key=lambda x: fitness(x, passenger))
                if best_driver['seats_available'] >= passenger['required_seats']:
                    assignments.append({"passenger_id": passenger['id'], "driver_id": best_driver['id']})
                    passenger['assigned'] = True
                    break  # Exit the current generation loop once a passenger is assigned
    
    # Printing final population chromosomes
    print("\nFinal Population Chromosomes:")
    for i, chromosome in enumerate(population, 1):
        print(f"Chromosome {i}: {chromosome}")
    
    return assignments

# Sample data
passengers = [
    {
        "id": 1,
        "longitude": 40.7128,
        "latitude": -74.0060,
        "destination_longitude": 4.7589,
        "destination_latitude": -7.9851,
        "max_pickup_distance": 5,
        "max_destination_distance": 5,
        "max_time_difference": 30,  # in minutes
        "required_seats": 2,
        "pickup_datetime": "23-12-2020 12:20:10"
    }
    # Add more passengers here
]

drivers = [
    {
        "id": 1,
        "longitude": 40.7128,
        "latitude": -74.0060,
        "seats_available": 4,
        "pickup_datetime": "23-12-2020 12:00:00"
    }
    # Add more drivers here
]

# Perform assignment
assignments = perform_assignment(passengers, drivers)

# Print assignments
print("\nAssignments:")
for assignment in assignments:
    print(f"Passenger {assignment['passenger_id']} assigned to Driver {assignment['driver_id']}")
