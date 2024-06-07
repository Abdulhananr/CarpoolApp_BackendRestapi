import requests
#pip install schedule
# import schedule
import time

def request_api():
    # Make the API request here
    response = requests.get("http://10.212.191.48/api/Shedulder")
    response1 =requests.get("http://10.212.191.48/api/Sheduldedrop")
    response2 =requests.get("http://10.212.191.48/api/CheckGa")
    # Process the API response
    if response.status_code == 200:
        # Process the successful response
        print("API request successful")
    else:
        # Handle the API request error
        print("API request failed")
    if response1.status_code == 200:
        # Process the successful response
        print("API request successful")
    else:
        # Handle the API request error
        print("API request failed")
    if response2.status_code == 200:
        # Process the successful response
        print("API request successful")
    else:
        # Handle the API request error
        print("API request failed")
# Schedule the API request to run every day at 12 AM
# schedule.every().day.at("00:00").do(request_api)

# while True:
#     # Run the scheduled tasks
#     schedule.run_pending()
#     time.sleep(1)
request_api()