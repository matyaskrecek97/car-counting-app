import os
import requests
import time


class EventSender:
    def __init__(self):
        self.location_id = os.environ.get('LOCATION_ID')
        # self.server_url = os.environ.get('SERVER_URL')
        self.server_url = "http://192.168.1.108:3000"

    def send_detection_event(self):
        url = f'{self.server_url}/detection'
        data = {'locationId': self.location_id}
        response = requests.post(url, data=data)
        print("Send detection request:", response.status_code,
              response.reason, response.text)
        return response

    def send_heartbeat_event(self):
        url = f'{self.server_url}/heartbeat'
        data = {'locationId': self.location_id}
        response = requests.post(url, data=data)
        print("Send heartbeat request:", response.status_code,
              response.reason, response.text)
        return response

    def start_hearbeat(self):
        print("Heartbeat started")
        while True:
            self.send_heartbeat_event()
            time.sleep(60)
