import os
import requests
import time


class EventSender:
    def __init__(self):
        self.location_id = os.environ.get('LOCATION_ID')
        self.server_url = os.environ.get('SERVER_URL')
        self.token = os.environ.get('TOKEN')
        self.headers = {'Authorization': 'Bearer ' + self.token}

    def send_detection_event(self):
        url = f'{self.server_url}/detection'
        data = {'locationId': self.location_id}
        response = requests.post(url, json=data, headers=self.headers)
        print("Send detection request:", response.status_code,
              response.reason, response.text)
        return response

    def send_heartbeat_event(self):
        url = f'{self.server_url}/heartbeat'
        data = {'locationId': self.location_id}
        response = requests.post(url, json=data, headers=self.headers)
        print("Send heartbeat request:", response.status_code,
              response.reason, response.text)
        return response

    def start_hearbeat(self):
        print("Heartbeat started")
        while True:
            self.send_heartbeat_event()
            time.sleep(60)


if __name__ == '__main__':
    sender = EventSender()
    sender.start_hearbeat()
