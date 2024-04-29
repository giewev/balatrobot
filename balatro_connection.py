import socket
import json
import subprocess
from enum import Enum
import random


class State(Enum):
    SELECTING_HAND = 1
    HAND_PLAYED = 2
    DRAW_TO_HAND = 3
    GAME_OVER = 4
    SHOP = 5
    PLAY_TAROT = 6
    BLIND_SELECT = 7
    ROUND_EVAL = 8
    TAROT_PACK = 9
    PLANET_PACK = 10
    MENU = 11
    TUTORIAL = 12
    SPLASH = 13
    SANDBOX = 14
    SPECTRAL_PACK = 15
    DEMO_CTA = 16
    STANDARD_PACK = 17
    BUFFOON_PACK = 18
    NEW_ROUND = 19


class Actions(Enum):
    SELECT_BLIND = 1
    SKIP_BLIND = 2
    PLAY_HAND = 3
    DISCARD_HAND = 4
    END_SHOP = 5
    REROLL_SHOP = 6
    BUY_CARD = 7
    BUY_VOUCHER = 8
    BUY_BOOSTER = 9
    SELECT_BOOSTER_CARD = 10
    SKIP_BOOSTER_PACK = 11
    SELL_JOKER = 12
    USE_CONSUMABLE = 13
    SELL_CONSUMABLE = 14
    REARRANGE_JOKERS = 15
    REARRANGE_CONSUMABLES = 16
    REARRANGE_HAND = 17
    PASS = 18
    START_RUN = 19
    SEND_GAMESTATE = 20


class BalatroConnection:
    def __init__(self, bot_port: int = 12346):
        self.bot_port = bot_port
        self.addr = ("localhost", self.bot_port)
        self.sock = None
        self.balatro_instance = None

    def start_balatro_instance(self):
        balatro_exec_path = (
            r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
        )
        self.balatro_instance = subprocess.Popen(
            [balatro_exec_path, str(self.bot_port)]
        )

    def stop_balatro_instance(self):
        if self.balatro_instance:
            self.balatro_instance.kill()

    def connect(self):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(1)
            self.sock.connect(self.addr)

    def send_cmd(self, cmd):
        self.connect()
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def actionToCmd(self, action):
        result = []

        for x in action:
            if isinstance(x, Actions):
                result.append(x.name)
            elif type(x) is list:
                result.append(",".join([str(y) for y in x]))
            else:
                result.append(str(x))

        return "|".join(result)

    def receive_data(self):
        self.connect()
        try:
            data = self.sock.recv(65536)
            jsondata = json.loads(data)
            if "response" in jsondata:
                print(jsondata["response"])
            return jsondata
        except socket.error as e:
            print(e)
            self.sock = None
            return {}

    def poll_state(self):
        self.send_cmd("HELLO")
        return self.receive_data()

    def send_action(self, action):
        if action[0] == Actions.START_RUN:
            seed = action[3]
            if seed is None:
                seed = self.random_seed()
            action[3] = seed
        cmdstr = self.actionToCmd(action)
        self.send_cmd(cmdstr)

    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))
