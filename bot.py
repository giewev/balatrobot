#!/usr/bin/python3

import sys
import json
import socket
import time
from enum import Enum
from gamestates import cache_state
import subprocess
import random
from balatro_connection import BalatroConnection, State, Actions


class Bot:
    def __init__(
        self,
        deck: str,
        stake: int = 1,
        seed: str = None,
        challenge: str = None,
        bot_port: int = 12346,
    ):
        self.G = None
        self.state = {}
        self.connection = BalatroConnection(bot_port)
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.challenge = challenge
        self.verified = False
        self.running = True

    def skip_or_select_blind(self, G):
        raise NotImplementedError(
            "Error: Bot.skip_or_select_blind must be implemented."
        )

    def select_cards_from_hand(self, G):
        raise NotImplementedError(
            "Error: Bot.select_cards_from_hand must be implemented."
        )

    def select_shop_action(self, G):
        raise NotImplementedError("Error: Bot.select_shop_action must be implemented.")

    def select_booster_action(self, G):
        raise NotImplementedError(
            "Error: Bot.select_booster_action must be implemented."
        )

    def sell_jokers(self, G):
        raise NotImplementedError("Error: Bot.sell_jokers must be implemented.")

    def rearrange_jokers(self, G):
        raise NotImplementedError("Error: Bot.rearrange_jokers must be implemented.")

    def use_or_sell_consumables(self, G):
        raise NotImplementedError(
            "Error: Bot.use_or_sell_consumables must be implemented."
        )

    def rearrange_consumables(self, G):
        raise NotImplementedError(
            "Error: Bot.rearrange_consumables must be implemented."
        )

    def rearrange_hand(self, G):
        raise NotImplementedError("Error: Bot.rearrange_hand must be implemented.")

    def start_balatro_instance(self):
        self.connection.start_balatro_instance()

    def stop_balatro_instance(self):
        self.connection.stop_balatro_instance()

    def verifyimplemented(self):
        try:
            self.skip_or_select_blind(self, {})
            self.select_cards_from_hand(self, {})
            self.select_shop_action(self, {})
            self.select_booster_action(self, {})
            self.sell_jokers(self, {})
            self.rearrange_jokers(self, {})
            self.use_or_sell_consumables(self, {})
            self.rearrange_consumables(self, {})
            self.rearrange_hand(self, {})
        except NotImplementedError as e:
            print(e)
            sys.exit(0)
        except:
            pass

    def chooseaction(self):
        if self.G["state"] == State.GAME_OVER:
            self.running = False

        match self.G["waitingFor"]:
            case "start_run":
                return [
                    Actions.START_RUN,
                    self.stake,
                    self.deck,
                    self.seed,
                    self.challenge,
                ]
            case "skip_or_select_blind":
                return self.skip_or_select_blind(self.G)
            case "select_cards_from_hand":
                return self.select_cards_from_hand(self.G)
            case "select_shop_action":
                return self.select_shop_action(self.G)
            case "select_booster_action":
                return self.select_booster_action(self.G)
            case "sell_jokers":
                return self.sell_jokers(self.G)
            case "rearrange_jokers":
                return self.rearrange_jokers(self.G)
            case "use_or_sell_consumables":
                return self.use_or_sell_consumables(self.G)
            case "rearrange_consumables":
                return self.rearrange_consumables(self.G)
            case "rearrange_hand":
                return self.rearrange_hand(self.G)

    def run_step(self):
        self.connection.connect()
        if not self.verified:
            self.verifyimplemented()
            self.verified = True

        if self.running:
            jsondata = self.connection.poll_state()

            if "response" in jsondata:
                print(jsondata["response"])
            else:
                self.G = jsondata
                if self.G.get("waitingForAction", False):
                    cache_state(self.G["waitingFor"], self.G)
                    action = self.chooseaction()
                    if action == None:
                        raise ValueError("All actions must return a value!")

                    self.connection.send_action(action)

    def run(self):
        while self.running:
            self.run_step()
