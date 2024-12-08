import gymnasium as gym
from balatro_connection import Actions
import time
from gamestates import cache_state


class BalatroBaseEnv(gym.Env):
    def __init__(self, env_config):
        self.balatro_connection = None
        self.port = env_config.worker_index + 12348
        if "agency_states" in env_config:
            self.agency_states = env_config["agency_states"]
        else:
            self.agency_states = None
        self.hand_size = 8
        self.deck = "Blue Deck"
        self.stake = 1
        self.challenge = None
        self.seed = None

    def get_gamestate(self, pending_action=None):
        G = self.balatro_connection.poll_state()
        # Wait for the game to be in a state where we can select cards
        i = 0
        while True:
            i += 1
            if i % 100 == 0:
                print(
                    f'spinning in get_gamestate {G.get("waitingFor", None)} {G.get("waitingForAction", False)} {G.get("lastAction", None)} {pending_action}'
                )
            if G.get("waitingForAction", False):
                if pending_action is not None:
                    last_action = G.get("lastAction", None)
                    if last_action == pending_action:
                        pending_action = None
                    else:
                        # time.sleep(0.05)
                        G = self.balatro_connection.poll_state()
                        continue

                auto_action = self.hardcoded_action(G)
                # cache_state(G.get("waitingFor", None), G)
                # print(f"Auto action: {auto_action}")
                if auto_action is not None:
                    if (
                        self.balatro_connection.last_action is None
                        or auto_action[0] != self.balatro_connection.last_action[0]
                    ):
                        # last_auto_action = auto_action[0]
                        self.balatro_connection.send_action(auto_action)
                        pending_action = auto_action[0].value
                else:
                    break

            # time.sleep(0.05)
            G = self.balatro_connection.poll_state()

        return G

    def hardcoded_action(self, game_state):
        if (
            self.agency_states is not None
            and game_state["waitingFor"] in self.agency_states
        ):
            # print(f"agency action for {game_state['waitingFor']}")
            return None
        match game_state["waitingFor"]:
            case "start_run":
                return [
                    Actions.START_RUN,
                    self.stake,
                    self.deck,
                    self.seed,
                    self.challenge,
                ]
            case "skip_or_select_blind":
                return [Actions.SELECT_BLIND]
            case "select_cards_from_hand":
                return [Actions.PLAY_HAND, [1]]
            case "select_shop_action":
                return [Actions.END_SHOP]
            case "select_booster_action":
                return [Actions.SKIP_BOOSTER_PACK]
            case "sell_jokers":
                return [Actions.SELL_JOKER, []]
            case "rearrange_jokers":
                return [Actions.REARRANGE_JOKERS, []]
            case "use_or_sell_consumables":
                return [Actions.USE_CONSUMABLE, []]
            case "rearrange_consumables":
                return [Actions.REARRANGE_CONSUMABLES, []]
            case "rearrange_hand":
                return [Actions.REARRANGE_HAND, []]

        return None

    def start_new_game(self):
        print("Starting new game")
        G = self.get_gamestate()
        if "current_round" in G:
            current_round = G["current_round"]
            if (
                current_round["hands_played"] == 0
                and current_round["discards_used"] == 0
            ):
                return
        self.balatro_connection.send_cmd("MENU")
        return self.get_gamestate(pending_action="MENU")
