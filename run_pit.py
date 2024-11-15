import json
import os
import random
import torch

from centuryrl.century.strategies import (
    RandomStrategy,
    RandomBuyStrategy,
    AllActionsThenRandomBuyStrategy,
    NoActionsRandomBuyStrategy,
    ArgmaxStrategy,
    PolicySamplingStrategy,
)
from main import pit
from model import Model


def populate_strategies():
    strategies = [
        "random",
        "random_buy",
        "all_actions_then_random_buy",
        "no_actions_random_buy",
    ]
    # find all .pth files in all directories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pth"):
                strategies.append(f"argmax:{os.path.join(root, file)}")
                strategies.append(f"policy_sampling:{os.path.join(root, file)}")

    return strategies


def load_model(model_path):
    model = Model(256, 8)
    model.load_state_dict(torch.load(model_path)["model"])
    model.cuda()
    model.eval()
    return model


def strategy_from_string(strategy_string):
    if strategy_string == "random":
        return RandomStrategy()
    elif strategy_string == "random_buy":
        return RandomBuyStrategy()
    elif strategy_string == "all_actions_then_random_buy":
        return AllActionsThenRandomBuyStrategy()
    elif strategy_string == "no_actions_random_buy":
        return NoActionsRandomBuyStrategy()
    elif strategy_string.startswith("argmax"):
        model_path = strategy_string.split(":")[1]
        return ArgmaxStrategy(load_model(model_path))
    elif strategy_string.startswith("policy_sampling"):
        model_path = strategy_string.split(":")[1]
        return PolicySamplingStrategy(load_model(model_path))
    else:
        raise ValueError(f"Unknown strategy: {strategy_string}")


def main():
    all_games = []
    try:
        with open("games.json", "r") as f:
            all_games = json.load(f)
    except FileNotFoundError:
        print("No games.json file found")

    strategies = populate_strategies()
    print(strategies)
    while True:
        player1 = random.choice(strategies)
        player2 = random.choice(strategies)
        pit_results = pit(
            [strategy_from_string(player1), strategy_from_string(player2)], 32, 200
        )
        for points in pit_results.my_points(0):
            all_games.append({"player1": player1, "player2": player2, "points": points})

        done = False
        while not done:
            try:
                with open("games.json", "w") as f:
                    json.dump(all_games, f, indent=4)
                done = True
            except KeyboardInterrupt:
                print("Interrupted")
                done = True

        print(
            f"Player1: {player1}, Player2: {player2}, Points: {pit_results.my_points(0)}"
        )


if __name__ == "__main__":
    main()
