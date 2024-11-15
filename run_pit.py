import json
import os
import random

from centuryrl.century.strategies import strategy_from_string
from centuryrl.rl.eval.selfplay import pit


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
