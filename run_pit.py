import json
import os
import random

from boardrl.games import games_library, strategy_from_string
from boardrl.rl.eval.selfplay import pit


def populate_strategies():
    strategies = [
        "random",
        "random_buy",
        "all_actions_then_random_buy",
        "no_actions_random_buy",
        "pick_best_mc_value:10",
    ]
    # find all .pth files in all directories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pth"):
                strategies.append(f"argmax:{os.path.join(root, file)}")
                strategies.append(f"policy_sampling:{os.path.join(root, file)}")

    return strategies


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--num_games", type=int, default=32)
    parser.add_argument("--player1", type=str, default=None)
    parser.add_argument("--player2", type=str, default=None)
    opts = parser.parse_args()

    all_games = []
    try:
        with open("games.json", "r") as f:
            all_games = json.load(f)
    except FileNotFoundError:
        print("No games.json file found")

    strategies = populate_strategies()
    print(strategies)
    game_desc = games_library("century")
    for _ in range(opts.games):
        player1 = opts.player1 or random.choice(strategies)
        player2 = opts.player2 or random.choice(strategies)
        pit_results = pit(
            game_desc.make_game,
            [strategy_from_string(player1), strategy_from_string(player2)],
            opts.num_games,
            opts.max_len,
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
