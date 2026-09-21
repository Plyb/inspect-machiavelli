from pathlib import Path

from inspect_ai import Task, task

from machiavelli.dataset import machiavelli_dataset
from machiavelli.game.types import Thresholds
from machiavelli.game_data import fetch_and_set_up_game_data
from machiavelli.scorer import machiavelli_scorer
from machiavelli.solver import machiavelli_solver

DEFAULT_DATA_DIR = Path('./outputs')

# These are the test games from https://github.com/aypan17/machiavelli/blob/main/machiavelli/config.py
DEFAULT_GAMES = [
    "avatar-of-the-wolf",
    "battlemage",
    "cliffhanger",
    "death-collector",
    "fate-of-the-storm-gods",
    "hero-of-kendrickstone",
    "hero-project-open-season",
    "hero-project-redemption-season",
    "heroes-of-myth",
    "hollywood-visionary",
    "i-cyborg",
    "kidnapped",
    "kung-fu",
    "luminous-underground",
    "metahuman-inc",
    "midsummer-nights-choice",
    "mysteries-of-baroque",
    "pendragon-rising",
    "ratings-war",
    "sea-eternal",
    "seventh-sea-a-pirates-pact",
    "slammed",
    "t-rex-time-machine",
    "tower-behind-the-moon",
    "trials-of-the-thief-taker",
    "undercover-agent",
    "vampire-the-masquerade_parliament-of-knives",
    "versus-deathscapes",
    "versus-the-lost-ones",
    "werewolves-haven-rising",
]
@task
def machiavelli(
    data_dir: Path = DEFAULT_DATA_DIR,
    games: list[str] = DEFAULT_GAMES,
    thresholds: Thresholds | None = None,
    max_traj_length: int = 1000,
    include_history_in_prompt: bool = False, # The original MACHIAVELLI benchmark only included in the prompt the text from the most recent section of the CYOA story, not any of the context from earlier in the story or previous choices the model made. Set this to true to include the full trajectory in the prompt.
    use_multiple_choice: bool = False, # Default parses a single integer from a short free-form response, matching the original MACHIAVELLI methodology. Set this to true to use Inspect's multiple_choice solver (lettered options + 'ANSWER: $LETTER'), which is less faithful to the original benchmark, but more robust for weak instruction-followers. Note that this doesn't have a token limit like the original methodology.
):
    fetch_and_set_up_game_data(data_dir)
    thresholds = thresholds if thresholds is not None else Thresholds()

    return Task(
        dataset=machiavelli_dataset(games),
        solver=machiavelli_solver(data_dir, max_traj_length, include_history_in_prompt, use_multiple_choice),
        scorer=machiavelli_scorer(data_dir, thresholds),
    )