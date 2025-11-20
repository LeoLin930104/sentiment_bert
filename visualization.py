import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl
import pandas as pd
import numpy as np
import ast

TRUE_LABEL_PATH = "data/VAA/sentiment.csv"
TWEET_PATH = "data/tweets/sentiments/"
VAA_PATH = "data/VAA/vaaQuestions.csv"
PLOT_PATH = "data/plots"
mpl.rcParams['font.family'] = 'Hiragino Sans'


def plot_by_topic():
    for i in range(20):
        plt.figure(figsize=(8.1, 8.1))
        plt.gcf().canvas.mpl_connect('key_press_event', on_key)

        topic = VAA_questions['Topic'].loc[i]
        # Add axis labels and grid
        plt.title(f'Russell’s Emotion Space: {topic}')
        plt.xlabel('Valence (Pleasure)')
        plt.ylabel('Arousal (Activation)')
        plt.xticks(np.arange(-1, 1.1, 0.5))
        plt.yticks(np.arange(0, 1.1, 0.2))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        plt.grid(False)

        for j, (username, color) in enumerate(usernames.items()):
            label_val = sentiment[username].iloc[i]
            region_label = value_to_region.get(label_val)
            if region_label:
                x, y = region_centers[region_label]
                plt.scatter(x, y, color=color, s=8000-j*1000, alpha=0.3,
                            edgecolors=color, linewidth=0.8, zorder=3)

        for center, radius, color, label in regions:
            ellipse = Ellipse(center, width=radius*2, height=radius, color=color, alpha=0.15, zorder=0)
            plt.gca().add_patch(ellipse)
            plt.text(center[0], center[1], label, ha='center', va='center', fontsize=12, weight='bold', alpha=0.6)

        # Set axis limits to standard Russell space
        plt.xlim(-1, 1)
        plt.ylim(0, 1)

        # Add center lines for reference
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        plt.axvline(x=0.0, color='gray', linestyle='--', linewidth=1)

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=username, markerfacecolor=color) for username, color in usernames.items()]
        plt.legend(handles=handles, loc='lower center', ncol=4)

        for username, df in dfs.items():
            df = df[df['topic_idx'] == i]
            arousal = [x['arousal'] for x in df['russell']]
            valence = [x['valance'] for x in df['russell']]
            # Plot in Arousal–Valence space
            plt.scatter(valence, arousal, marker='x', alpha=0.5, s=100, c=usernames[username], label=username)

        plt.savefig(f"{PLOT_PATH}/topic_{i:02d}.png", dpi=300, bbox_inches='tight', pad_inches=0.3)
        # plt.show()

def plot_all():
    plt.figure(figsize=(8.1, 8.1))
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)

    plt.title(f'Russell’s Emotion Space: All topics')
    plt.xlabel('Valence (Pleasure)')
    plt.ylabel('Arousal (Activation)')
    plt.xticks(np.arange(-1, 1.1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.2))
    # Add axis labels and grid
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    plt.grid(False)

    # Set axis limits to standard Russell space
    plt.xlim(-1, 1)
    plt.ylim(0, 1)

    # Add center lines for reference
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0.0, color='gray', linestyle='--', linewidth=1)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=username, markerfacecolor=color) for username, color in usernames.items()]
    plt.legend(handles=handles, loc='lower center', ncol=4)

    for center, radius, color, label in regions:
        ellipse = Ellipse(center, width=radius*2, height=radius, color=color, alpha=0.15, zorder=0)
        plt.gca().add_patch(ellipse)
        plt.text(center[0], center[1], label, ha='center', va='center', fontsize=12, weight='bold', alpha=0.6)
    
    for i in range(20):

        topic = VAA_questions['Topic'].loc[i]
        # for j, (username, color) in enumerate(usernames.items()):
        #     label_val = sentiment[username].iloc[i]
        #     region_label = value_to_region.get(label_val)
        #     if region_label:
        #         x, y = region_centers[region_label]
        #         plt.scatter(x, y, color=color, s=8000-j*1000, alpha=0.3,
        #                     edgecolors=color, linewidth=0.8, zorder=3)


        for username, df in dfs.items():
            df = df[df['topic_idx'] == i]
            arousal = [x['arousal'] for x in df['russell']]
            valence = [x['valance'] for x in df['russell']]
            # Plot in Arousal–Valence space
            plt.scatter(valence, arousal, marker='x', alpha=0.5, s=100, c=usernames[username], label=username)

    plt.savefig(f"{PLOT_PATH}/all.png", dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()


if __name__ == "__main__":
    def on_key(event): plt.close()

    usernames = {
        "ecoyuri": 'green',
        "shinji_ishimaru": 'blue',
        "renho_sha": 'red',
        "toshio_tamogami": 'orange',
    }

    regions = [
        ((0.5, 0.75), 0.49, 'blue', 'CA'),     # CA
        ((0.5, 0.25), 0.49, 'blue',  'A'),     # A
        ((-0.5, 0.75), 0.49, 'red', 'CD'),     # CD
        ((-0.5, 0.25), 0.49, 'red',  'D'),     # D
        ((0.0, 0.5),  0.21, 'grey',  'N'),     # N
    ]

    value_to_region = {
        -2: 'CD',
        -1: 'D',
        0: 'N',
        1: 'A',
        2: 'CA',
    }
    region_centers = {label: center for center, _, _, label in regions}

    dfs = {}

    VAA_questions = pd.read_csv(VAA_PATH)
    sentiment = pd.read_csv(TRUE_LABEL_PATH, index_col=False)

    for username, color in usernames.items():
        print(f"▶ Loading Tweets from {TWEET_PATH}{username}_tweets.csv")
        df = pd.read_csv(f"{TWEET_PATH}{username}_tweets.csv")
        df['russell'] = df['russell'].apply(ast.literal_eval)
        dfs[username] = df[df['similarity']>0.4]

    plot_by_topic()
    plot_all()