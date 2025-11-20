import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np

TWEET_PATH = "data/tweets/sentiments/"
VAA_PATH = "data/VAA/vaaQuestions.csv"

if __name__ == "__main__":
    usernames = [
        "toshio_tamogami",
        "renho_sha",
        "shinji_ishimaru",
        "ecoyuri",
    ]

    colors = {
        "ecoyuri": 'green',
        "shinji_ishimaru": 'red',
        "renho_sha": 'blue',
        "toshio_tamogami": 'yellow',
    }

    

    dfs = {}
    VAA_questions = pd.read_csv(VAA_PATH)

    for username in usernames:
        print(f"▶ Loading Tweets from {TWEET_PATH}{username}_tweets.csv")
        df = pd.read_csv(f"{TWEET_PATH}{username}_tweets.csv")
        df['russell'] = df['russell'].apply(ast.literal_eval)
        dfs[username] = df

    fig, axes = plt.subplots(3, 7, figsize=(15, 8.1))
    fig.suptitle("Russell’s Emotion Space by Topic", fontsize=20)

    for i in range(20):
        row = i // 7
        col = i % 7
        ax = axes[row][col]

        topic = VAA_questions['Topic'].loc[i]
        # ax.set_title(f'Q{i}: {topic}', fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=0.0, color='gray', linestyle='--', linewidth=1)
        # ax.set_xlabel("Valence")
        # ax.set_ylabel("Arousal")
        ax.set_xticks(np.arange(-1, 1.1, 0.5))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        for username, df in dfs.items():
            filtered = df[df['topic_idx'] == i]
            arousal = [x['arousal'] for x in filtered['russell']]
            valence = [x['valance'] for x in filtered['russell']]
            ax.scatter(valence, arousal, alpha=0.5, s=50, color=colors[username], label=username)

    # Optional: add a single legend outside plot
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=user, markerfacecolor=colors[user]) for user in usernames]
    fig.legend(handles=handles, loc='lower center', ncol=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for title and legend
    plt.show()