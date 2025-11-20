from bert_score import score
import torch

MODEL_DIR = "model/bert-base-japanese-v3"   # folder that holds config.json, pytorch_model.bin, etc.

# Japanese texts
candidates = [
    "東京都は多摩都市モノレールを瑞穂町まで延伸する計画を立てています。現在の終点である東大和市の上北台から瑞穂町の箱根ケ崎まで延伸し、7駅を新設する予定です。また、開業すれば都内の市で唯一、駅がない武蔵村山市に初の鉄道駅ができることになります。",
    "これまで都立高校は国の支援により無償化され、私立高校については国の支援に上乗せして都が助成してきましたが、年収910万円未満が目安となっていました。2024年度からは都が所得制限を撤廃し、所得要件により国の支援が得られない場合は都が支援することにより「実質無償化」となります。保護者が都内に住んでいれば、都外の私立高校に通っても対象となりますが、都外から都内へ通学するケースは対象外です",

]       # tweet
references = [
    "小池都知事・細谷しょうこ演説 #東久留米市長選挙 の細谷しょうこ演説会東久留米駅 市政を力強く前に進めるためには東京都との力強いパートナーシップも重要です。#小池百合子都知事 と連携し、進めていきます。ふたりによる #東久留米の未来ビジョン",
    "高校の授業料の無償化は税ではなく国債発行で出来る。今なお税が財源だという頓珍漢がいる。自公と維新は高校授業料の無償化で国民民主党提案の減税案を潰した。高校の授業料無償化で恩恵を受ける人など限られている。全国民を対象とした減税を何故自公はやらないのか。それは国民生活など全く考慮の外だからだ。自公は腐っているが維新もこれに加担した。今、減税することなしに国民生活を豊かにすることは出来ない",

]   # topic or seed sentence

P, R, F1 = score(
    candidates,            # list[str]  ─ your tweets or seed sentences
    references,            # list[str]  ─ sentences to compare against
    model_type=MODEL_DIR,  # local path or HF repo name
    num_layers=12,         # BERT‑base has 12 layers
    lang="ja",             # keeps tokenizer’s Japanese normalisation
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    batch_size=32,         # adjust to fit GPU/CPU memory
    verbose=True
)
print(F1)
#print(f"Precision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}")
