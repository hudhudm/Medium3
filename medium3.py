import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv('YOUTUBE_CHANNELS_DATASET.csv')

df['Subscribers'] = df['Subscribers'].replace({'M': '', ',': ''}, regex=True).apply(lambda x: float(x) * 1_000_000 if x != '0' else 0)
df['Views'] = df['Views'].replace({'M': '', ',': ''}, regex=True).apply(lambda x: float(x) * 1_000_000 if x != '0' else 0)
df['Uploads'] = df['Uploads'].replace({',': ''}, regex=True).apply(lambda x: float(x) if x != '0' else 0)

if 'country' in df.columns:
    df = df.drop(columns=['country'])

df_features = df[['Subscribers', 'Views', 'Uploads']]
df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

scaler = StandardScaler()
df_normalize = scaler.fit_transform(df_features)
similarity_matrix = cosine_similarity(df_normalize)

mrbeast_rank = 0
similarity_score = similarity_matrix[mrbeast_rank]

top_10_similar_ranks = similarity_score.argsort()[-11:-1][::-1]
top_10_similar_channels = df.iloc[top_10_similar_ranks].copy()
top_10_similar_channels.loc[:, 'Similarity'] = similarity_score[top_10_similar_ranks]

print(top_10_similar_channels[['Ranking', 'Username', 'Subscribers', 'Views', 'Uploads', 'Similarity']])
