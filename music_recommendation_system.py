#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import base64


# In[2]:


CLIENT_ID = '605cdf1a95cb48c0bb25eb7a87eb0316'
CLIENT_SECRET = '1d5cf96d201d4b808170cc5bedd8c8b4'


# In[3]:


client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
client_credentials_base64 = base64.b64encode(client_credentials.encode())


# In[4]:


token_url = 'https://accounts.spotify.com/api/token'
headers = {
    'Authorization': f'Basic {client_credentials_base64.decode()}'
}
data = {
    'grant_type': 'client_credentials'
}
response = requests.post(token_url, data=data, headers=headers)


# In[5]:


if response.status_code == 200:
    access_token = response.json()['access_token']
    print("Access token obtained successfully.")
else:
    print("Error obtaining access token.")
    exit()


# In[6]:





# In[7]:


import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth


# In[20]:


def get_trending_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    playlist_tracks = sp.playlist_tracks(playlist_id, fields='items(track(id, name, artists, album(id, name)))')
    music_data = []
    for track_info in playlist_tracks['items']:
        track = track_info['track']
        track_name = track['name']
        artists = ', '.join([artist['name'] for artist in track['artists']])
        album_name = track['album']['name']
        album_id = track['album']['id']
        track_id = track['id']
        audio_features = sp.audio_features(track_id)[0] if track_id != 'Not available' else None
        try:
            album_info = sp.album(album_id) if album_id != 'Not available' else None
            release_date = album_info['release_date'] if album_info else None
        except:
            release_date = None
        try:
            track_info = sp.track(track_id) if track_id != 'Not available' else None
            popularity = track_info['popularity'] if track_info else None
        except:
            popularity = None
        track_data = {
            'Track Name': track_name,
            'Artists': artists,
            'Album Name': album_name,
            'Album ID': album_id,
            'Track ID': track_id,
            'Popularity': popularity,
            'Release Date': release_date,
            'Duration (ms)': audio_features['duration_ms'] if audio_features else None,
            'Explicit': track_info.get('explicit', None),
            'External URLs': track_info.get('external_urls', {}).get('spotify', None),
            'Danceability': audio_features['danceability'] if audio_features else None,
            'Energy': audio_features['energy'] if audio_features else None,
            'Key': audio_features['key'] if audio_features else None,
            'Loudness': audio_features['loudness'] if audio_features else None,
            'Mode': audio_features['mode'] if audio_features else None,
            'Speechiness': audio_features['speechiness'] if audio_features else None,
            'Acousticness': audio_features['acousticness'] if audio_features else None,
            'Instrumentalness': audio_features['instrumentalness'] if audio_features else None,
            'Liveness': audio_features['liveness'] if audio_features else None,
            'Valence': audio_features['valence'] if audio_features else None,
            'Tempo': audio_features['tempo'] if audio_features else None,
         }
        music_data.append(track_data)
    df = pd.DataFrame(music_data)

    return df

        


# In[21]:


playlist_id = '5ABHKGoOzxkaa28ttQV9sE' #That's the top 100 popular songs on Spotify
music_df = get_trending_playlist_data(playlist_id, access_token)
print(music_df)

print(music_df.isnull().sum())


# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

data = music_df


# In[14]:


def calculate_weighted_popularity(release_date):
    release_date = datetime.strptime(release_date, '%Y-%m-%d')
    time_span = datetime.now() - release_date
    weight = 1 / (time_span.days + 1)
    return weight
    


# In[15]:


scaler = MinMaxScaler()
music_features = music_df[['Danceability', 'Energy', 'Key', 
                           'Loudness', 'Mode', 'Speechiness', 'Acousticness',
                           'Instrumentalness', 'Liveness', 'Valence', 'Tempo']].values
music_features_scaled = scaler.fit_transform(music_features)


# In[17]:


# a function to get content-based recommendations based on music features
def content_based_recommendations(input_song_name, num_recommendations=5):
    if input_song_name not in music_df['Track Name'].values:
        print(f"'{input_song_name}' not found in the dataset. Please enter a valid song name.")
        return
    input_song_index = music_df[music_df['Track Name'] == input_song_name].index[0]
    similarity_scores = cosine_similarity([music_features_scaled[input_song_index]], music_features_scaled)
    similar_song_indices = similarity_scores.argsort()[0][::-1][1:num_recommendations + 1]
    content_based_recommendations = music_df.iloc[similar_song_indices][['Track Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]

    return content_based_recommendations
        
    


# In[22]:


def hybrid_recommendations(input_song_name, num_recommendations=5, alpha=0.5):
    if input_song_name not in music_df['Track Name'].values:
        print(f"'{input_song_name}' not found in the dataset. Please enter a valid song name.")
        return
    content_based_rec = content_based_recommendations(input_song_name, num_recommendations)
    popularity_score = music_df.loc[music_df['Track Name'] == input_song_name, 'Popularity'].values[0]
    weighted_popularity_score = popularity_score * calculate_weighted_popularity(music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0])
    input_song_data = {
        'Track Name': [input_song_name],
        'Artists': [music_df.loc[music_df['Track Name'] == input_song_name, 'Artists'].values[0]],
        'Album Name': [music_df.loc[music_df['Track Name'] == input_song_name, 'Album Name'].values[0]],
        'Release Date': [music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0]],
        'Popularity': [weighted_popularity_score]
    }
    input_song_df = pd.DataFrame(input_song_data)
    hybrid_recommendations = pd.concat([content_based_rec, input_song_df], ignore_index=True)
    hybrid_recommendations = hybrid_recommendations.sort_values(by='Popularity', ascending=False)
    hybrid_recommendations = hybrid_recommendations[hybrid_recommendations['Track Name'] != input_song_name]

    return hybrid_recommendations
    
    
    







