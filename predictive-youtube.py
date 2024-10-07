import env
from typing import Dict, Any, List

import isodate
from googleapiclient.discovery import build 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

api_key=env.TOKEN_API

youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_videos(channel_id:str, youtube:Any)-> List[Dict[str,str]]:
    """Obtiene los últimos 100 videos de un canal de YouTube
    y devuelve información sobre cada uno

    Args:
        channel_id (str): El ID único del canal que se quieren obtener datos
        youtube (any): El cliente de la API de youtube

    Returns:
        list[dict[str,str]]: Una lista de diccionarios, donde cada diccionario contiene
        la siguiente información sobre el video:
        'video_id': el ID del video en YouTube
        'title': el título del video
        'publishedAt': La fecha y hora en la que fue publicado (en formato ISO 8601)
    
    Notas:
        -La función solo devuelve hasta 50 videos, ordenados del más reciente al más antiguo.
        - Si el canal tiene más de 100 videos, solo obtendrá los más recientes,
        ya que 'maxResults' está limitado a 50.
        - Para tener más de 50 videos sería necesario crear paginaciones.
    """
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=50,
        order="date"
    )
    response=request.execute()
    
    video_data=[]
    
    for item in response['items']:
        video_id=item['id']['videoId']
        video_data.append({
            'video_id':video_id,
            'title':item['snippet']['title'],
            'publishedAt':item['snippet']['publishedAt']
        })
    return video_data

def get_video_stats(video_id:str, youtube:Any)-> Dict[str, Any]:
    """Obtiene estadísiticas básicas y la duración de un video en YouTube.

    Args:
        video_id (str): El ID único del video que se quieren obtener datos
        youtube (any): El cliente de la API de youtube

    Returns:
        Dict[str, Any]: Nos devuelve un diccionario con las siguientes estadísticas.
        - 'views': El número de visualizaciones del video (int)
        - 'likes': El número de likes que tiene el video (int)
        - 'duration': la duración del video en segundos (float)
        - 'comments': El número de comentarios que recive el video (int)
    
    Notas:
        - Si alguna estádistica no está disponible
        (por ejemplo un video sin comentarios recive el valor 0 por defecto)
        - La duración de los videos es convertida a ISO 8601 por la facilidad de uso.
    """
    request=youtube.videos().list(
        part="statistics,contentDetails",
        id=video_id
    )
    response=request.execute()
    
    stats=response['items'][0]['statistics']
    content_details=response['items'][0]['contentDetails']
    
        # Convertir la duración de ISO 8601 a segundos
    duration_iso = content_details.get('duration', 'PT0S')
    duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
    
    return {
        'views':int(stats.get('viewCount', 0)),
        'likes':int(stats.get('likeCount', 0)),
        'duration':duration_seconds,
        'comments':int(stats.get('commentCount', 0)),
    }

channel_id=env.ID_YOUTUBE
videos=get_channel_videos(channel_id, youtube)

videos_stats = [get_video_stats(video['video_id'], youtube) for video in videos]
    
for i, video in enumerate(videos):
    videos[i].update(videos_stats[i])
    
df = pd.DataFrame(videos)
print(df.head(10))

df['engagement_rate'] = (df['likes'] + df['comments'])/df['views']

# Selecciona las características que quieres usar para predecir
features = ['views', 'likes', 'comments','duration']  # Puedes ajustar las características
target = 'engagement_rate'

scaler = StandardScaler()

# Divide los datos en entrenamiento y prueba
X = df[features]
y = df[target]

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Entrena un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realiza predicciones
y_pred = model.predict(X_test)

# Evalúa el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

corr_matrix = df[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

sns.boxplot(data=df[features])
plt.show()

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

