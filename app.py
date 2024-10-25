import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Movie Rating Prediction with Python", page_icon="🎬")

st.title("Movie Rating Prediction with Python 🎬")

# Load the dataset
def load_data():
    df = pd.read_csv('IMDb-Movies-India.csv', encoding='latin1')
    return df

df = load_data()

# Display the dataset
st.header("IMDb Movies India Dataset")
st.write(df)

def main():
    d = pd.read_csv('imdb_top_1000.csv')
    d = d[np.isfinite(pd.to_numeric(d.Released_Year, errors="coerce"))]
    d = d[['Released_Year', 'Runtime', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'IMDB_Rating']]
    d["Runtime"] = d.Runtime.replace({'min': ''}, regex=True)

    df = d.copy()

    # Get dropdown values
    df = df.dropna()
    genres = df.Genre.replace({', ': ','}, regex=True)
    genres = genres.str.split(',').explode('Genre')
    genres = np.unique(genres)
    director = np.sort(df["Director"].unique())

    stars = pd.Series(pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']]).unique())
    stars = np.sort(stars)

    runtime = 0
    star1 = star2 = star3 = star4 = ''
    
    st.header("User Input Features")
    
    release_selection = st.number_input("Select the Release Year", step=1, min_value=1920, max_value=2050)
    runtime_selection = st.number_input("Enter the Duration (in minutes)", runtime)
    genre_selection = st.multiselect("Select the Genres", genres)
    director_selection = st.selectbox("Select the Director", director)
    star_selection = st.multiselect("Select the top 4 Stars of the Film:", stars, placeholder="Select no more than 4 Stars for better Prediction")

    if st.button("Predict"):
        if len(star_selection) >= 1:
            star1 = star_selection[0]
        if len(star_selection) >= 2:
            star2 = star_selection[1]
        if len(star_selection) >= 3:
            star3 = star_selection[2]
        if len(star_selection) >= 4:
            star4 = star_selection[3]

        genre_input = ' '.join(genre_selection)

        prediction_data = [release_selection, runtime_selection, genre_input, director_selection, star1, star2, star3, star4, np.nan]
        prediction_df = pd.DataFrame([prediction_data], columns=d.columns)
        d = pd.concat([d, prediction_df])

        for col in ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']:
            d[col] = d[col].astype('category').cat.codes

        prediction_data = d.iloc[-1].drop('IMDB_Rating').values.reshape(1, -1)
        d = d.drop(d.index[-1])

        training_columns = ['Released_Year', 'Runtime', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
        x, y = d[training_columns], d['IMDB_Rating']
        sc = StandardScaler()
        x = sc.fit_transform(x)
        prediction_data = sc.transform(prediction_data)

        model = LinearRegression()
        model.fit(x, y)
        predictions = model.predict(prediction_data)

        st.write(f"Predicted IMDB Rating: **{predictions.round(2)[0]}**")

if __name__ == "__main__":
    main()
