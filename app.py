"""
    @Author: Minh Duc
    @Since: 12/24/2021 3:12 PM
"""
import copy
import json

import streamlit as st
import pandas as pd
import base64
import numpy as np
import PIL.Image as Image
import requests
import io
from NCFModel import NCFModel
from imdb_request import get_film_info, get_title_id
model = NCFModel()

st.set_page_config(
        page_title="Group 8 - NCF",
    )

st.title('Group 8')
st.markdown("""
* **Deployed by: Hồ Ngọc Minh Đức - 19127368**
\n**Other members:** 
* 19127353 - Lê Tấn Đạt
* 19127429 - Trần Tuấn Kha
* 19127651 - Trần Anh Túc
""")
st.title('Recommender System')
st.title('Neural Collaborative Filtering')
st.markdown("""
This projects use Movielens 100K dataset!
* **Python libraries:** base64, pandas, streamlit, numpy, ...
* **Paper Source:** [paperswithcode.com](https://paperswithcode.com/paper/neural-collaborative-filtering)
* **Model reference:** [github.com/microsoft](https://github.com/microsoft/recommenders)
* **Source Code: ** [github.com/MinhDuc](https://github.com/steveho29/NCFApp)
""")

st.sidebar.header('Filter Dataset')


def filedownload(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download CSV File</a>'
    return href

@st.cache
def load_train_data():
    train_data = copy.deepcopy(model.train_data)
    # train_data.sort_values(by=['userID', 'itemID'], ascending=[True, True])
    item_data = copy.deepcopy(model.item_data)
    return item_data, train_data


def getImage(url):
    return Image.open(io.BytesIO(requests.get(url).content))


item_data, train_data = load_train_data()

# genre_data = sorted(train_data['genre'].unique())
genre_data = model.genre_columns

# Sidebar - Genre selection
selected_genre = st.sidebar.multiselect('Movie Genres', genre_data, genre_data)

# Sidebar - Rate selection
rating = [i for i in range(1, 6)]
selected_rating = st.sidebar.multiselect('Ratings', rating, rating)

# Filtering data
df_train_data_selected = train_data[(train_data.genre.isin(selected_genre)) & (train_data.rating.isin(selected_rating))]
st.header('Dataset Movielens 100k Ratings')

st.write('Collected from 943 Users - 1682 Movies')
st.write('Data Dimension: ' + str(df_train_data_selected.shape[0]) + ' rows and ' + str(
    df_train_data_selected.shape[1]) + ' columns.')
des = train_data.describe().drop(columns=['userID', 'itemID'])[:2]

st.dataframe(df_train_data_selected)

# -------------- DOWNLOAD BUTTON -----------------
st.markdown(filedownload(df_train_data_selected, "Group8 - MovieLens 100k Dataset.csv"), unsafe_allow_html=True)

st.write('Description: ')
st.dataframe(des)

genre_bar_chart = [(len(train_data[train_data['genre'] == genre]))for genre in genre_data]
genre_bar_chart = pd.DataFrame(np.array(genre_bar_chart).reshape(1, len(genre_data)), columns=genre_data)
st.bar_chart(genre_bar_chart)


# -------------- TRAINING LOSS LINE CHART-----------------
st.header('Training Loss')
neumf_error = np.load('Error_neumf.npy')
gmf_error = np.load('Error_gmf.npy')
mlp_error = np.load("Error_mlp.npy")
st.subheader("MLP Model")
st.write('MLP Model training loss\'s really weird (is a line) but NeuMF is still correct')
st.line_chart(pd.DataFrame(np.array([mlp_error]).T, columns=["MLP"]))

st.subheader("NeuMF & GMF Model")
st.write('NeuMF & GMF Model has the training loss convergence (exactly same as paper)')
st.line_chart(pd.DataFrame(np.array([gmf_error, neumf_error]).T, columns=["GMF", "NeuMF"]))

st.header("Evaluation Metrics")
metrics = {}
metric_columns = []
for modelType in ['neumf', 'gmf', 'mlp']:
    metric = json.load(open('metrics_'+modelType+'.json'))
    metric_columns = metric.keys()
    metrics[modelType] = metric.values()
st.dataframe(pd.DataFrame.from_dict(metrics, orient='index', columns=metric_columns))

# Heatmap
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_train_data_selected.to_csv('output.csv', index=False)
#     df = pd.read_csv('output.csv')
#
#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot()

st.sidebar.header('Choose User to recommend movie')
selected_user = st.sidebar.selectbox('UserID', sorted(train_data['userID'].unique()))

recommendations = model.get_recommendations(selected_user, 5)

isShowImage = st.sidebar.checkbox('Show Movie Image')
# isShowMovieDescription = st.sidebar.checkbox('Show Movie Description')


st.header("Applying model for Recommendation")
st.write("Use recommendation option in sidebar to see the model working ")
st.subheader("Recommendation for User: #"+str(selected_user))
for i, (itemID, value) in enumerate(recommendations):
    # urls = [urllib.parse.unquote(url.replace("http://us.imdb.com/M/title-exact?", "")) for url in item_data[item_data['itemID'] == itemID]['url'].tolist()]
    title = item_data[item_data['itemID'] == itemID]['movie_title'].tolist()[0]
    st.subheader(str(i + 1) + ". " + title)
    st.markdown(f"""
        * **Predict Rating:** {str(round(float(value) * 5, 3)) + "/5"}
        * **MovieID:** {itemID}
        """)
    if isShowImage:
        try:
            id = get_title_id(title)
            print(id)
            if id:
                info = get_film_info(id)
                print(info)
                st.image(getImage(info["Poster"]), width=200)
                st.write(info)
        except Exception as e:
            # print(e)
            st.write("IMDB server search not found!")


