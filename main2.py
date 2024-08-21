import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = sns.load_dataset('iris')
df

x = 4*np.random.rand(100)
y = np.sin(2*x + 1) + 0.1* np.random.randn(100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

st.sidebar.title('Classification')
classifier = st.sidebar.selectbox('Select classifier', ('KNN', 'DT', 'RF', 'SVM', 'NN'))
k = st.sidebar.slider('K', 1, 120, 3)
from sklearn.metrics import accuracy_score
y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.scatter(x, y_pred)
st.pyplot(fig)

