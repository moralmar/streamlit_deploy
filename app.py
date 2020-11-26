"""
1) pip install streamlit (in case of new env)
2) go to file
3) streamlit run app.py


"""
import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
import shap


# DATA_PATH_ABS = r"D:\\c-square\\data\\Reporting\Heatmaps\\WMCH2019\\"  # <-- old
DATA_PATH_ABS = "../streamlitapp/"
FILE_NAME = 'data_ready_to_predict.xlsx'
COL_TO_PREDICT = 'SATIS_1_top2'
suffix_top2 = 'top2'



# get and prepare data
# ----------------------
st.title('Explain my Survey Data')
st.sidebar.title('Sidebar Title')
data_load_state = st.text('Loading data...')


@st.cache
def helper_read_excel(path):
    return pd.read_excel(path)


data_load_state.text("Done!")
data = helper_read_excel(path=os.path.join(DATA_PATH_ABS, FILE_NAME))
data_load_state.text("Done! (using st.cache)")

cols_transformed = []
cols_original = []
for col in data.columns:
    if col.endswith(suffix_top2):
        cols_transformed.append(col)
    else:
        cols_original.append(col)
cols_features = cols_transformed
cols_features.remove(COL_TO_PREDICT)
X = data[cols_features]
y = data[COL_TO_PREDICT]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)




st.subheader('Your transformed data looks like:')
st.table(X_train.head(5))

# add corr plot
# ------------------
# Filter Method: Spearman's Cross Correlation > 0.95...maybe


def create_corr_plot(data):
    """create simple correlation plot

    REMEMBER: the plot goes to the axe, but the figure has to be plotted

    Example:
        corr_fig, _ = create_corr_plot(data=X_train)
        st.pyplot(corr_fig)
    """
    corr_matrix = data.corr(method="spearman").abs()
    # Draw the heatmap
    sns.set(font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(corr_matrix, cmap="YlGnBu", square=True)
    return fig, ax


st.title("Spearman Correlation")
corr_fig, _ = create_corr_plot(data=X_train)
st.pyplot(corr_fig)


###############################################################################
#                               4. Classifiers                                #
###############################################################################
# Create list of tuples with classifier label and classifier object
# reference: https://towardsdatascience.com/model-design-and-selection-with-scikit-learn-18a29041d02a
classifiers = {}
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
classifiers.update({"Random Forest": RandomForestClassifier()})
classifiers.update({"DTC": DecisionTreeClassifier()})
classifiers.update({"ETC": ExtraTreeClassifier()})

cls_list = [classifier_key for classifier_key, _ in classifiers.items()]
classifier = st.selectbox('Which algorithm?', cls_list)
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    cls_list
)
###############################################################################
#                             5. Hyper-parameters                             #
###############################################################################
parameters = {}
# Update dict with Extra Trees
parameters.update({"Extra Trees Ensemble": {
                                            "classifier__n_estimators": [200],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"]     ,
                                            "classifier__n_jobs": [-1]
                                             }})
parameters.update({"Random Forest": {
                                    "classifier__n_estimators": [200],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__criterion" :["gini", "entropy"]     ,
                                    "classifier__n_jobs": [-1]
                                     }})
# Update dict with Decision Tree Classifier
parameters.update({"DTC": {
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# Update dict with Extra Tree Classifier
parameters.update({"ETC": {
                            "classifier__criterion" :["gini", "entropy"],
                            "classifier__splitter": ["best", "random"],
                            "classifier__class_weight": [None, "balanced"],
                            "classifier__max_features": ["auto", "sqrt", "log2"],
                            "classifier__max_depth" : [1,2,3, 4, 5, 6, 7, 8],
                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                             }})

# REMEMBER: which classifiers has been chosen: classifier
# write out json
st.write(parameters[classifier])
# for now, only RF is to be selected
classifier = "Random Forest"
st.subheader(classifier)


rf = RandomForestClassifier()
param_dist={
        'bootstrap': [True],
        'criterion': ['entropy'],  # removed: gini
        'max_features': ['sqrt'],  # removed: log20, None
        'max_depth': [9],
        'min_samples_leaf': [15],
        'n_estimators': [10, 15]
        }

rf.fit(X_train, y_train)
cv_rf = GridSearchCV(rf,
                     cv=5,
                     param_grid=param_dist,
                     scoring='f1',
                     n_jobs=3)

cv_rf.fit(X_train, y_train)
st.write(cv_rf.best_estimator_)
st.write(cv_rf.best_params_)
# By default, parameter search uses the score function of the estimator to evaluate a parameter setting.
# These are the sklearn.metrics.accuracy_score for classification and sklearn.metrics.r2_score for regression.
st.write(cv_rf.best_score_)  # Mean cross-validated score of the best_estimator
with st.beta_expander("See explanation"):
    st.write("""
            The chart above shows some numbers I picked for you.
            I rolled actual dice for these, so they're *guaranteed* to
            be random.
            """)
    st.image("https://static.streamlit.io/examples/dice.jpg")

col1, col2, col3 = st.beta_columns(3)
with col1:
    st.header("A cat")
    st.image("https://c-square.ch/wp-content/uploads/2020/08/Mira_Lukas_3737_Klein-1-300x300.jpg", use_column_width=True)

with col2:
    st.header("A dog")
    st.image("https://c-square.ch/wp-content/uploads/2020/08/Severin.png", use_column_width=True)

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", use_column_width=True)


# SHAP
# calculate shapley values
explainer = shap.TreeExplainer(cv_rf.best_estimator_)
shap_values = explainer.shap_values(X_train)

import streamlit.components.v1 as components


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# explain model prediction results
# def explain_model_prediction(data):
#     # Calculate Shap values
#     shap_values = explainer.shap_values(data)
#     p = shap.force_plot(explainer.expected_value[1], shap_values[1], data)
#     return p, shap_values
# p, shap_values = explain_model_prediction(X_train)

st.subheader('Model Prediction Interpretation Plot')
fig, ax = plt.subplots(nrows=1, ncols=1)
shap.summary_plot(shap_values[1], X_train)
st.pyplot(fig)
# st_shap(p)

# st_shap(shap.summary_plot(shap_values[1], X_train))

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:]))
# visualize the training set predictions
# st_shap(shap.force_plot(explainer.expected_value, shap_values[1], X_train), 400)

