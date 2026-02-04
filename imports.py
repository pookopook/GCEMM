# 라이브러리 임포트
import numpy as np
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)  # 모든 컬럼 다 보이게 설정

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import shapiro

# 지리 라이브러리
from geopy.distance import geodesic

# 전처리용
from itertools import combinations

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 윈도우용 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')  # 말굿 고딕 (Windows 기본 한글 폰트)
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 - 전처리, 모델, 평가
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 머신러닝 알고리즘 (필요에 따라 선택적으로 추가)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor

# 텍스트 처리
from rapidfuzz import process, fuzz

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore')

# # # 딥러닝 (선택사항)
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout

# 모델 해석
import shap

