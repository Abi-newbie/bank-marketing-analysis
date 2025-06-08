Python 3.13.4 (tags/v3.13.4:8a526ec, Jun  3 2025, 17:46:04) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import pandas as pd
df = pd.read_csv
df = pd.read_csv("C:\Users\Abigail Hefty\Downloads\bank_marketing_cleaned.csv")
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape
import pandas as pd
df = pd.read_csv(r"C:\Users\Abigail Hefty\Downloads\bank_marketing_cleaned.csv")
print (df.head())
   age       job  marital  ... euribor3m nr_employed   y
0   57  services  married  ...     4.857      5191.0  no
1   37  services  married  ...     4.857      5191.0  no
2   40    admin.  married  ...     4.857      5191.0  no
3   56  services  married  ...     4.857      5191.0  no
4   45  services  married  ...     4.857      5191.0  no

[5 rows x 21 columns]

print(df.ifo())
Traceback (most recent call last):
  File "<pyshell#5>", line 1, in <module>
    print(df.ifo())
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\generic.py", line 6318, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'ifo'. Did you mean: 'info'?
print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41187 entries, 0 to 41186
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41187 non-null  int64  
 1   job             41187 non-null  object 
 2   marital         41187 non-null  object 
 3   education       41187 non-null  object 
 4   default         41187 non-null  object 
 5   housing         41187 non-null  object 
 6   loan            41187 non-null  object 
 7   contact         41187 non-null  object 
 8   month           41187 non-null  object 
 9   day_of_week     41187 non-null  object 
 10  duration        41187 non-null  int64  
 11  campaign        41187 non-null  int64  
 12  pdays           41187 non-null  int64  
 13  previous        41187 non-null  int64  
 14  poutcome        41187 non-null  object 
 15  emp_var_rate    41187 non-null  float64
 16  cons_price_idx  41187 non-null  float64
 17  cons_conf_idx   41187 non-null  float64
 18  euribor3m       41187 non-null  float64
 19  nr_employed     41187 non-null  float64
 20  y               41187 non-null  object 
dtypes: float64(5), int64(5), object(11)
memory usage: 6.6+ MB
None
print(df.describe())
                age      duration  ...     euribor3m   nr_employed
count  41187.000000  41187.000000  ...  41187.000000  41187.000000
mean      40.023673    258.284944  ...      3.621261   5167.035329
std       10.421079    259.282396  ...      1.734458     72.252308
min       17.000000      0.000000  ...      0.634000   4963.600000
25%       32.000000    102.000000  ...      1.344000   5099.100000
50%       38.000000    180.000000  ...      4.857000   5191.000000
75%       47.000000    319.000000  ...      4.961000   5228.100000
max       98.000000   4918.000000  ...      5.045000   5228.100000

[8 rows x 10 columns]
print(df.nunique())
age                 78
job                 12
marital              4
education            8
default              3
housing              3
loan                 3
contact              2
month               10
day_of_week          5
duration          1544
campaign            42
pdays               27
previous             8
poutcome             3
emp_var_rate        10
cons_price_idx      26
cons_conf_idx       26
euribor3m          316
nr_employed         11
y                    2
dtype: int64
print(df.isnull())
         age    job  marital  ...  euribor3m  nr_employed      y
0      False  False    False  ...      False        False  False
1      False  False    False  ...      False        False  False
2      False  False    False  ...      False        False  False
3      False  False    False  ...      False        False  False
4      False  False    False  ...      False        False  False
...      ...    ...      ...  ...        ...          ...    ...
41182  False  False    False  ...      False        False  False
41183  False  False    False  ...      False        False  False
41184  False  False    False  ...      False        False  False
41185  False  False    False  ...      False        False  False
41186  False  False    False  ...      False        False  False

[41187 rows x 21 columns]
print(df['y'].value_counts(normalize=True) * 100)
y
no     88.734309
yes    11.265691
Name: proportion, dtype: float64
import matplotlib.pyplot as plt
Matplotlib is building the font cache; this may take a moment.
df['y'].value_counts().plot(kind = 'bar', color = ['skyblue', 'orange'])
<Axes: xlabel='y'>
pl
pl
Traceback (most recent call last):
  File "<pyshell#13>", line 1, in <module>
    pl
NameError: name 'pl' is not defined. Did you mean: 'pd'?
import matplotlib.pyplot as plt
df['y'].value_counts().plot(kind = 'bar', color = ['skyblue', 'orange'])
<Axes: xlabel='y'>
plt.title("Target Variable Distribution(susbscribed or Not)")
Text(0.5, 1.0, 'Target Variable Distribution(susbscribed or Not)')
plt.xlabel("subscription")
Text(0.5, 0, 'subscription')
plt.ylable("count")
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    plt.ylable("count")
AttributeError: module 'matplotlib.pyplot' has no attribute 'ylable'. Did you mean: 'table'?
import matplotlib.pyplot as plt
df['y'].value_counts().plot(kind = 'bar', color = ['skyblue', 'orange'])
<Axes: title={'center': 'Target Variable Distribution(susbscribed or Not)'}, xlabel='y'>
plt.title("Target Variable Distribution(susbscribed or Not)")
Text(0.5, 1.0, 'Target Variable Distribution(susbscribed or Not)')
plt.xlabel("subscription")
Text(0.5, 0, 'subscription')
plt.ylabel("Not")
Text(0, 0.5, 'Not')
plt.xticks(rotation = 0)
(array([0, 1]), [Text(0, 0, 'no'), Text(1, 0, 'yes')])
plt.show()
import matplotlib.pyplot as plt
df['job'].value_counts().plot(kind = 'bar', figsize = (10,4), color = ['green'])
<Axes: xlabel='job'>
plt.title("Job Distribution")
Text(0.5, 1.0, 'Job Distribution')
plt.xlabel("Job Type")
Text(0.5, 0, 'Job Type')
plt.ylabel("Count")
Text(0, 0.5, 'Count')
plt.xticks(rotation = 45)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), [Text(0, 0, 'admin.'), Text(1, 0, 'blue-collar'), Text(2, 0, 'technician'), Text(3, 0, 'services'), Text(4, 0, 'management'), Text(5, 0, 'retired'), Text(6, 0, 'entrepreneur'), Text(7, 0, 'self-employed'), Text(8, 0, 'housemaid'), Text(9, 0, 'unemployed'), Text(10, 0, 'student'), Text(11, 0, 'unknown')])
plt.tight_layout()
plt.show()
print(df['job'].value_counts())
job
admin.           10422
blue-collar       9254
technician        6743
services          3969
management        2924
retired           1720
entrepreneur      1456
self-employed     1421
housemaid         1059
unemployed        1014
student            875
unknown            330
Name: count, dtype: int64
print(df['marital'].value_counts())
marital
married     24927
single      11568
divorced     4612
unknown        80
Name: count, dtype: int64
print(df['education'].value_counts())
education
university.degree      12168
high.school             9515
basic.9y                6045
professional.course     5243
basic.4y                4175
basic.6y                2292
unknown                 1731
illiterate                18
Name: count, dtype: int64
import pandas as pd
df = pd.read_csv(r"C:\Users\Abigail Hefty\Downloads\cleaned_bank_data.csv")
print(df.head())
   duration    contact      source  previous  ... pdays     poutcome  loan default
0       487   cellular  additional         0  ...   999  nonexistent    no      no
1       346  telephone  additional         0  ...   999  nonexistent    no      no
2       227  telephone  additional         0  ...   999  nonexistent    no      no
3        17  telephone  additional         0  ...   999  nonexistent   NaN      no
4        58   cellular  additional         0  ...   999  nonexistent    no      no

[5 rows x 16 columns]
print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 53792 entries, 0 to 53791
Data columns (total 16 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   duration   53792 non-null  int64 
 1   contact    39507 non-null  object
 2   source     53792 non-null  object
 3   previous   53792 non-null  int64 
 4   housing    53687 non-null  object
 5   age        53792 non-null  int64 
 6   campaign   53792 non-null  int64 
 7   y          53792 non-null  object
 8   month      53792 non-null  object
 9   education  51640 non-null  object
 10  marital    53781 non-null  object
 11  job        53486 non-null  object
 12  pdays      53792 non-null  int64 
 13  poutcome   13187 non-null  object
 14  loan       53687 non-null  object
 15  default    52989 non-null  object
dtypes: int64(5), object(11)
memory usage: 6.6+ MB
None
print(df.isnull())
       duration  contact  source  previous  ...  pdays  poutcome   loan  default
0         False    False   False     False  ...  False     False  False    False
1         False    False   False     False  ...  False     False  False    False
2         False    False   False     False  ...  False     False  False    False
3         False    False   False     False  ...  False     False   True    False
4         False    False   False     False  ...  False     False  False    False
...         ...      ...     ...       ...  ...    ...       ...    ...      ...
53787     False    False   False     False  ...  False      True  False    False
53788     False     True   False     False  ...  False      True  False    False
53789     False    False   False     False  ...  False      True  False    False
53790     False    False   False     False  ...  False     False  False    False
53791     False    False   False     False  ...  False     False  False    False

[53792 rows x 16 columns]
print(df.nunique())
duration     1577
contact         2
source          3
previous       41
housing         2
age            77
campaign       48
y               2
month          12
education      10
marital         3
job            11
pdays         563
poutcome        4
loan            2
default         2
dtype: int64
print(df.describe())
           duration      previous           age      campaign         pdays
count  53792.000000  53792.000000  53792.000000  53792.000000  53792.000000
mean     258.607897      0.547925     40.884667      2.748717    110.670862
std      257.568129      2.175604     10.593944      3.062760    268.261432
min        0.000000      0.000000     18.000000      1.000000     -1.000000
25%      103.000000      0.000000     33.000000      1.000000     -1.000000
50%      181.000000      0.000000     39.000000      2.000000     -1.000000
75%      319.000000      0.000000     48.000000      3.000000     -1.000000
max     4918.000000    275.000000     95.000000     63.000000    999.000000
print(df['y'].value_counts(normalize=True) * 100)
y
no     88.362582
yes    11.637418
Name: proportion, dtype: float64
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Abigail Hefty\Downloads\cleaned_bank_data.csv")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
<Axes: >
plt.title("Correlation Heatmap of Numeric Features")
Text(0.5, 1.0, 'Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()
import squarify
job_counts = df['job'].value_counts()
plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
squarify.plot(sizes=job_counts.values, label=job_counts.index, alpha=0.8)
<Axes: >
plt.axis('off')
(np.float64(0.0), np.float64(100.0), np.float64(0.0), np.float64(100.0))
plt.title("Treemap of Client Jobs")
Text(0.5, 1.0, 'Treemap of Client Jobs')
plt.show()
import squarify
import matplotlib.pyplot as plt
job_counts = df['job'].value_counts()
labels = [f"{job}\n{count}" for job, count in zip(job_counts.index, job_counts.values)]
plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
squarify.plot(
    sizes=job_counts.values,
    label=labels,
    alpha=0.8,
    color=sns.color_palette("pastel"),
    pad=True
    )
<Axes: >
plt.axis('off')
(np.float64(0.0), np.float64(100.0), np.float64(0.0), np.float64(100.0))
plt.title("Treemap of Client Jobs with Counts")
Text(0.5, 1.0, 'Treemap of Client Jobs with Counts')
plt.show()
import squarify
import matplotlib.pyplot as plt
import seaborn as sns
edu_counts = df['education'].value_counts()
labels = [f"{edu}\n{count}" for edu, count in zip(edu_counts.index, edu_counts.values)]
plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
squarify.plot(
    sizes=edu_counts.values,
    label=labels,
    alpha=0.8,
    color=sns.color_palette("light:#5A9", n_colors=len(edu_counts)),
    pad=True
    )
<Axes: >
plt.axis('off')
(np.float64(0.0), np.float64(100.0), np.float64(0.0), np.float64(100.0))
plt.title("Treemap of Education Levels")
Text(0.5, 1.0, 'Treemap of Education Levels')
plt.show()
import matplotlib.pyplot as plt
y_counts = df['y'].value_counts()
labels = y_counts.index
sizes = y_counts.values
colors = ['#66b3ff', '#ff9999']
plt.figure(figsize=(6, 6))
<Figure size 600x600 with 0 Axes>
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
([<matplotlib.patches.Wedge object at 0x000001D69790C980>, <matplotlib.patches.Wedge object at 0x000001D697DC5810>], [Text(0.5341748448194182, -0.9615909916186562, 'no'), Text(-0.5341751581455055, 0.9615908175623478, 'yes')], [Text(0.29136809717422807, -0.5245041772465397, '88.4%'), Text(-0.2913682680793666, 0.5245040823067351, '11.6%')])
plt.axis('equal')
(np.float64(-1.09999999982052), np.float64(1.0999999967185574), np.float64(-1.0999997697980182), np.float64(1.0999997871539524))
plt.title("Subscription Outcome Distribution (y)")
Text(0.5, 1.0, 'Subscription Outcome Distribution (y)')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
sns.scatterplot(data=df, x='age', y='balance', hue='y', alpha=0.6)
Traceback (most recent call last):
  File "<pyshell#105>", line 1, in <module>
    sns.scatterplot(data=df, x='age', y='balance', hue='y', alpha=0.6)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\relational.py", line 615, in scatterplot
    p = _ScatterPlotter(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\relational.py", line 396, in __init__
    super().__init__(data=data, variables=variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_base.py", line 634, in __init__
    self.assign_variables(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_base.py", line 679, in assign_variables
    plot_data = PlotData(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_core\data.py", line 58, in __init__
    frame, names, ids = self._assign_variables(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_core\data.py", line 232, in _assign_variables
    raise ValueError(err)
ValueError: Could not interpret value `balance` for `y`. An entry with this name does not appear in `data`.
print(df.columns.tolist())
['duration', 'contact', 'source', 'previous', 'housing', 'age', 'campaign', 'y', 'month', 'education', 'marital', 'job', 'pdays', 'poutcome', 'loan', 'default']
df.columns = df.columns.str.strip()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
sns.scatterplot(data=df, x='age', y='balance', hue='y', alpha=0.6)
Traceback (most recent call last):
  File "<pyshell#111>", line 1, in <module>
    sns.scatterplot(data=df, x='age', y='balance', hue='y', alpha=0.6)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\relational.py", line 615, in scatterplot
    p = _ScatterPlotter(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\relational.py", line 396, in __init__
    super().__init__(data=data, variables=variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_base.py", line 634, in __init__
    self.assign_variables(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_base.py", line 679, in assign_variables
    plot_data = PlotData(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_core\data.py", line 58, in __init__
    frame, names, ids = self._assign_variables(data, variables)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\seaborn\_core\data.py", line 232, in _assign_variables
    raise ValueError(err)
ValueError: Could not interpret value `balance` for `y`. An entry with this name does not appear in `data`.
'balance' in df.columns
False
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
sns.scatterplot(data=df, x='age', y='duration', hue='y', alpha=0.6)
<Axes: xlabel='age', ylabel='duration'>
plt.title("Scatter Plot: Age vs Duration (colored by subscription outcome)")
Text(0.5, 1.0, 'Scatter Plot: Age vs Duration (colored by subscription outcome)')
plt.xlabel("Age")
Text(0.5, 0, 'Age')
plt.ylabel("Call Duration (seconds)")
Text(0, 0.5, 'Call Duration (seconds)')
plt.legend(title='Subscribed')
<matplotlib.legend.Legend object at 0x000001D698D42850>
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
sns.scatterplot(data=df, x='age', y='duration', hue='y', alpha=0.6)
<Axes: xlabel='age', ylabel='duration'>
0000
0
)
SyntaxError: unmatched ')'
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
sns.scatterplot(data=df, x='campaign', y='pdays', hue='y', alpha=0.6)
<Axes: xlabel='campaign', ylabel='pdays'>
plt.title("Scatter Plot: Campaign vs Pdays (colored by subscription outcome)")
Text(0.5, 1.0, 'Scatter Plot: Campaign vs Pdays (colored by subscription outcome)')
plt.xlabel("Number of Contacts (campaign)")
Text(0.5, 0, 'Number of Contacts (campaign)')
plt.ylabel("Days Since Last Contact (pdays)")
Text(0, 0.5, 'Days Since Last Contact (pdays)')
plt.legend(title='Subscribed')
<matplotlib.legend.Legend object at 0x000001D69F3CB4D0>
plt.show()
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
   df[col] = df[col].map({'yes': 1, 'no': 0})
   )
   
SyntaxError: unmatched ')'
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
df[col] = df[col].map({'yes': 1, 'no': 0})
SyntaxError: expected an indented block after 'for' statement on line 1
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
    ,
    
SyntaxError: invalid syntax
categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'source']

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
from sklearn.preprocessing import StandardScaler
Traceback (most recent call last):
  File "<pyshell#152>", line 1, in <module>
    from sklearn.preprocessing import StandardScaler
ModuleNotFoundError: No module named 'sklearn'
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
Traceback (most recent call last):
  File "<pyshell#153>", line 1, in <module>
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\reshape\encoding.py", line 170, in get_dummies
    data_to_encode = data[columns]
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\frame.py", line 4113, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\indexes\base.py", line 6212, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\indexes\base.py", line 6261, in _raise_if_missing
    raise KeyError(f"None of [{key}] are in the [{axis_name}]")
KeyError: "None of [Index(['job', 'marital', 'education', 'contact', 'month', 'poutcome',\n       'source'],\n      dtype='object')] are in the [columns]"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df.shape)
(53792, 47)
print(df.head())
   duration  previous housing  ...  poutcome_success  source_basic source_full
0  0.886733 -0.251852     yes  ...             False         False       False
1  0.339300 -0.251852      no  ...             False         False       False
2 -0.122718 -0.251852     yes  ...             False         False       False
3 -0.938044 -0.251852     NaN  ...             False         False       False
4 -0.778861 -0.251852     yes  ...             False         False       False

[5 rows x 47 columns]
binary_cols = ['default', 'housing', 'loan', 'y']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
   categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome', 'source']
   
SyntaxError: unindent does not match any outer indentation level
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#171>", line 1, in <module>
    model.fit(X_train, y_train)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\base.py", line 1363, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\linear_model\_logistic.py", line 1239, in fit
    X, y = validate_data(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 2971, in validate_data
    X, y = check_X_y(X, y, **check_params)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 1368, in check_X_y
    X = check_array(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 971, in check_array
    array = array.astype(new_dtype)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\generic.py", line 6662, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\internals\managers.py", line 430, in astype
    return self.apply(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\internals\managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\internals\blocks.py", line 784, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\dtypes\astype.py", line 237, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\dtypes\astype.py", line 182, in astype_array
    values = _astype_nansafe(values, dtype, copy=copy)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\pandas\core\dtypes\astype.py", line 133, in _astype_nansafe
    return arr.astype(dtype, copy=True)
ValueError: could not convert string to float: 'no'
df.select_dtypes(include='object').columns.tolist()
['housing', 'y', 'loan', 'default']
df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
df_encoded['y'] = df['y']
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
SyntaxError: multiple statements found while compiling a single statement
df.select_dtypes(include='object').columns.tolist()
['housing', 'y', 'loan', 'default']
df_encoded = pd.get_dummies(df.drop('y', axis=1), drop_first=True)
df_encoded['y'] = df['y']
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
LogisticRegression(max_iter=1000)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  
y_pred = model.predict(X_test)
  
print("Accuracy:", accuracy_score(y_test, y_pred))
  
Accuracy: 0.9015707779533414
print("\nReport:\n", classification_report(y_test, y_pred))
  

Report:
               precision    recall  f1-score   support

          no       0.92      0.98      0.95      9484
         yes       0.66      0.35      0.46      1275

    accuracy                           0.90     10759
   macro avg       0.79      0.66      0.70     10759
weighted avg       0.89      0.90      0.89     10759

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

Confusion Matrix:
 [[9249  235]
 [ 824  451]]
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
<Figure size 600x400 with 0 Axes>
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
<Axes: >
plt.xlabel('Predicted')
Text(0.5, 14.722222222222216, 'Predicted')
plt.ylabel('Actual')
Text(45.722222222222214, 0.5, 'Actual')
plt.title('Confusion Matrix')
Text(0.5, 1.0, 'Confusion Matrix')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
feature_names = X_train.columns
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(model.coef_[0])
    })
importance = importance.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
sns.barplot(x='Importance', y='Feature', data=importance.head(15), palette='viridis')

Warning (from warnings module):
  File "<pyshell#215>", line 1
FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

<Axes: xlabel='Importance', ylabel='Feature'>
plt.title('Top 15 Important Features (Logistic Regression)')
Text(0.5, 1.0, 'Top 15 Important Features (Logistic Regression)')
plt.tight_layout()
plt.show()
from sklearn.metrics import roc_curve, auc
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
Traceback (most recent call last):
  File "<pyshell#221>", line 1, in <module>
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_ranking.py", line 1163, in roc_curve
    fps, tps, thresholds = _binary_clf_curve(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_ranking.py", line 880, in _binary_clf_curve
    pos_label = _check_pos_label_consistency(pos_label, y_true)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 2651, in _check_pos_label_consistency
    raise ValueError(
ValueError: y_true takes value in {'no', 'yes'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_test_binary = y_test.map({'yes': 1, 'no': 0})
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_binary, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 5))
<Figure size 800x500 with 0 Axes>
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
[<matplotlib.lines.Line2D object at 0x000001D6A12C8050>]
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
[<matplotlib.lines.Line2D object at 0x000001D6A12C8190>]
plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
plt.title('ROC Curve')
Text(0.5, 1.0, 'ROC Curve')
plt.legend(loc='lower right')
<matplotlib.legend.Legend object at 0x000001D6A12C82D0>
plt.grid(True)
plt.show()
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
Traceback (most recent call last):
  File "<pyshell#238>", line 1, in <module>
    cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
NameError: name 'y_binary' is not defined
df['y'] = df['y'].map({'yes': 1, 'no': 0})
y_binary = df['y']
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
Cross-validation accuracy scores: [0.89562227 0.87861325 0.8793456  0.83928239 0.82942926]
print("Mean accuracy:", cv_scores.mean())
Mean accuracy: 0.8644585566922522
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid.fit(X, y_binary)
GridSearchCV(cv=5, estimator=LogisticRegression(max_iter=1000),
             param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]},
             scoring='accuracy')
print("Best parameters:", grid.best_params_)
Best parameters: {'C': 0.001}
print("Best cross-validation accuracy:", grid.best_score_)
Best cross-validation accuracy: 0.886451148323731
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
RandomForestClassifier(random_state=42)
rf_preds = rf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, rf_preds))
              precision    recall  f1-score   support

          no       0.93      0.97      0.95      9484
         yes       0.67      0.44      0.53      1275

    accuracy                           0.91     10759
   macro avg       0.80      0.70      0.74     10759
weighted avg       0.90      0.91      0.90     10759

import joblib
joblib.dump(model, 'final_model.pkl')  # or grid.best_estimator_
['final_model.pkl']
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
RandomForestClassifier(random_state=42)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap='Blues')
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001D6A0DE2CF0>
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
rf_probs = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
Traceback (most recent call last):
  File "<pyshell#271>", line 1, in <module>
    fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_ranking.py", line 1163, in roc_curve
    fps, tps, thresholds = _binary_clf_curve(
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\metrics\_ranking.py", line 880, in _binary_clf_curve
    pos_label = _check_pos_label_consistency(pos_label, y_true)
  File "C:\Users\Abigail Hefty\AppData\Roaming\Python\Python313\site-packages\sklearn\utils\validation.py", line 2651, in _check_pos_label_consistency
    raise ValueError(
ValueError: y_true takes value in {'no', 'yes'} and pos_label is not specified: either make y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.
y_test_binary = y_test.map({'yes': 1, 'no': 0})
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test_binary, rf_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
<Figure size 600x400 with 0 Axes>
plt.plot(fpr, tpr, color='green', label=f'Random Forest (AUC = {roc_auc:.2f})')
[<matplotlib.lines.Line2D object at 0x000001D6B808A850>]
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
[<matplotlib.lines.Line2D object at 0x000001D6B808A990>]
plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
plt.title('ROC Curve - Random Forest')
Text(0.5, 1.0, 'ROC Curve - Random Forest')
plt.legend()
<matplotlib.legend.Legend object at 0x000001D6B808AAD0>
plt.grid(True)
plt.tight_layout()
plt.show()
plt.show()
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test_binary, rf_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
<Figure size 600x400 with 0 Axes>
plt.plot(fpr, tpr, color='green', label=f'Random Forest (AUC = {roc_auc:.2f})')
[<matplotlib.lines.Line2D object at 0x000001D6B808A850>]
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
[<matplotlib.lines.Line2D object at 0x000001D6B808A990>]
plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
plt.title('ROC Curve - Random Forest')
Text(0.5, 1.0, 'ROC Curve - Random Forest')
plt.legend()
<matplotlib.legend.Legend object at 0x000001D6B808AAD0>
plt.grid(True)
plt.tight_layout()
plt.show()
SyntaxError: multiple statements found while compiling a single statement
from sklearn.metrics import roc_curve, auc
... import matplotlib.pyplot as plt
... fpr, tpr, thresholds = roc_curve(y_test_binary, rf_probs)
... roc_auc = auc(fpr, tpr)
... plt.figure(figsize=(6, 4))
... <Figure size 600x400 with 0 Axes>
... plt.plot(fpr, tpr, color='green', label=f'Random Forest (AUC = {roc_auc:.2f})')
... [<matplotlib.lines.Line2D object at 0x000001D6B808A850>]
... plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
... [<matplotlib.lines.Line2D object at 0x000001D6B808A990>]
... plt.xlabel('False Positive Rate')
... Text(0.5, 0, 'False Positive Rate')
... plt.ylabel('True Positive Rate')
... Text(0, 0.5, 'True Positive Rate')
... plt.title('ROC Curve - Random Forest')
... Text(0.5, 1.0, 'ROC Curve - Random Forest')
... plt.legend()
... <matplotlib.legend.Legend object at 0x000001D6B808AAD0>
... plt.grid(True)
... plt.tight_layout()
)
SyntaxError: invalid decimal literal
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test_binary, rf_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
<Figure size 600x400 with 0 Axes>
plt.plot(fpr, tpr, color='green', label=f'Random Forest (AUC = {roc_auc:.2f})')
[<matplotlib.lines.Line2D object at 0x000001D6B7C2A210>]
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
[<matplotlib.lines.Line2D object at 0x000001D6B7C2A350>]
plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
Traceback (most recent call last):
  File "<pyshell#300>", line 1, in <module>
    Text(0, 0.5, 'True Positive Rate')
NameError: name 'Text' is not defined. Did you mean: 'next'?
plt.title('ROC Curve - Random Forest')
Text(0.5, 1.0, 'ROC Curve - Random Forest')
Text(0.5, 1.0, 'ROC Curve - Random Forest')
Traceback (most recent call last):
  File "<pyshell#302>", line 1, in <module>
    Text(0.5, 1.0, 'ROC Curve - Random Forest')
NameError: name 'Text' is not defined. Did you mean: 'next'?
plt.legend()
<matplotlib.legend.Legend object at 0x000001D6B7C2A490>
plt.grid(True)
plt.tight_layout()
plt.show()
from sklearn.linear_model import LogisticRegression
logreg_balanced = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg_balanced.fit(X_train, y_train_binary)
Traceback (most recent call last):
  File "<pyshell#309>", line 1, in <module>
    logreg_balanced.fit(X_train, y_train_binary)
NameError: name 'y_train_binary' is not defined. Did you mean: 'y_test_binary'?
y_train_binary = y_train.map({'yes': 1, 'no': 0})
y_test_binary = y_test.map({'yes': 1, 'no': 0})
print(y_train_binary.value_counts())
y
0    38048
1     4985
Name: count, dtype: int64
print(y_test_binary.value_counts())
y
0    9484
1    1275
Name: count, dtype: int64
logreg_balanced.fit(X_train, y_train_binary)
LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train_binary)
RandomForestClassifier(class_weight='balanced', random_state=42)
from sklearn.metrics import ConfusionMatrixDisplay
y_pred_log = logreg_balanced.predict(X_test)
y_pred_rf = rf_balanced.predict(X_test)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test_binary, y_pred_log, ax=axes[0], cmap='Blues')
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001D6B80E0B90>
axes[0].set_title("Logistic Regression")
Text(0.5, 1.0, 'Logistic Regression')
ConfusionMatrixDisplay.from_predictions(y_test_binary, y_pred_rf, ax=axes[1], cmap='Greens')
<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay object at 0x000001D6B80E0B90>
axes[1].set_title("Random Forest")
Text(0.5, 1.0, 'Random Forest')
plt.tight_layout()
plt.show()
from sklearn.metrics import roc_curve, auc
logreg_probs = logreg_balanced.predict_proba(X_test)[:, 1]
rf_probs = rf_balanced.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test_binary, logreg_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test_binary, rf_probs)
auc_log = auc(fpr_log, tpr_log)
auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
plt.plot(fpr_log, tpr_log, label=f'LogReg (AUC = {auc_log:.2f})', linestyle='--')
[<matplotlib.lines.Line2D object at 0x000001D6B7CFD1D0>]
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
[<matplotlib.lines.Line2D object at 0x000001D6B7CFD310>]
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
[<matplotlib.lines.Line2D object at 0x000001D6B7CFD450>]
plt.xlabel('False Positive Rate')
Text(0.5, 0, 'False Positive Rate')
plt.ylabel('True Positive Rate')
Text(0, 0.5, 'True Positive Rate')
plt.title('ROC Curve Comparison')
Text(0.5, 1.0, 'ROC Curve Comparison')
plt.legend()
<matplotlib.legend.Legend object at 0x000001D6B7CFD590>
plt.grid(True)
plt.show()
from sklearn.metrics import classification_report
print("Logistic Regression Classification Report:")
Logistic Regression Classification Report:
print(classification_report(y_test_binary, y_pred_log))
              precision    recall  f1-score   support

           0       0.97      0.84      0.90      9484
           1       0.41      0.81      0.54      1275

    accuracy                           0.84     10759
   macro avg       0.69      0.82      0.72     10759
weighted avg       0.90      0.84      0.86     10759

print("Random Forest Classification Report:")
Random Forest Classification Report:
print(classification_report(y_test_binary, y_pred_rf))
              precision    recall  f1-score   support

           0       0.92      0.98      0.95      9484
           1       0.68      0.38      0.49      1275

    accuracy                           0.91     10759
   macro avg       0.80      0.68      0.72     10759
weighted avg       0.89      0.91      0.89     10759

import pandas as pd
importances = rf_balanced.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
    }).sort_values(by='Importance', ascending=False)
print(importance_df)
                          Feature    Importance
0                        duration  4.026484e-01
2                             age  1.179342e-01
3                        campaign  5.137029e-02
4                           pdays  4.173448e-02
40               poutcome_success  4.021109e-02
1                        previous  3.445899e-02
43                    housing_yes  2.841556e-02
34                      month_may  1.909067e-02
33                      month_mar  1.411459e-02
44                       loan_yes  1.336873e-02
36                      month_oct  1.324397e-02
27                      month_aug  1.249089e-02
15                marital_married  1.235607e-02
31                      month_jul  1.193604e-02
13                 job_technician  1.131715e-02
23            education_secondary  1.103965e-02
16                 marital_single  1.078210e-02
5                 job_blue-collar  1.075636e-02
32                      month_jun  1.049391e-02
24             education_tertiary  1.038200e-02
8                  job_management  1.011128e-02
35                      month_nov  9.650556e-03
26              contact_telephone  8.896181e-03
29                      month_feb  8.696759e-03
21              education_primary  7.219444e-03
11                   job_services  7.146869e-03
37                      month_sep  6.704487e-03
42                    source_full  6.431190e-03
30                      month_jan  5.554971e-03
9                     job_retired  5.512553e-03
12                    job_student  5.333941e-03
41                   source_basic  5.047103e-03
10              job_self-employed  4.815499e-03
14                 job_unemployed  4.385850e-03
6                job_entrepreneur  4.281351e-03
39                 poutcome_other  4.108252e-03
7                   job_housemaid  3.254610e-03
28                      month_dec  3.020904e-03
38           poutcome_nonexistent  3.000800e-03
25    education_university.degree  2.152087e-03
45                    default_yes  2.106746e-03
19          education_high.school  1.593223e-03
22  education_professional.course  1.331049e-03
18             education_basic.9y  9.920144e-04
17             education_basic.6y  5.062188e-04
20           education_illiterate  9.247259e-07
>>> import joblib
>>> joblib.dump(rf_balanced, 'random_forest_model.pkl')
['random_forest_model.pkl']
>>> loaded_model = joblib.load('random_forest_model.pkl')
>>> predictions = loaded_model.predict(X_test)
>>> joblib.dump(scaler, 'scaler.pkl')
['scaler.pkl']


