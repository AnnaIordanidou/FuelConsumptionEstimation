import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error




file_path = 'C:\\Users\\Άννα\\Desktop\\Thesis\\dataset\\data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Drop columns that contain only NaN values
df = df.dropna(axis=1, how='all')

#print(df.dtypes)

df = df.drop(columns=['DTC_NUMBER'])

# AFAIRESI MONADWN
# Convert to float
df['SPEED'] = df['SPEED'].str.replace('km/h', '').astype(float)
#print(df['SPEED'].head(2))


# Identify problematic rows
invalid_maf_rows = df[~df['MAF'].str.contains('g/s|,', na=False)]
#print("Invalid MAF values:", invalid_maf_rows['MAF'])

# Handle invalid values (replace them with NaN)
df['MAF'] = df['MAF'].apply(lambda x: x if 'g/s' in str(x) else None)

# Convert to float
df['MAF'] = df['MAF'].str.replace(',', '.').str.replace('g/s', '').astype(float)
#print(df['MAF'].head(2))

#Convert to float
invalid_barometric_pressure_rows = df[~df['BAROMETRIC_PRESSURE'].str.contains('kPa', na=False)]
df['BAROMETRIC_PRESSURE'] = df['BAROMETRIC_PRESSURE'].apply(lambda x: x if 'kPa' in str(x) else None)
df['BAROMETRIC_PRESSURE'] = df['BAROMETRIC_PRESSURE'].str.replace('kPa', '').astype(float)
#print(df['BAROMETRIC_PRESSURE'].head(2))

# # Convert to float
invalid_engine_coolant_temp_rows = df[~df['ENGINE_COOLANT_TEMP'].str.contains('C', na=False)]
df['ENGINE_COOLANT_TEMP'] = df['ENGINE_COOLANT_TEMP'].apply(lambda x: x if 'C' in str(x) else None)
df['ENGINE_COOLANT_TEMP'] = df['ENGINE_COOLANT_TEMP'].str.replace('C', '').astype(float)
#print(df['ENGINE_COOLANT_TEMP'].head())

# # Convert to float
invalid_air_intake_temp_rows = df[~df['AIR_INTAKE_TEMP'].str.contains('C', na=False)]
df['AIR_INTAKE_TEMP'] = df['AIR_INTAKE_TEMP'].apply(lambda x: x if 'C' in str(x) else None)
df['AIR_INTAKE_TEMP'] = df['AIR_INTAKE_TEMP'].str.replace('C', '').astype(float)
#print(df['AIR_INTAKE_TEMP'].head(2))

# # Convert to float
# invalid_ambient_air_temp_rows = df[~df['AMBIENT_AIR_TEMP'].str.contains('C', na=False)]
# df['AMBIENT_AIR_TEMP'] = df['AMBIENT_AIR_TEMP'].apply(lambda x: x if 'C' in str(x) else None)
df['AMBIENT_AIR_TEMP'] = df['AMBIENT_AIR_TEMP'].str.replace('C', '').astype(float)
#print(df['AMBIENT_AIR_TEMP'].head(2))

# # Convert to float
invalid_intake_manifold_pressure_rows = df[~df['INTAKE_MANIFOLD_PRESSURE'].str.contains('kPa', na=False)]
df['INTAKE_MANIFOLD_PRESSURE'] = df['INTAKE_MANIFOLD_PRESSURE'].apply(lambda x: x if 'kPa' in str(x) else None)
df['INTAKE_MANIFOLD_PRESSURE'] = df['INTAKE_MANIFOLD_PRESSURE'].str.replace('kPa', '').astype(float)
#print(df['INTAKE_MANIFOLD_PRESSURE'].head(2))

# Convert to float
df['THROTTLE_POS'] = pd.to_numeric(df['THROTTLE_POS'], errors='coerce')
#print(df['THROTTLE_POS'].head(2))

# Convert to float
df['FUEL_LEVEL'] = df['FUEL_LEVEL'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce')

# Convert to float
df['ENGINE_RPM'] = df['ENGINE_RPM'].astype(str).str.replace(r'[^\d.]+', '', regex=True).apply(pd.to_numeric, errors='coerce')

#print(df[['FUEL_LEVEL', 'ENGINE_RPM']].head(2))

# Convert to float
df['Short Term Fuel Trim Bank 1'] = df['Short Term Fuel Trim Bank 1'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce')
#print(df['Short Term Fuel Trim Bank 1'])

# Convert to float
df['TIMING_ADVANCE'] = df['TIMING_ADVANCE'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce')
#print(df['TIMING_ADVANCE'])

# Convert to float
df['EQUIV_RATIO'] = df['EQUIV_RATIO'].astype(str).str.replace('%', '').apply(pd.to_numeric, errors='coerce')
#print(df['EQUIV_RATIO'])

print(df.dtypes)


# Elegxw poses midanikes times uparxoun se kathe column
zero_count = (df['SPEED'] == 0).sum()
print(f"Number of zeros: {zero_count}")
print(len(df['SPEED']))

zero_count = (df['BAROMETRIC_PRESSURE'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['AMBIENT_AIR_TEMP'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['INTAKE_MANIFOLD_PRESSURE'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['ENGINE_COOLANT_TEMP'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['AIR_INTAKE_TEMP'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['MAF'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['ENGINE_RPM'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['THROTTLE_POS'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['FUEL_LEVEL'] == 0).sum()
# print(f"Number of zeros: {zero_count}")
# zero_count = (df['DTC_NUMBER'] == 0).sum()
print(f"Number of zeros: {zero_count}")
zero_count = (df['ENGINE_LOAD'] == 0).sum()
print(f"Number of zeros: {zero_count}")


# Kanw heatmap gia na dw to correlation etsi wste na epileksw pws tha gemisw kena
# Select only the numerical columns (float and int)
numerical_df = df.select_dtypes(include=['float64', 'int64'])

corr_matrix = numerical_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation Heatmap (Numerical Features Only)")
plt.show()


# Drop NaN values to plot existing data
# speed_no_na = df['SPEED'].dropna()
#
# # Plot histogram
# plt.figure(figsize=(8, 4))
# sns.histplot(speed_no_na, kde=True, bins=30)
# plt.title('Distribution of SPEED')
# plt.xlabel('Speed')
# plt.ylabel('Frequency')
# plt.show()
#
# # Plot boxplot
# plt.figure(figsize=(8, 4))
# sns.boxplot(x=speed_no_na)
# plt.title('Boxplot of SPEED')
# plt.show()



# Elegxw an to engine runtime einai mideniko otan speed=0 kai rpm>0
df["ENGINE_RUNTIME"] = df["ENGINE_RUNTIME"].astype(str)

idling_data = df[(df["SPEED"] == 0) & (df["ENGINE_RPM"] > 0)]

# Check if engine runtime is "00:00:00"
runtime_check = idling_data["ENGINE_RUNTIME"] == "00:00:00"

# Count how many times engine runtime is "00:00:00"
count_zero_runtime = runtime_check.sum()

#print(f"Number of cases where ENGINE_RUNTIME is '00:00:00': {count_zero_runtime}")


# Elegxw tis grammes pou speed kai rpm einai tautoxrona miden
filtered_rows = df[(df['SPEED'] == 0) & (df['ENGINE_RPM'] == 0)]

runtime = filtered_rows['ENGINE_RUNTIME'] == "00:00:00"

# Count how many times engine runtime is "00:00:00"
count_zero = runtime_check.sum()

#print(f"Number of cases where ENGINE_RUNTIME is '00:00:00': {count_zero}")



# Increase display settings for columns
pd.set_option('display.max_columns', None)
#print(filtered_rows)

# Diagrafw tis grammes pou einai tautoxrona miden
df = df[~((df['SPEED'] == 0) & (df['ENGINE_RPM'] == 0))]


# # elegxos an uparxoun tautoxrona midenika
removed_rows_check = df[(df['SPEED'] == 0) & (df['ENGINE_RPM'] == 0)]
#
# # apotelesma empty->tote den yparxoun tautoxrona midenika
print("Rows where speed and RPM are all zero:")
print(removed_rows_check)

df = df[df['ENGINE_RPM'] != 0]



# Count the rows where SPEED is 0 and ENGINE_RPM is less than 1000
rpm_normal = ((df['SPEED'] == 0) & (df['ENGINE_RPM'] >= 500) & (df['ENGINE_RPM'] <= 1000)).sum()

rpm_zero = ((df['SPEED'] == 0) & (df['ENGINE_RPM'] >= 0) & (df['ENGINE_RPM'] < 500)).sum()

rpm_under = ((df['SPEED'] == 0) & (df['ENGINE_RPM'] > 1300)).sum()

print("Number of rows where SPEED is 0 and ENGINE_RPM is between 500 and 1000:", rpm_normal)

print("Number of rows where SPEED is 0 and ENGINE_RPM is less than 1000:", rpm_zero)

print("Number of rows where SPEED is 0 and ENGINE_RPM is under 1300:", rpm_under)



# Posa missing values exw gia kathe column
missing_values = df.isnull().sum()
print(missing_values)

# Elegxw to pososto twn NaN timwn pou uparxoun
missing_percent = df.isna().mean() * 100
print(missing_percent)

# # Fill missing values me tin pio sixni timi
#df['DTC_NUMBER'] = df['DTC_NUMBER'].fillna(df['DTC_NUMBER'].mode()[0])



# Convert ENGINE_RUNTIME to string and strip spaces (to handle possible issues)
df['ENGINE_RUNTIME'] = df['ENGINE_RUNTIME'].astype(str).str.strip()

# Convert to total seconds (assuming HH:MM:SS format)
df['ENGINE_RUNTIME'] = pd.to_timedelta(df['ENGINE_RUNTIME'], errors='coerce').dt.total_seconds()

# Interpolate missing values within each VEHICLE_ID group
df['ENGINE_RUNTIME'] = df.groupby('VEHICLE_ID')['ENGINE_RUNTIME'].transform(lambda x: x.interpolate(method='linear'))

df['ENGINE_RUNTIME'] = df.groupby('VEHICLE_ID')['ENGINE_RUNTIME'].transform(lambda x: x.fillna(x.median()))

# Convert back to HH:MM:SS format properly
df['ENGINE_RUNTIME'] = df['ENGINE_RUNTIME'].apply(lambda x: str(pd.Timedelta(seconds=x)) if pd.notnull(x) else None)


#print(df['ENGINE_RUNTIME'].isnull().sum())






# ================ Method 1: Median  ================

# df_imputed = df.copy()
#
# # Fill missing values simfwna me vehicle id
# for col in df.select_dtypes(include='number').columns:
#     df_imputed[col] = df_imputed.groupby('VEHICLE_ID')[col].transform(lambda x: x.fillna(x.median()))
#
# # Now compare the original dataset with the imputed dataset
# variables_to_impute = df.select_dtypes(include='number').columns  # Numeric columns to check
#
# # Create a DataFrame showing the differences between the original and imputed values
# df_compare = df[variables_to_impute] - df_imputed[variables_to_impute]
#
# # Check the descriptive statistics of the differences to see if changes exist
# print(df_compare.describe())  # This will show the mean, std, min, and max differences
#
# # Posa missing values exw gia kathe column
# missing_values = df_imputed.isnull().sum()
# print(missing_values)




def variables_plot(df, variab):
    for variable in variab:
        plt.figure(figsize=(12, 6))
        for vehicle_id, vehicle_data in df.groupby('VEHICLE_ID'):
            if vehicle_data[variable].var() == 0:  # Check for zero variance
                plt.axvline(
                    x=vehicle_data[variable].iloc[0],
                    color='red',
                    linestyle='--',
                    label=f'VEHICLE_ID {vehicle_id} (constant)'
                )
            else:
                sns.kdeplot(
                    data=vehicle_data,
                    x=variable,
                    fill=True,
                    label=f'VEHICLE_ID {vehicle_id}',
                    warn_singular=False
                )

        plt.title(f'Distribution of {variable} by Vehicle ID')
        plt.xlabel(variable)
        plt.ylabel('Density')
        plt.legend(title='Vehicle ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()




#variables = ['FUEL_LEVEL', 'ENGINE_LOAD', 'ENGINE_RPM', 'MAF', 'SPEED','BAROMETRIC_PRESSURE', 'AMBIENT_AIR_TEMP', 'INTAKE_MANIFOLD_PRESSURE', 'AIR_INTAKE_TEMP', 'THROTTLE_POS', 'ENGINE_COOLANT_TEMP']


# variables = ['FUEL_LEVEL', 'ENGINE_LOAD']
# variables_plot(df, variables)
# variables_plot(df_imputed, variables)



def impute_selected_columns(group, imputer, variables_to_impute):
    # Select only the columns to impute
    to_impute = group[variables_to_impute]

    # Perform iterative imputation
    imputed = imputer.fit_transform(to_impute)

    # Update the group with imputed values
    group[variables_to_impute] = imputed
    return group



# ================ Method 2: Median & MICE ================
# Weak Variables
weak_variables = ['BAROMETRIC_PRESSURE', 'AMBIENT_AIR_TEMP', 'INTAKE_MANIFOLD_PRESSURE', 'AIR_INTAKE_TEMP', 'THROTTLE_POS', 'ENGINE_COOLANT_TEMP',
                  'TIMING_ADVANCE', 'EQUIV_RATIO', 'Short Term Fuel Trim Bank 1']

# Fill missing values using the median within each driver group
for col in weak_variables:

    df[col] = df.groupby('VEHICLE_ID')[col].transform(lambda x: x.fillna(x.median()))

# To verify the result:
print(df[weak_variables].isnull().sum())


# List of variables for imputation
variables = ['FUEL_LEVEL', 'ENGINE_LOAD', 'ENGINE_RPM', 'MAF', 'SPEED']

# Create the Iterative Imputer
iter_imputer = IterativeImputer(random_state=42, max_iter=20)
# Group by VEHICLE_ID and apply the imputation
df_imputed = df.groupby('VEHICLE_ID', group_keys=False).apply(impute_selected_columns, imputer = iter_imputer, variables_to_impute = variables)



print(df_imputed[variables].isnull().sum())
print(df_imputed[variables].head())


missing_values = df_imputed.isnull().sum()
print(missing_values)


df_compare = df[variables] - df_imputed[variables]
print(df_compare.describe())


# variables_plot(df, variables)
# variables_plot(df_imputed, variables)


# ================ Method 3: KNN ================

# knn_imputer = KNNImputer(n_neighbors=5)
# variables = df.select_dtypes(include=['number']).columns.tolist()
#
# df_imputed = df.groupby('VEHICLE_ID', group_keys=False).apply(
#     impute_selected_columns, imputer=knn_imputer, variables_to_impute=variables
# )
#
# print(df_imputed[variables].isnull().sum())
# print(df_imputed[variables].head())
#
# missing_values = df_imputed.isnull().sum()
# print(missing_values)
#
# df_compare = df[variables] - df_imputed[variables]
# print(df_compare.describe())  # Check if differences exist
#
# variables_plot(df, variables)
# variables_plot(df_imputed, variables)




# ================ New Values ================

# Acceleration
# Convert speed from km/h to m/s
speed_mps = df_imputed['SPEED'] * (1000 / 3600)

# Convert time from milliseconds to seconds if needed
time_s = df_imputed['TIME'] / 1000

# Calculate acceleration (Δspeed / Δtime)
df_imputed['Acceleration'] = speed_mps.diff() / time_s.diff()
df_imputed['Acceleration'] = df_imputed['Acceleration'].bfill()


# Fuel Consumption using speed and MAF
a = 7.718
b = 235.215

df_imputed['MAF'] = df_imputed['MAF'].replace(0, np.nan)  # Avoid division by zero
df_imputed['FL_MAF'] = (df_imputed['SPEED'] * a * b) / df_imputed['MAF']
df_imputed['FL_MAF'] = df_imputed['FL_MAF'].bfill()
df_imputed['MAF'] = df_imputed['MAF'].fillna(df_imputed['MAF'].median())



# Fuel Consumption using RPM and Throttle Position
p_1 = 2.685
p_2 = -0.1246
p_3 = 1.243
df_imputed['FL_RPM'] = p_1*pow(df_imputed['ENGINE_RPM'],2) + p_2*df_imputed['THROTTLE_POS'] + p_3*df_imputed['ENGINE_RPM']*df_imputed['THROTTLE_POS']
df_imputed['FL_RPM'] = df_imputed['FL_RPM'].bfill()

print(df_imputed.isnull().sum())
print(f"Statistics summary:", df_imputed.describe())



# print((df_imputed['ENGINE_RPM']==0).sum())
# print((df_imputed['ENGINE_LOAD']==0).sum())
# print((df_imputed['EQUIV_RATIO']==0).sum())
# print((df_imputed['SPEED']==0).sum())


correlation_matrix = df_imputed.corr(method='pearson', numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='hot', fmt='.2f', vmin=-1, vmax=1)




# ==================== Feature Importance ====================

X = df_imputed[['TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'BAROMETRIC_PRESSURE',
'ENGINE_COOLANT_TEMP', 'FUEL_LEVEL', 'ENGINE_LOAD', 'AMBIENT_AIR_TEMP',
'ENGINE_RPM', 'INTAKE_MANIFOLD_PRESSURE', 'MAF', 'AIR_INTAKE_TEMP', 'SPEED', 'Short Term Fuel Trim Bank 1',
'THROTTLE_POS', 'TIMING_ADVANCE', 'EQUIV_RATIO', 'Acceleration']]

y_rpm = df_imputed['FL_RPM']
y_maf = df_imputed['FL_MAF']

X_train_maf, X_test_maf, y_train_maf, y_test_maf = train_test_split(X, y_maf, test_size=0.2, random_state=42)
X_train_rpm, X_test_rpm, y_train_rpm, y_test_rpm = train_test_split(X, y_rpm, test_size=0.2, random_state=42)

fr_maf = RandomForestRegressor(n_estimators=100,random_state=42)
fr_rpm = RandomForestRegressor(n_estimators=100,random_state=42)

fr_maf.fit(X_train_maf, y_train_maf)
fr_rpm.fit(X_train_rpm, y_train_rpm)

importance_maf = fr_maf.feature_importances_
importance_rpm = fr_rpm.feature_importances_


for feature, importance in zip(X.columns, importance_maf):
    print(f"{feature}: {importance:.6f}")

for feature, importance in zip(X.columns, importance_rpm):
    print (f"{feature}: {importance:.6f}")


features = X.columns
x = np.arange(len(features))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, importance_maf, width, label='FL_MAF', color='skyblue')
bars2 = ax.bar(x + width/2, importance_rpm, width, label='FL_RPM', color='salmon')


ax.set_ylabel('Feature Importance')
ax.set_title('Feature Importance for FL_MAF vs FL_RPM')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=90)
ax.legend()

plt.tight_layout()
plt.show()


# ==================== Statistics and Plots ====================

boxplot_stats_dictionary = {}
for var in ['ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF', 'Acceleration', 'FL_MAF', 'FL_RPM']:
    summary_statistics = df_imputed.groupby('VEHICLE_ID', observed=False)[var].describe()

    # Select and rename the relevant statistics
    boxplot_tab = summary_statistics[['min', '25%', '50%', '75%', 'max']].rename(columns={
        'min': 'Minimum',
        '25%': 'Q1 (25%)',
        '50%': 'Median (50%)',
        '75%': 'Q3 (75%)',
        'max': 'Maximum'
    })
    boxplot_tab['IQR'] = boxplot_tab['Q3 (75%)'] - boxplot_tab['Q1 (25%)']
    boxplot_tab = boxplot_tab.sort_index(key=lambda x: x.str.extract(r's(\d+)').astype(int)[0])
    boxplot_tab = boxplot_tab.round(2)

    boxplot_stats_dictionary[var] = boxplot_tab
    print(f"Boxplot Stats for {var}:", boxplot_stats_dictionary[var])




df_imputed['VEHICLE_ID'] = df_imputed['VEHICLE_ID'].astype('category')
sorted_categories = sorted(df_imputed['VEHICLE_ID'].unique(), key=lambda x: int(x[1:]))
df_imputed['VEHICLE_ID'] = df_imputed['VEHICLE_ID'].cat.set_categories(sorted_categories)


# Boxplots for speed, enine load, engine rpm, maf, acceleration, fl_maf, fl_rpm
var_box = df_imputed[['SPEED', 'ENGINE_LOAD', 'ENGINE_RPM', 'MAF', 'Acceleration', 'FL_MAF', 'FL_RPM']]

# for col in var_box:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='VEHICLE_ID', y=col, data=df_imputed, order=sorted_categories)
#         plt.title(f'Boxplot of {col} for all drivers ')
#         plt.xlabel('VEHICLE_ID')
#         plt.ylabel(col)
#         plt.savefig(f'C:\\Users\\Άννα\\Desktop\\thesis_photos1\\boxplot_{col}_all_drivers.png')




# Scatter plots for speed, maf, acceleration, fl_maf, fl_rpm
# g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
# g.map(sns.scatterplot, "SPEED", "FL_MAF", alpha=0.6)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Fuel Consumption (MAF) vs Speed for Each Driver", fontsize=16)
# plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\facet_fuel_consumption.png')
# plt.show()


summary_stats = df_imputed.groupby("VEHICLE_ID")[["SPEED", "FL_MAF"]].describe()
summary_stats = summary_stats.round(1)

print(f"Summary Stats speed vs fl_maf:", summary_stats)





# g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
# g.map(sns.scatterplot, "SPEED", "FL_RPM", alpha=0.6)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Fuel Consumption (RPM) vs Speed for Each Driver", fontsize=16)
# plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\facet_fuel_consumption_rpm.png')
# plt.show()


summary_stats = df_imputed.groupby("VEHICLE_ID")[["SPEED", "FL_RPM"]].describe()
summary_stats = summary_stats.round(1)
print(f"Summary Stats speed vs fl_rpm:", summary_stats)





# g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
# g.map(sns.scatterplot, "SPEED", "Acceleration", alpha=0.6)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Speed vs Acceleration for Each Driver", fontsize=16)
# plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\facet_speed.png')
# plt.show()



summary_stats = df_imputed.groupby("VEHICLE_ID")[["SPEED", "Acceleration"]].describe()
summary_stats = summary_stats.round(1)

print(f"Summary Stats speed vs acceleration:",summary_stats)




# g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
# g.map(sns.scatterplot, "Acceleration", "FL_MAF", alpha=0.6)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Acceleration vs Fuel Consumption (MAF) for Each Driver", fontsize=16)
# plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\facet_acceleration_maf.png')
# plt.show()


# g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
# g.map(sns.scatterplot, "Acceleration", "FL_RPM", alpha=0.6)
# plt.subplots_adjust(top=0.9)
# g.fig.suptitle("Acceleration vs Fuel Consumption (RPM) for Each Driver", fontsize=16)
# plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\facet_acceleration_rpm.png')
# plt.show()


g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
g.map(sns.scatterplot, "ENGINE_LOAD", "FL_MAF", alpha=0.6)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Engine Load vs Fuel Consumption (MAF) for Each Driver", fontsize=16)
plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\engine_load_maf.png')
plt.show()


g = sns.FacetGrid(df_imputed, col="VEHICLE_ID", col_wrap=4, height=4, aspect=1.5)
g.map(sns.scatterplot, "ENGINE_RPM", "FL_RPM", alpha=0.6)
plt.subplots_adjust(top=0.9)
g.fig.suptitle("RPM vs Fuel Consumption (RPM) for Each Driver", fontsize=16)
plt.savefig('C:\\Users\\Άννα\\Desktop\\thesis_photos1\\engine_rpm_maf.png')
plt.show()




# ==================== Clustering ====================

behavior_features = df_imputed[['ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF',
                                'Acceleration', 'THROTTLE_POS', 'TIMING_ADVANCE']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(behavior_features)



# Original statistics
original_stats = behavior_features.describe().T[['mean', 'std', 'min', 'max']]
original_stats.columns = ['Mean (Original)', 'Std (Original)', 'Min (Original)', 'Max (Original)']

# Scaled statistics
scaled_df = pd.DataFrame(X_scaled, columns=behavior_features.columns)
scaled_stats = scaled_df.describe().T[['mean', 'std', 'min', 'max']]
scaled_stats.columns = ['Mean (Scaled)', 'Std (Scaled)', 'Min (Scaled)', 'Max (Scaled)']

# Combine into one table
scaling_comparison = pd.concat([original_stats, scaled_stats], axis=1)
print(scaling_comparison.round(2))




inertia = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2, 6), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for KMeans')
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df_imputed['KMeans_Cluster'] = kmeans_labels

cluster_colors = {
    0: 'cornflowerblue',
    1: 'mediumseagreen',
    2: 'tomato'
}

dbscan = DBSCAN(eps=1.5, min_samples=50)
dbscan_labels = dbscan.fit_predict(X_scaled)
df_imputed['DBSCAN_Cluster'] = dbscan_labels

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.subplot(1, 2, 1)
for label in np.unique(kmeans_labels):
    plt.scatter(
        X_pca[kmeans_labels == label, 0],
        X_pca[kmeans_labels == label, 1],
        color=cluster_colors[label],
        s=10,
        label=f'Cluster {label}'
    )
plt.title("KMeans Clusters")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='Set2', s=10)
plt.title("DBSCAN Clusters")

plt.tight_layout()
plt.show()




cluster_names = {
    0: 'Slow/Eco',
    1: 'Normal',
    2: 'Aggressive'
}


cluster_colors = {
    'Slow/Eco': 'cornflowerblue',
    'Normal': 'mediumseagreen',
    'Aggressive': 'tomato'
}

driver_clusters = df_imputed.groupby('VEHICLE_ID')['KMeans_Cluster'].value_counts(normalize=True).unstack()
print(driver_clusters)


cluster_means = df_imputed.groupby('KMeans_Cluster')[behavior_features.columns].mean()
print(cluster_means)


fuel_by_cluster = df_imputed.groupby('KMeans_Cluster')[['FL_MAF', 'FL_RPM']].mean()
print(fuel_by_cluster)


driver_clusters_named = driver_clusters.rename(columns=cluster_names)

driver_clusters_named = driver_clusters_named[['Slow/Eco', 'Normal', 'Aggressive']]

driver_clusters_named.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    color=[cluster_colors[col] for col in driver_clusters_named.columns]
)

plt.ylabel("Proportion of Driving Segments")
plt.title("Driver Behavior Distribution by Cluster")
plt.legend(title="Cluster")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# driver_clusters.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set1')
# plt.ylabel("Proportion of Driving Segments")
# plt.title("Driver Behavior Distribution by Cluster")
# plt.legend(title="Cluster")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()





df_imputed['Driving_Behavior'] = df_imputed['KMeans_Cluster'].map(cluster_names)

boxplot_stats_dict = {}

for var in ['ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF', 'Acceleration', 'THROTTLE_POS', 'TIMING_ADVANCE']:

    sns.boxplot(data=df_imputed, x='Driving_Behavior', y=var)
    plt.title(f"{var} per Driving Behavior")
    plt.xlabel('Driving Behavior')
    plt.ylabel(var)
    plt.show()

    summary_stats = df_imputed.groupby('Driving_Behavior')[var].describe()

    # Select and rename the relevant statistics
    boxplot_table = summary_stats[['min', '25%', '50%', '75%', 'max']].rename(columns={
        'min': 'Minimum',
        '25%': 'Q1 (25%)',
        '50%': 'Median (50%)',
        '75%': 'Q3 (75%)',
        'max': 'Maximum'
    })
    boxplot_table['IQR'] = boxplot_table['Q3 (75%)'] - boxplot_table['Q1 (25%)']

    # Store in dictionary
    boxplot_stats_dict[var] = boxplot_table

print("Boxplot Stats for SPEED:")
print(boxplot_stats_dict['SPEED'])


print("Boxplot Stats for Acceleration:")
print(boxplot_stats_dict['Acceleration'])


print("Boxplot Stats for ENGINE RPM:")
print(boxplot_stats_dict['ENGINE_RPM'])


print("Boxplot Stats for MAF:")
print(boxplot_stats_dict['MAF'])








# dimiourgia profil odigwn, me pososto. px 27% aggressive klp
driver_behavior_pct = (
    df_imputed.groupby(['VEHICLE_ID', 'Driving_Behavior'], observed=False)
    .size()
    .groupby(level=0)
    .apply(lambda x: x / x.sum())
    .unstack()
    .fillna(0)
    .round(2)
)

fuel_per_behavior = (
    df_imputed.groupby(['VEHICLE_ID', 'Driving_Behavior'], observed=False)[['FL_MAF', 'FL_RPM']]
    .mean()
    .round(2)
    .unstack()
)

fuel_per_behavior.columns = ['_'.join(col).strip() for col in fuel_per_behavior.columns.values]
fuel_per_behavior.head()
fuel_per_behavior.index = fuel_per_behavior.index.get_level_values(0)
driver_behavior_pct.index = driver_behavior_pct.index.get_level_values(0)


# weighted fuel cons
weighted_fl_maf = (
    driver_behavior_pct['Aggressive'] * fuel_per_behavior['FL_MAF_Aggressive'] +
    driver_behavior_pct['Normal'] * fuel_per_behavior['FL_MAF_Normal'] +
    driver_behavior_pct['Slow/Eco'] * fuel_per_behavior['FL_MAF_Slow/Eco']
)

weighted_fl_rpm = (
    driver_behavior_pct['Aggressive'] * fuel_per_behavior['FL_RPM_Aggressive'] +
    driver_behavior_pct['Normal'] * fuel_per_behavior['FL_RPM_Normal'] +
    driver_behavior_pct['Slow/Eco'] * fuel_per_behavior['FL_RPM_Slow/Eco']
)

weighted_fuel = pd.DataFrame({
    'FL_MAF_weighted': weighted_fl_maf.round(2),
    'FL_RPM_weighted': weighted_fl_rpm.round(2)
})




driver_profiles = pd.concat([driver_behavior_pct*100, weighted_fuel], axis=1)
print(driver_profiles)


# driving behavior kai fuel consumption
sns.boxplot(data=df_imputed, x='Driving_Behavior', y='FL_MAF')
plt.title("FL_MAF per Driving Behavior")
plt.show()

sns.boxplot(data=df_imputed, x='Driving_Behavior', y='FL_RPM')
plt.title("FL_RPM per Driving Behavior")
plt.show()


maf_stats = df_imputed.groupby('Driving_Behavior')['FL_MAF'].describe()
boxplot_maf = maf_stats[['min', '25%', '50%', '75%', 'max']].rename(columns={
    'min': 'MAF_Min',
    '25%': 'MAF_Q1',
    '50%': 'MAF_Median',
    '75%': 'MAF_Q3',
    'max': 'MAF_Max'
})
boxplot_maf['MAF_IQR'] = boxplot_maf['MAF_Q3'] - boxplot_maf['MAF_Q1']

rpm_stats = df_imputed.groupby('Driving_Behavior')['FL_RPM'].describe()
boxplot_rpm = rpm_stats[['min', '25%', '50%', '75%', 'max']].rename(columns={
    'min': 'RPM_Min',
    '25%': 'RPM_Q1',
    '50%': 'RPM_Median',
    '75%': 'RPM_Q3',
    'max': 'RPM_Max'
})
boxplot_rpm['RPM_IQR'] = boxplot_rpm['RPM_Q3'] - boxplot_rpm['RPM_Q1']

combined_boxplot_stats = pd.concat([boxplot_maf, boxplot_rpm], axis=1)

print("Combined Boxplot Statistics per Driving Behavior:")
print(combined_boxplot_stats)





driver_profiles['Dominant_Behavior'] = driver_profiles[['Aggressive', 'Normal', 'Slow/Eco']].idxmax(axis=1)

sns.boxplot(data=driver_profiles, x='Dominant_Behavior', y='FL_MAF_weighted', palette='pastel')
plt.title("FL_MAF_weighted by Dominant Driver Behavior")
plt.ylabel("Weighted FL_MAF")
plt.xlabel("Dominant Behavior")
plt.tight_layout()
plt.show()

summary_stats_MAF = driver_profiles.groupby('Dominant_Behavior')['FL_MAF_weighted'].describe()

boxplot_table_MAF = summary_stats_MAF[['min', '25%', '50%', '75%', 'max']].rename(columns={
    'min': 'Minimum',
    '25%': 'Q1 (25%)',
    '50%': 'Median (50%)',
    '75%': 'Q3 (75%)',
    'max': 'Maximum'
})

boxplot_table_MAF['IQR'] = boxplot_table_MAF['Q3 (75%)'] - boxplot_table_MAF['Q1 (25%)']




# Boxplot for FL_RPM_weighted
sns.boxplot(data=driver_profiles, x='Dominant_Behavior', y='FL_RPM_weighted', palette='Set2')
plt.title("FL_RPM_weighted by Dominant Driver Behavior")
plt.ylabel("Weighted FL_RPM")
plt.xlabel("Dominant Behavior")
plt.tight_layout()
plt.show()


summary_stats_RPM = driver_profiles.groupby('Dominant_Behavior')['FL_RPM_weighted'].describe()

boxplot_table_RPM = summary_stats_RPM[['min', '25%', '50%', '75%', 'max']].rename(columns={
    '25%': 'Q1 (25%)',
    '50%': 'Median (50%)',
    '75%': 'Q3 (75%)',
    'min': 'Minimum',
    'max': 'Maximum'
})

boxplot_table_RPM['IQR'] = boxplot_table_RPM['Q3 (75%)'] - boxplot_table_RPM['Q1 (25%)']

combined_wboxplot_stats = pd.concat([boxplot_table_MAF, boxplot_table_RPM], axis=1)

# Display the table
print("Boxplot Statistics Driving Behavior & Weighted Fuel Cosnumption:")
print(combined_wboxplot_stats)




grouped_maf = df_imputed.groupby('Driving_Behavior')['FL_MAF'].mean()
grouped_rpm = df_imputed.groupby('Driving_Behavior')['FL_RPM'].mean()

labels = grouped_maf.index.tolist()
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 0.2, grouped_maf.values, width=0.4, label='FL_MAF', color='skyblue')
ax.bar(x + 0.2, grouped_rpm.values, width=0.4, label='FL_RPM', color='salmon')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Average Fuel Consumption')
ax.set_xlabel('Driving Behavior')
ax.set_title('Average Fuel Consumption by Driving Behavior')
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()

beh_acc = df_imputed.groupby(['VEHICLE_ID', 'Driving_Behavior'])[['Acceleration']].mean()




table = df_imputed.groupby(['VEHICLE_ID', 'Driving_Behavior'])[['FL_MAF', 'FL_RPM']].mean()
print(table)


# Correlation metaksi twn behavior kai fuel consumption
correlation_table = pd.DataFrame()

for behavior in ['Aggressive', 'Normal', 'Slow/Eco']:
    corr_with_fl_maf = driver_profiles[[behavior, 'FL_MAF_weighted']].corr().iloc[0, 1]
    corr_with_fl_rpm = driver_profiles[[behavior, 'FL_RPM_weighted']].corr().iloc[0, 1]

    correlation_table[behavior] = [corr_with_fl_maf, corr_with_fl_rpm]

correlation_table.index = ['Corr with FL_MAF_weighted', 'Corr with FL_RPM_weighted']

print(correlation_table.round(3))


# ============== prediction model =================

# ===== Prwti periptwsi =====
targets = ['FL_MAF', 'FL_RPM']

X = df_imputed[['FUEL_LEVEL', 'ENGINE_LOAD', 'ENGINE_RPM', 'MAF', 'SPEED',
                'BAROMETRIC_PRESSURE', 'AMBIENT_AIR_TEMP', 'INTAKE_MANIFOLD_PRESSURE',
                'AIR_INTAKE_TEMP', 'THROTTLE_POS', 'ENGINE_COOLANT_TEMP', 'Acceleration']]


results = []

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [10, 100],
    'epsilon': [0.1, 0.2],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

for target in targets:
    y = df_imputed[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    grid = GridSearchCV(SVR(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_svr = grid.best_estimator_
    print(f"\nBest SVR parameters for target {target}: {grid.best_params_}")

    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(max_depth=5, random_state=100),
        'Support Vector Regression': best_svr
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='green', alpha=0.6, label='Actual')
        plt.scatter(range(len(y_test_pred)), y_test_pred, color='orange', alpha=0.6, label='Predicted')
        plt.title(f'Actual vs Predicted {target} - {model_name}')
        plt.xlabel('Sample Index')
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.show()

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)


        results.append({
            'Target': target,
            'Model': model_name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train R2': train_r2,
            'Test R2': test_r2
        })





results_df = pd.DataFrame(results)
print("\n📊 Final Model Comparison:")
print(results_df)


sns.set(style="whitegrid")
#
# for target in targets:
#     plt.figure(figsize=(10, 6))
#     temp_df = results_df[results_df['Target'] == target]
#     sns.barplot(x='Model', y='Test RMSE', data=temp_df, palette='pastel')
#     plt.title(f'Model Comparison for {target} (Test RMSE)', fontsize=16)
#     plt.ylabel('Test RMSE')
#     plt.xlabel('Model')
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.show()
#
#
# for target in targets:
#     plt.figure(figsize=(10, 6))
#     temp_df = results_df[results_df['Target'] == target]
#     sns.barplot(x='Model', y='Train RMSE', data=temp_df, palette='pastel')
#     plt.title(f'Model Comparison for {target} (Train RMSE)', fontsize=16)
#     plt.ylabel('Train RMSE')
#     plt.xlabel('Model')
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.show()
#



import shap

# Random Forest ksana afou exei ta kalitera apotelesmata
rf_models = {}
X_trains = {}

for target in targets:
    y = df_imputed[target]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


    rf_model = RandomForestRegressor(max_depth=5, random_state=100)
    rf_model.fit(X_train, y_train)

    rf_models[target] = rf_model
    X_trains[target] = X_train

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_train)

    print(f"SHAP Summary Plot for {target}:")
    shap.summary_plot(shap_values, X_train, plot_type="dot")


# ===== Deuteri periptwsi =====


targets = ['FL_MAF_weighted', 'FL_RPM_weighted']
X = driver_profiles[['Aggressive', 'Normal', 'Slow/Eco']]

results = []

param_grid = {
    'C': [10, 100],
    'epsilon': [0.1, 0.2],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

for target in targets:
    y = driver_profiles[target]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    grid = GridSearchCV(SVR(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_svr = grid.best_estimator_
    print(f"\nBest SVR parameters for target {target}: {grid.best_params_}")


    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Support Vector Regression': best_svr
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='green', alpha=0.6, label='Actual')
        plt.scatter(range(len(y_test_pred)), y_test_pred, color='orange', alpha=0.6, label='Predicted')
        plt.title(f'Actual vs Predicted {target} - {model_name}')
        plt.xlabel('Sample Index')
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        plt.show()

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results.append({
            'Target': target,
            'Model': model_name,
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'Train R2': train_r2,
            'Test R2': test_r2
        })

results_df = pd.DataFrame(results)

print("\n📊 Final Model Comparison (Per Driver Cluster Behavior):")
print(results_df.round(3))

sns.set(style="whitegrid")

# for target in targets:
#     plt.figure(figsize=(10, 6))
#     temp_df = results_df[results_df['Target'] == target]
#     sns.barplot(x='Model', y='Test RMSE', data=temp_df, palette='pastel')
#     plt.title(f'Model Comparison for {target} (Test RMSE)', fontsize=16)
#     plt.ylabel('Test RMSE')
#     plt.xlabel('Model')
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.show()
#
# for target in targets:
#     plt.figure(figsize=(10, 6))
#     temp_df = results_df[results_df['Target'] == target]
#     sns.barplot(x='Model', y='Train RMSE', data=temp_df, palette='pastel')
#     plt.title(f'Model Comparison for {target} (Train RMSE)', fontsize=16)
#     plt.ylabel('Train RMSE')
#     plt.xlabel('Model')
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.show()






# # Save the updated dataset
# output_file_path = 'C:\\Users\\Άννα\\Desktop\\Thesis\\dataset\\final_dataaaa.xlsx'
# df_imputed.to_excel(output_file_path, index=False)
#

