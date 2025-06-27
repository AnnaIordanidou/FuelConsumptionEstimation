# FuelConsumptionEstimation
Here you can find my thesis for the postgraduate program of the Department of Computer Engineering and Informatics

### Abstract 

This project investigates fuel consumption estimation using vehicle diagnostic system data, focusing on driving behavior. Two models were developed: an instantaneous and a weighted model. The data was cleaned and transformed, enriched with new variables like acceleration and fuel consumption, and subjected to feature analysis using the Random Forest algorithm. Drivers were grouped based on behavior variables, and three profiles were created: Aggressive, Normal, and Slow/Eco. The study found that drivers from the Aggresive group tend to have higher fuel consumption values. The instantaneous model used regression algorithms like Random Forest, SVM, and Linear Regression, with Random Forest excelling in accuracy and generalizability. The weighted model calculated fuel consumption per driver, combining the percentage of participation in each behavior with average consumption observed in each category. The models showed encouraging performance during training, but no model achieved satisfactory levels of prediction accuracy.

### Data Cleaning and missing values
In the first step, we clean the data by removing the empty columns and also values where the engine runtime is zero and speed is also zero. We do this as we want the vehicle to be active so we can calculate its consumption. We also remove the values where speed and RPM are zero at the same time, cause when the car is idling, that is, when it is not moving but the engine is still running, the RPM value typically ranges between 600 to 900 RPM. This can happen when the vehicle is at a stoplight, stopped in traffic, at a crosswalk, or has just parked, but the engine has not yet been turned off. As the data comes from a real-time sensor, there is a possibility that some of it is incorrect. This can happen either due to a lag or error of OBD, or because the vehicle has a start-stop system. In the characteristics of the experimental car, there is no mention of such a system, so it is an OBD problem, so we remove this data. We will remove the lines where the speed and the RPM are zero.  <br/>

To handle missing values, 3 different methods were tested
- mean
- median for values ​​with low correlation and MICE for values ​​with high correlation
- K-NN

After statistical analysis of all three methods, and calculating the difference between the original and the imputed data set, we confirm that the applied techniques were strictly limited to the completion of missing values, without altering the statistical structure of the remaining data. Therefore, the final form of the data set maintains the reliability of the original observations and can be safely used in the subsequent stages of the analysis. The MICE algorithm, an advanced technique for iterative prediction of missing values, was chosen for its ability to account for multivariate data relationships. This hybrid approach balances simplicity and efficiency with statistical validity, ensuring the integrity of the analysis based on the most complete and reliable set of observations.




### New Variables
We calculated 3 new variables for each driver 
- Acceleration $\bar{a}=\frac{VS-VS_0}{time_{end}-time_{start}}=\frac{\Delta \upsilon}{\Delta t}$


- Fuel consumption using MAF and speed
  
  $$\text{Fuel Consumption}=\frac {VS\times \alpha}{MAF} \times\beta$$
  where $$\alpha = 7.718$$ is a constant to convert the value of f to US MPG, and $$\beta$$ is a constant to convert MPG to liters per 100km

- Fuel consumption using RPM and throttle position
  
  $$\text{Fuel Consumption} = p00\times x^2 + p10\times x + p01\times x\times y $$
  where x and y are the RPM and throttle position and p00 = 2.685, p10 = -0.1246, p01 = 1.243

  
```
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
```



  ### Feature importance
  We calculated the feature importance for both fuel consumption equitions using Random Forest. Below we can see the results:
  ![Figure_2](https://github.com/user-attachments/assets/8956aec3-d578-42fc-9005-cb354d1f9cb4)
  For the equation based on MAF and speed, three main factors affect consumption: speed = 0.316, MAF = 0.226, and engine load = 0.448, while only the RPM (RPM = 0.999) affects the second equation.


```X = df_imputed[['TIME', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'BAROMETRIC_PRESSURE',
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
```


### EDA
In this chapter, exploratory data analysis (EDA) is performed, with the aim of understanding the distribution and dispersion of the variables in the data set. To this end, boxplots for the main numerical variables are presented, without further commentary, in order to clearly capture patterns, anomalies and possible deviations.

![boxplot_SPEED_all_drivers](https://github.com/user-attachments/assets/e0cffa1f-64e4-48d5-9aeb-7d28b88c0776)

![boxplot_ENGINE_RPM_all_drivers](https://github.com/user-attachments/assets/792b21fc-1841-4eae-81a0-7d92b30aedb6)
![boxplot_ENGINE_LOAD_all_drivers](https://github.com/user-attachments/assets/22cfeaad-3474-4914-9aa9-a1edb1cd1b68)
![boxplot_Acceleration_all_drivers](https://github.com/user-attachments/assets/19fb54b8-cbab-406b-a3e2-2f017d750131)
![boxplot_SPEED_all_drivers](https://github.com/user-attachments/assets/3bf1c769-3fb1-45ea-bd98-807f9a911133)
![boxplot_MAF_all_drivers](https://github.com/user-attachments/assets/8e7568fc-fa42-437e-9361-0398e719ba97)

![boxplot_FL_RPM_all_drivers](https://github.com/user-attachments/assets/37eca64a-611c-4b95-8e30-71d1167d3957)
![boxplot_FL_MAF_all_drivers](https://github.com/user-attachments/assets/12eba974-54de-4503-9724-f98ce015fbf0)

![facet_fuel_consumption](https://github.com/user-attachments/assets/5eeb55cf-48c1-438c-b27b-3293fb412ee2)
![engine_rpm_maf](https://github.com/user-attachments/assets/3bdd641d-0303-45a5-90fb-908c3d5e5cda)



### Clustering
In this part we grouped the drivers into categories. We used two methods, K-means and DBSCAN to find out which one has the best result.

```
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
```


![cluster](https://github.com/user-attachments/assets/83e48d2a-1931-4711-8da5-1a820bd99ea5)


The K-Means gave us three clusters. CLuster 2 gave us the highest values for the features ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF','Acceleration', 'THROTTLE_POS', 'TIMING_ADVANCE', so we name this group as Aggressive one, while the cluster with the lowest value labeled as Slow/Eco driving. We also created the combination of the three behaviors for each driver.


```
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


driver_clusters.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set1')
plt.ylabel("Proportion of Driving Segments")
plt.title("Driver Behavior Distribution by Cluster")
plt.legend(title="Cluster")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


df_imputed['Driving_Behavior'] = df_imputed['KMeans_Cluster'].map(cluster_names)

boxplot_stats_dict = {}


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
```


![drivercluster](https://github.com/user-attachments/assets/29c125c0-f083-45c5-802c-040d2ec2c1ca)







### Estimation models
```
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

```




For the second 

```# weighted fuel cons
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
```







```
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
print(results_df.round(3))```
