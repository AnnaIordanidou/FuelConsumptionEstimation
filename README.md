# Fuel consumption estimation in automotive vehicles using OBD-II data processing
This project constitutes the master's thesis for the postgraduate program of the Department of Computer Engineering and Informatics.
You can read my thesis here: https://hdl.handle.net/10889/30005



## Abstract 

This project investigates fuel consumption estimation using vehicle diagnostic system data, focusing on driving behavior. Two models were developed: an instantaneous and a weighted model. The data was cleaned and transformed, enriched with new variables like acceleration and fuel consumption, and subjected to feature analysis using the Random Forest algorithm. Drivers were grouped based on behavior variables, and three profiles were created: Aggressive, Normal, and Slow/Eco. The study found that drivers from the Aggresive group tend to have higher fuel consumption values. The instantaneous model used regression algorithms like Random Forest, SVM, and Linear Regression, with Random Forest excelling in accuracy and generalizability. The weighted model calculated fuel consumption per driver, combining the percentage of participation in each behavior with average consumption observed in each category. The models showed encouraging performance during training, but no model achieved satisfactory levels of prediction accuracy.


## Data Cleaning and missing values
In the first step, we clean the data by removing the empty columns and also values where the engine runtime is zero and speed is also zero. We do this as we want the vehicle to be active so we can calculate its consumption. We also remove the values where speed and RPM are zero at the same time, cause when the car is idling, that is, when it is not moving but the engine is still running, the RPM value typically ranges between 600 to 900 RPM. This can happen when the vehicle is at a stoplight, stopped in traffic, at a crosswalk, or has just parked, but the engine has not yet been turned off. As the data comes from a real-time sensor, there is a possibility that some of it is incorrect. This can happen either due to a lag or error of OBD, or because the vehicle has a start-stop system. In the characteristics of the experimental car, there is no mention of such a system, so it is an OBD problem, so we remove this data. We will remove the lines where the speed and the RPM are zero.  <br/>

To handle missing values, 3 different methods were tested
- mean
- median for values ​​with low correlation and MICE for values ​​with high correlation

```
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
```

- K-NN


After statistical analysis of all three methods, and calculating the difference between the original and the imputed data set, we confirm that the applied techniques were strictly limited to the completion of missing values, without altering the statistical structure of the remaining data. Therefore, the final form of the data set maintains the reliability of the original observations and can be safely used in the subsequent stages of the analysis. The MICE algorithm, an advanced technique for iterative prediction of missing values, was chosen for its ability to account for multivariate data relationships. This hybrid approach balances simplicity and efficiency with statistical validity, ensuring the integrity of the analysis based on the most complete and reliable set of observations.




## New Variables
We calculated 3 new variables for each driver 
- Acceleration
```math
\bar{a}=\frac{VS-VS_0}{time_{end}-time_{start}}=\frac{\Delta \upsilon}{\Delta t}
```


- Fuel consumption using MAF and speed
```math
  \text{Fuel Consumption}_{\text{MAF}}=\frac {VS\times \alpha}{MAF} \times\beta
```
  where $$\alpha = 7.718$$ is a constant to convert the value of f to US MPG, and $$\beta$$ is a constant to convert MPG to liters per 100km

- Fuel consumption using RPM and throttle position
```math  
\text{Fuel Consumption}_{\text{RPM}} = p00\times x^2 + p10\times x + p01\times x\times y
```
  where x and y are the RPM and throttle position and p00 = 2.685, p10 = -0.1246, p01 = 1.243

  

  ## Feature importance
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
```






## Clustering
In this part we grouped the drivers into categories. We used two methods, K-Means and DBSCAN to find out which one has the best result.

```
behavior_features = df_imputed[['ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF',
                                'Acceleration', 'THROTTLE_POS', 'TIMING_ADVANCE']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(behavior_features)

inertia = []
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)


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
```


![cluster](https://github.com/user-attachments/assets/83e48d2a-1931-4711-8da5-1a820bd99ea5)


The K-Means gave us three clusters. CLuster 2 gave us the highest values for the features ENGINE_RPM', 'ENGINE_LOAD', 'SPEED', 'MAF','Acceleration', 'THROTTLE_POS', 'TIMING_ADVANCE', so we name this group as Aggressive one, while the cluster with the lowest value labeled as Slow/Eco driving. We also created the cluster proportions for each driver.


![drivercluster](https://github.com/user-attachments/assets/29c125c0-f083-45c5-802c-040d2ec2c1ca)




## Results of Fuel Consumption per driving behavior

Aggressive drivers appeared to have the lowest fuel consumption according to the FL MAF metric. However, this does not necessarily reflect real-world efficiency. The formula
```math
  \text{Fuel Consumption}_{\text{MAF}}=\frac {VS\times \alpha}{MAF} \times\beta
```
places airflow (MAF) in the denominator. In aggressive driving, both speed and MAF increase, but if MAF increases faster, the resulting value becomes artificially lower. This biases the metric to underestimate consumption in high-MAF conditions. This highlights a limitation of the FL MAF method, suggesting it should be interpreted cautiously and ideally combined with other indicators.


The second method, which incorporates RPM and throttle position, aligns much more closely with expected outcomes:

- Aggressive drivers show higher fuel consumption, with a broader and less stable distribution—reflecting frequent acceleration and high engine loads.

- Eco drivers consistently have the lowest consumption, with a tighter spread and lower RPM values, indicating smoother and more efficient driving.

$`\text{Fuel Consumption}_{\text{MAF}}`$          |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![mafdriving](https://github.com/user-attachments/assets/fb618fbc-bfc9-4d52-81a6-9b84fa43aa37) | ![rpmdriving](https://github.com/user-attachments/assets/72a9055b-3883-40b8-8aea-e2a1f49c69bb)


## Estimation models
The technique and findings of predictive modeling experiments that were conducted
to estimate fuel consumption using two different methodologies are presented in
this section. While the second study examines whether driving behavior clusters,
which are formed from unsupervised learning, can forecast a driver’s average fuel
consumption, the first study concentrates on predicting instantaneous fuel consumption
values based on raw sensor readings from the car (OBD data). Both real-time
and profile-level estimation are possible with this dual method, which offers further
insights into how driving behavior affects fuel efficiency.

We trained three different regression models to explore both linear and non-linear
relationship, Linear Regression, Random Forest and SVR.
To optimize the SVR model for fuel consumption prediction, GridSearchCV was used to search for the best combination of hyperparameters (C, epsilon, gamma, and kernel).
Initially, a wide parameter grid was defined, but the high number of combinations—combined with 5-fold cross-validation and a large dataset—led to excessive memory usage and system errors during execution.
To resolve this, the grid was reduced to fewer values per parameter, and the number of cross-validation folds was decreased. This simplification allowed the process to run successfully while still achieving a good balance between computational efficiency and model performance.




### Instantaneous Fuel Consumption Estimation using OBD Features


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


        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)       
```


Random Forest has the best results for both fuel consumption formulas with $`R^2=0.94`$ for $`\text{Fuel Consumption}_{\text{MAF}}`$ and $`R^2=0.99`$ for $`\text{Fuel Consumption}_{\text{RPM}}`$. These results highlight the suitability of Random Forest for real-time, sensor-based estimation tasks in automotive contexts, particularly when dealing with dynamic and noisy OBD data streams.


$`\text{Fuel Consumption}_{\text{MAF}}`$          |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![randomforestmaf](https://github.com/user-attachments/assets/eb97fcfd-86eb-492a-91f9-fe1cb1c6ea7b) | ![rpmrandomforest](https://github.com/user-attachments/assets/bc19c1c3-b08d-4e2c-93ad-a06d6047da4a)


Linear regression often underestimates fuel consumption at higher actual values and overestimates it when actual values are low, as shown in the figure. While it generally captures the overall trend, it fails to represent the full range of variation in the data.

A major limitation of this model is that it occasionally produces negative predictions for fuel consumption—values that are physically meaningless. This indicates that linear regression lacks the flexibility to capture the complex, nonlinear relationships present in the dataset.

In terms of performance, the model achieved an R² of 0.73 in the first scenario and improved to 0.96 in a later configuration. However, the issue of negative predictions remained unresolved, highlighting the limitations of using linear regression for this task.


$`\text{Fuel Consumption}_{\text{MAF}}`$  |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![linearmaf](https://github.com/user-attachments/assets/2792f835-20f6-4ca1-827c-cf9d2b657f65) | ![rpmlinear](https://github.com/user-attachments/assets/cc713ba0-feee-4d95-a58b-306dfb71c3bd)




The Support Vector Regression (SVR) model yielded the poorest performance, with R² scores of -0.09 (FL MAF) and 0.027 (FL RPM). These results clearly indicate that the model failed to capture the underlying patterns in the data—either due to its limited flexibility or sensitivity to noise and outliers.

Hyperparameter tuning was performed using GridSearchCV, but even with optimized settings, the model underperformed. For the FL MAF variable, the best parameters identified were:

- C = 100

- epsilon = 0.2

- gamma = 'scale'

- kernel = 'rbf'

For the FL RPM variable, the optimal parameters were slightly different:

- C = 100

- epsilon = 0.1

- gamma = 'scale'

- kernel = 'rbf'

Despite careful tuning, the SVR model's sensitivity to parameter selection and lack of robustness made it unsuitable for this fuel consumption prediction task.


$`\text{Fuel Consumption}_{\text{MAF}}`$  |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![mafsvr](https://github.com/user-attachments/assets/89c5a393-a38c-45b3-95f3-186b6497ab6e) |![rpmsvr](https://github.com/user-attachments/assets/29c9530e-f24a-4560-85cf-4f308a00aa27)







### Fuel Consumption Prediction Using Driving Behavior Profiles

To better assess the impact of driving behavior on overall fuel consumption, a weighted model was developed. Instead of analyzing each trip or segment separately, this approach calculates a single, aggregated fuel consumption score per driver, taking into account how often they exhibit aggressive, normal, or economical driving patterns.

Each driver's fuel consumption is weighted by the proportion of time they spend in each behavior cluster. This provides a more holistic view of their long-term driving habits and allows us to explore whether it's possible to predict their overall consumption based solely on their behavior profile.

The goal of this model is not just prediction accuracy, but to explore how well behavioral patterns alone can explain real-world fuel usage trends. 

For the second method, we calculated the weighted fule consumption to use it as a target for our model, according to this formula:
```math
\text{Weighted Fuel Consumption} = p_1 * FL_{Aggressive} + p_2*FL_{Normal} + p_3*FL_{Slow/Eco}
```
where $`p_1, p_2, p_3 `$ are the cluster proportions.



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


        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
```

Initially, the Random Forest model showed significant differences between the two cases. In the case of FL MAF weighted, despite the high performance in training with R² of about 0.87, the model suffered from overfitting, with the R² in the test falling negatively (-2.92). This is probably due to the small size of the data and the complexity of the relationship between driving behavior and fuel consumption. Although the predictions were not accurate in this case, the model showed a more consistent performance for FL RPM weighted, with R² of about 0.93 in both training and testing, indicating that it was able to explain more than 90% of the variance of the target variable. However, a tendency to underestimate high consumption values ​​was observed for some drivers, without this significantly affecting the overall accuracy.


$`\text{Fuel Consumption}_{\text{MAF}}`$          |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![mafwrandom](https://github.com/user-attachments/assets/40374cbd-c6f8-4275-b2b9-148d3c0a4109) | ![rpmwrandom](https://github.com/user-attachments/assets/b77b2945-3f70-49ab-bf80-dd1c74f6aa0e)



The Linear Regression model consistently performed poorly in both cases. In FL MAF weighted, training performed moderately (R² around 0.5), but test performance was very poor with a negative R² (-4.18). The model had a systematic tendency to overestimate fuel consumption, especially for drivers with low actual values. In FL RPM weighted, although the training model scored a satisfactory R² (0.807), test performance was poor, with underestimation of large values ​​and a tendency to compress predictions towards the mean. This suggests that the linear model is unable to capture the nonlinear relationships and complexity of the data.





$`\text{Fuel Consumption}_{\text{MAF}}`$          |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![mafwlinear](https://github.com/user-attachments/assets/cea83acf-4d62-4f62-a5ef-7abf62fe217c) | ![rpmwlinear](https://github.com/user-attachments/assets/c6602ad0-86a7-4ed7-b104-fe7995a68103)


The Support Vector Regression (SVR) model performed the worst in both cases. For FL MAF weighted, the R² was -0.09, while for FL RPM weighted it was even lower (-0.247). Despite attempts to optimize the hyperparameters (C=100, epsilon=0.1, gamma=’scale’, kernel=’rbf’), the model failed to capture the variability of the data, presenting predictions clustered around a fixed mean value. This led to an underestimation of the extreme values, i.e. high fuel consumption, which reduces its effectiveness in cases with high heterogeneity in the data.

$`\text{Fuel Consumption}_{\text{MAF}}`$          |  $`\text{Fuel Consumption}_{\text{RPM}}`$
:-------------------------:|:-------------------------:
![mafwsvr](https://github.com/user-attachments/assets/91ef7fcb-0756-429a-88f3-2424a94252fc) | ![rpmwsvr](https://github.com/user-attachments/assets/0f881a68-238c-47ca-9022-d9b8f343a41a)





### Conslusion

Among all models tested, Random Forest consistently delivered the best performance, achieving R² scores of 0.94 (FL MAF) and 0.99 (FL RPM). Its robustness to noise and ability to model nonlinear relationships made it well-suited for this type of real-world data.

Linear Regression, although showing strong R² in the FL RPM case (0.96), suffered from prediction errors—most notably producing negative fuel values, due to its linearity and sensitivity to outliers.

Support Vector Regression (SVR) performed poorly overall, with negative R² values on both training and test sets. This was attributed to the model’s high sensitivity to hyperparameter selection and its limited resilience to noisy data.

In a follow-up experiment, the idea of predicting overall fuel consumption based solely on driving behavior profiles (e.g., proportion of aggressive vs. economical driving) was tested. However, none of the models, including Random Forest, managed to generalize effectively, indicating that behavioral proportions alone are not sufficient predictors in this context—possibly due to limited feature representation or the need for richer temporal and contextual data.

Overall, the study highlights the strength of ensemble models like Random Forest in complex, sensor-based prediction tasks and the importance of combining both technical vehicle signals and contextual driving behavior for accurate fuel consumption estimation.
