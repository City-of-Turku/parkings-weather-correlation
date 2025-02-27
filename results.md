# Analysis of Parking Forecasting Models

## Methodology for Weather Impact Assessment

The weather impact analysis was conducted using a consistent methodology across all models and locations:
1. **Base Forecast**: Each model was first used to generate a 30-day forecast without including any weather variables. This established the baseline prediction. The total number of cars parked over the 30-day forecast period is calculated as the sum of the predicted hourly parked cars across all hours within those 30 days.

2. **Weather Scenario Forecasts**: The same models were then used to generate 30-day forecasts with specific static weather conditions applied across the entire period:
   - **Rain scenarios**: Continuous rain measure and boolean rain indicator >=3mm/h
   - **Temperature scenarios**: Fixed temperatures at -20°C, -10°C, 0°C, +10°C, and +20°C
   - **Combined scenarios**: Rain and specific temperatures together (Prophet only)

3. **Impact Measurement**: The effect of each weather condition was calculated by:
   - Comparing the total forecasted cars for each weather scenario against the baseline
   - Calculating the absolute difference in forecasted parking volume
   - Computing the percentage change relative to the baseline forecast

4. **Cross-Model Comparison**: The results from different models (Prophet, MLPRegression, LinearRegression, XGBoost) were compared to identify consistent patterns and notable differences.

This approach isolates the specific impact of each weather variable while holding all other factors constant, allowing for a clear assessment of how different weather conditions might affect parking demand across locations.


## Model Performance Comparison

### Accuracy Across Models
1. **Prophet Models**:
   - Zone 1: Moderate accuracy (RMSE: 65.9, MAE: 53.0, Coverage: 88%)
   - Porthaninkatu 6: Moderate errors (RMSE: 9.9, MAE: 7.8, Coverage: 86%)
   - Puutarhakatu 6: Decent accuracy (RMSE: 1.2, MAE: 0.9, Coverage: 90%)

2. **MLPRegression Models**:
   - Zone 1: MSE of 3625 (base) improving to 3182 with temperature variables
   - Porthaninkatu 6: MSE of 74 (base) improving to 68 with temperature variables
   - Puutarhakatu 6: MSE ranging from 0.93 to 0.96

3. **LinearRegression Models**:
   - Zone 1: Highest error rates (MSE: 5370-5588)
   
4. **XGBoost Models**:
   - Zone 1: MSE of 4699 (base) improving to 4130 with temperature variables

### Key Performance Insights
- MLPRegression consistently shows the lowest error rates across locations
- Weather variables (especially temperature) improve model accuracy slightly
- All models exhibit similar patterns in error metrics
- Prophet models provide consistent coverage in the 86-90% range

## Weather Impact Analysis

### Temperature Effects

1. **Cold Weather Impact (-20°C, -10°C)**:
   - **Prophet models**:
     - Zone 1: Slight increase (+0.9% to +1.6%)
     - Porthaninkatu 6: Substantial increase (+22.1%)
     - Puutarhakatu 6: Significant increase (+11.2% to +14.2%)
   - **MLPRegression models**:
     - Zone 1: Significant increase (+9.3% to +12.0%)
     - Porthaninkatu 6: Dramatic increase (+38.6% to +73.6%)
     - Puutarhakatu 6: Substantial increase (+43.8% to +93.1%)
   - **LinearRegression (Zone 1)**: Moderate increase (+4.7% to +6.7%)
   - **XGBoost (Zone 1)**: Decrease (-4.2%)

2. **Mild Weather Impact (0°C)**:
   - **Prophet models**: Minimal change (-0.4% to +1.6%)
   - **MLPRegression models**: 
     - Zone 1: Increase (+11.2%)
     - Porthaninkatu 6: Increase (+15.8%)
     - Puutarhakatu 6: Slight increase (+6.5%)
   - **LinearRegression (Zone 1)**: Small increase (+2.6%)
   - **XGBoost (Zone 1)**: Decrease (-7.2%)

3. **Warm Weather Impact (+10°C, +20°C)**:
   - **Prophet models**:
     - Zone 1: Slight decrease (-0.4% to -1.1%)
     - Porthaninkatu 6: Substantial decrease (-20.3% to -33.1%)
     - Puutarhakatu 6: Moderate increase (+4.3% to +6.4%)
   - **MLPRegression models**:
     - Zone 1: Significant increase (+12.7% to +14.7%)
     - Porthaninkatu 6: Slight increase (+5.8% to +8.5%)
     - Puutarhakatu 6: Modest decrease to slight decrease (-14.0% to -3.9%)
   - **LinearRegression (Zone 1)**: Slight change (+0.5% to -1.6%)
   - **XGBoost (Zone 1)**: Increase (+1.1% to +4.3%)

### Rain Effects

1. **Continuous Rain Measure**:
   - **Prophet models**:
     - Zone 1: Decrease (-3.1%)
     - Porthaninkatu 6: Decrease (-8.2%)
     - Puutarhakatu 6: Decrease (-5.1%)
   - **MLPRegression models**:
     - Zone 1: Slight decrease (-1.6%)
     - Porthaninkatu 6: Decrease (-5.2%)
     - Puutarhakatu 6: Significant decrease (-14.6%)
   - **LinearRegression (Zone 1)**: Decrease (-2.4%)
   - **XGBoost (Zone 1)**: Slight increase (+1.2%)

2. **Boolean Rain Indicator**:
   - **Prophet models**:
     - Zone 1: Decrease (-4.5%)
     - Porthaninkatu 6: Substantial decrease (-16.9%)
     - Puutarhakatu 6: Minimal decrease (-0.9%)
   - **MLPRegression models**:
     - Zone 1: Dramatic increase (+24.5%)
     - Porthaninkatu 6: Increase (+15.2%)
     - Puutarhakatu 6: Significant decrease (-15.6%)
   - **LinearRegression (Zone 1)**: Decrease (-4.8%)
   - **XGBoost (Zone 1)**: Slight increase (+0.6%)

## Location-Specific Patterns

### Zone 1 (Broader Area)
- More consistent results than for parking areas, Porthaninkatu 6 and Puutarhakatu 6
- More moderate weather effects than for individual parking areas
- MLPRegression and Prophet models show opposite temperature effects
- Rain generally decreases parking demand (except in XGBoost)
- Weather effects magnitude: 1-15% (most models), with outlier of 24.5% for boolean rain in MLPRegression

### Porthaninkatu 6
- Most sensitive to temperature changes
- Extreme cold dramatically increases parking (+22% to +74% in MLPRegression)
- Extreme heat significantly decreases parking (-20% to -33% at +20°C in Prophet)
- Rain generally decreases parking, except Boolean rain in MLPRegression
- Shows the most extreme variations across weather conditions

### Puutarhakatu 6
- Cold temperatures affects the most
- Inconsistent warm weather effects across models (Prophet increases and MLPRegression decreases)
- Smallest absolute volume of cars (forecasts in hundreds rather than thousands)
- Rain consistently decreases parking demand
- Shows non-linear temperature relationships in Prophet model

## Model Agreement and Contradictions

### Areas of Agreement
1. **Cold weather increases parking** in most locations and models
2. **Rain generally decreases parking** across most locations and models
3. **Temperature has stronger effects** than rain in most scenarios
4. **Weather variables improve model performance metrics slightly**

### Notable Contradictions
1. **XGBoost temperature effects** are opposite to other models for Zone 1
2. **Boolean rain effects in MLPRegression** show increased parking (contrary to most models)
3. **Warm temperature effects** are inconsistent across locations and models
4. **MLPRegression shows dramatically larger weather effects** than Prophet

## Practical Implications

1. **Forecasting Reliability**:
   - Weather effects (typically 1-15%) are often smaller than model error rates
   - Prophet models provide good coverage but moderate error rates
   - MLPRegression offers the best accuracy metrics

2. **Real World Conclusions**:
   - Cold weather consistently increases parking demand across most locations
   - Rain generally reduces parking demand by 3-17% in most scenarios

3. **Model Selection**:
   - MLPRegression offers the best base accuracy
   - Temperature variables improve forecasting performance more than rain variables
   - The choice of rain measure (continuous vs. boolean) significantly impacts forecasts  
      -The threshold for what is considered as rain in boolean measurement should also be taken into consideration 

4. **Location-Specific Insights**:
   - Different parking areas respond uniquely to weather conditions
   - Zonal forecasts show more moderate weather effects than specific locations

## Overall Conclusions

1. **Weather significantly impacts parking demand**, with effects varying by location
2. **Temperature has more consistent and often stronger effects** than precipitation
3. **Cold weather generally increases parking demand** across most locations and models
4. **Rain typically decreases parking demand**, with some model exceptions
5. **Model selection matters**, with different models showing contradictory weather effects
6. **Location-specific factors** lead to substantially different weather sensitivities
7. **Weather impacts are often smaller than model error rates**, limiting practical application

This analysis suggests that while weather variables can improve forecasting accuracy marginally, their practical significance varies by location, model type, and weather condition. The most reliable insights come from areas where multiple models show consistent directional effects.