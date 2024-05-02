# To Load the necessary libraries
library(forecast)
library(zoo)

# To Load the dataset
eeg.data <- read.csv("C:/Users/STSC/Desktop/EEGEyeState.csv")

# To Look at the structure of the dataset
str(eeg.data)

# To Summarize the data using summary() function to identify anomalies and get an overview
summary(eeg.data)

# To Check for missing values and handle them if found
sum(is.na(eeg.data))
eeg.data <- na.omit(eeg.data)

# Converting the columns to numeric where appropriate
numeric_columns <- c('X.ATTRIBUTE.AF3.NUMERIC', 'X.ATTRIBUTE.F7.NUMERIC', 'X.ATTRIBUTE.F3.NUMERIC', 'X.ATTRIBUTE.FC5.NUMERIC', 'X.ATTRIBUTE.T7.NUMERIC', 'X.ATTRIBUTE.P7.NUMERIC', 'X.ATTRIBUTE.O1.NUMERIC', 'X.ATTRIBUTE.O2.NUMERIC', 'X.ATTRIBUTE.P8.NUMERIC', 'X.ATTRIBUTE.T8.NUMERIC', 'X.ATTRIBUTE.FC6.NUMERIC', 'X.ATTRIBUTE.F4.NUMERIC', 'X.ATTRIBUTE.F8.NUMERIC', 'X.ATTRIBUTE.AF4.NUMERIC', 'X.ATTRIBUTE.eyeDetection')
eeg.data[numeric_columns] <- lapply(eeg.data[numeric_columns], function(x) as.numeric(as.character(x)))

# To Check for any NA values that might have been introduced during the conversion
sapply(eeg.data[numeric_columns], function(x) sum(is.na(x)))

# Checking for missing values
sum(is.na(eeg.data))

# Creating time series object using ts() function
eeg.ts <- ts(eeg.data, frequency=1)

# Plotting time series data by using plot()
plot(eeg.ts[,1], main = "EEG Channel 1 Time Series", xlab = "Time", ylab = "Readings")
for(i in 2:ncol(eeg.ts)) {
  plot(eeg.ts[,i], main=paste("EEG Time Series - Channel", i), xlab="Time", ylab="Reading")
}

# Establish x-axis scale interval using range of data
time_index <- seq_along(eeg.ts[, "X.ATTRIBUTE.AF3.NUMERIC"])

# To Convert the AF3 channel to a numeric vector if not already
AF3 <- as.numeric(eeg.data$`X.ATTRIBUTE.AF3.NUMERIC`)

# Calculating z-scores To Identify and replace outliers
# Calculate the median value
AF3_median <- median(AF3,na.rm = TRUE)
print(AF3_median)
# Compute mean of AF3data, ignoring any NA values
AF3_mean <- mean(AF3, na.rm = TRUE)
print(AF3_mean) 
# Calculate standard deviation 
AF3_sd <- sd(AF3, na.rm = TRUE)
print(AF3_sd)  
# Compute z-score foreach data point
AF3_z <- (AF3 - AF3_mean) / AF3_sd
print(head(AF3_z))  
# Identify datapoints which have a z-score with an absolute value greater than 3
# which is a common cut-off for identifying outliers in many fields
outliers <- which(abs(AF3_z) > 3)
print(outliers)  

if(length(outliers) > 0) {
  # Replace identified outliers with median
  AF3[outliers] <- AF3_median
  print(paste("Replaced", length(outliers), "outliers with median value."))
} else {
  print("No outliers detected.")
}


# Converting back to a ts object to continue with time series analysis
AF3_ts <- ts(AF3, frequency=1)

# Trend Analysis with specified y-axis limits
plot(AF3_ts, main = "AF3 EEG Readings Over Time", ylim = c(4000, 4700), xlab = "Time", ylab = "AF3 Readings")
axis(2, at=seq(4000, 4700, by=100), labels=seq(4000, 4700, by=100))  # Custom y-axis ticks
lines(lowess(AF3_ts), col = "red", lwd = 2)  # Adding a smoothed trend line

# Seasonality Check
acf(AF3_ts, main="ACF for AF3")  # Autocorrelation to check for seasonality
pacf(AF3_ts, main="PACF for AF3")  # Partial autocorrelation


# Applying a simple moving average with a window size of 5
moving_avg <- filter(AF3_ts, rep(1/5, 5), sides=2)
moving_avg

# To Plot the moving average using plot() function
plot(moving_avg, main="Moving Average of AF3 Readings", xlab="Time", ylab="AF3 Readings", col="blue")
lines(AF3_ts, col="black")
legend("topright", legend=c("Moving Average", "Actual"), col=c("blue", "black"), lty=1, cex=0.8)


# Set the proportion of the dataset to use for training (e.g., 80% for training)
train_proportion <- 0.8

# To Calculate the number of observations to include in the training set
train_size <- floor(train_proportion * length(AF3_ts))

# Splitting the data into training and validation sets
train_set <- AF3_ts[1:train_size]
valid_set <- AF3_ts[(train_size + 1):length(AF3_ts)]

# Fitting the exponential smoothing model to the training set
exp_smoothing_model_train <- ets(train_set)

# To Forecast using the model
exp_smoothing_forecast <- forecast(exp_smoothing_model_train, h = length(valid_set))
exp_smoothing_forecast

# Plotting the forecast using plot() function
plot(exp_smoothing_forecast, main="Exponential Smoothing Forecast vs. Actual", xlab="Time", ylab="AF3 Readings")
lines(seq_along(valid_set) + length(train_set), valid_set, col='red')
legend("bottomleft", legend=c("Exponential Smoothing Forecast", "Actual Validation Data"), col=c("blue", "red"), lty=1, cex=0.8)

# Calculate and print the accuracy of the exponential smoothing forecast using accuracy() function
exp_smoothing_accuracy <- accuracy(exp_smoothing_forecast, valid_set)
print(exp_smoothing_accuracy)

# Fit linear, quadratic, and ARIMA models to the cleaned dataset
time_index <- 1:length(AF3_ts)
linear_model <- lm(AF3_ts ~ time_index)
quadratic_model <- lm(AF3_ts ~ time_index + I(time_index^2))
ar1_model <- Arima(linear_residuals, order = c(1,0,0))
auto_fit <- auto.arima(AF3_ts)

# Using summary() function
summary(linear_model)
summary(quadratic_model)
summary(ar1_model)
summary(auto_fit)

# Model diagnostics
checkresiduals(auto_fit)

# Fit linear, quadratic, and ARIMA models to the original data
time_index_original <- 1:length(AF3.ts)
linear_model_original <- lm(AF3.ts ~ time_index_original)
quadratic_model_original <- lm(AF3.ts ~ time_index_original + I(time_index_original^2))
auto_fit_original <- auto.arima(AF3.ts)

# Using summary() function to Summarize the original models
summary(linear_model_original)
summary(quadratic_model_original)
summary(auto_fit_original)

# Model diagnostics
checkresiduals(auto_fit_original)


# Fit linear, quadratic, and ARIMA models to the training set
time_index_train <- 1:length(train_set)
linear_model_train <- lm(train_set ~ time_index_train)
quadratic_model_train <- lm(train_set ~ time_index_train + I(time_index_train^2))
auto_fit_train <- auto.arima(train_set)

# # Using summary() function to Summarize the models on the training set
summary(linear_model_train)
summary(quadratic_model_train)
summary(auto_fit_train)

# Model diagnostics
checkresiduals(auto_fit_train)


# Forecast on the validation set using the models
# Linear prediction
linear_predictions <- predict(linear_model_train, newdata = data.frame(time_index_train=(train_size+1):length(AF3_clean)))
summary(linear_predictions)
# Quadratic prediction
quadratic_predictions <- predict(quadratic_model_train, newdata = data.frame(time_index_train=(train_size+1):length(AF3_clean)))
summary(quadratic_predictions)
# ARIMA prediction
arima_forecast <- forecast(auto_fit_train, h = length(valid_set))
summary(arima_forecast)
# Plotting the validation set and the forecasts using plot() function
plot(time_index_valid, valid_set, type = 'l', col = 'black', ylim = c(4000, 4700), xlab = "Time", ylab = "AF3 Readings", main = "Validation Set vs. Forecasts", yaxt = 'n')
axis(2, at=seq(4000, 4700, by=100), labels=seq(4000, 4700, by=100))
lines(time_index_valid, linear_predictions, col = 'red')
lines(time_index_valid, quadratic_predictions, col = 'blue')
lines(time_index_valid, arima_forecast$mean, col = 'green')
legend("topleft", legend = c("Actual", "Linear", "Quadratic", "ARIMA"), col = c("black", "red", "blue", "green"), lty = 1, cex = 0.8)

# Evaluating forecast accuracy using accuracy() function
print(accuracy(arima_forecast, valid_set))

# Create a new data frame for the time index of the validation set
newdata_linear <- data.frame(time_index = seq_along(valid_set) + length(train_set) - 1)

# Calculate accuracy on the validation (test) set for linear model
linear_predictions <- predict(linear_model_train, newdata = newdata_linear)
accuracy_linear_test <- accuracy(linear_predictions, valid_set)

# Repeat the process for the quadratic model
newdata_quadratic <- data.frame(time_index = seq(from = (length(train_set) + 1), length = length(valid_set)))
newdata_quadratic$time_index_squared <- newdata_quadratic$time_index^2
quadratic_predictions <- predict(quadratic_model_train, newdata = newdata_quadratic)
accuracy_quadratic_test <- accuracy(quadratic_predictions, valid_set)

# For the ARIMA model, we can use the forecast function as usual
accuracy_arima_test <- accuracy(forecast(auto_fit_train, h = length(valid_set)), valid_set)

# Create the accuracy summary data frame
accuracy_summary <- data.frame(
  Model = c("Linear", "Quadratic", "ARIMA"),
  RMSE_Train = c(accuracy_linear_train[1,"RMSE"], accuracy_quadratic_train[1,"RMSE"], accuracy_arima_train[1,"RMSE"]),
  MAPE_Train = c(accuracy_linear_train[1,"MAPE"], accuracy_quadratic_train[1,"MAPE"], accuracy_arima_train[1,"MAPE"]),
  RMSE_Test = c(accuracy_linear_test[1,"RMSE"], accuracy_quadratic_test[1,"RMSE"], accuracy_arima_test[1,"RMSE"]),
  MAPE_Test = c(accuracy_linear_test[1,"MAPE"], accuracy_quadratic_test[1,"MAPE"], accuracy_arima_test[1,"MAPE"])
)

# Round the numeric columns in the accuracy_summary data frame
accuracy_summary[-1] <- round(accuracy_summary[-1], 2)

# Print the updated summary table with MAPE included
print(accuracy_summary)
