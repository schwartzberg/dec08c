### What the stock experiments show

None of the tested neural models beats persistence on this test period.<br>
Longer lookback windows do not improve validation error consistently.<br>
Validation selects 60 observations for the RNN and 120 for the LSTM; their test RMSEs are USD 19.78 and USD 19.44, compared with USD 5.32 for persistence.<br>
The 60-observation LSTM has a lower test RMSE of USD 13.40, but its validation result did not select it.<br>
I keep the choice made before examining the test results.

The direct five-step models also fall short of persistence: aggregate test RMSE is USD 19.66 for the RNN and USD 18.84 for the LSTM, compared with USD 9.17 for persistence.<br>
These figures cover 19 forecast origins and 95 origin–horizon pairs.<br>
Errors generally increase at longer horizons, although the LSTM's fifth-step error is slightly lower than its fourth-step error.<br>
The overlapping forecasts do not provide 95 independent test observations.

The Problem 2 test prices exceed the maximum price used to fit the scaler, so their scaled values exceed one.<br>
I retain that training scale and do not clip the later prices.<br>
These experiments show no forecasting improvement over the most recent observed opening price for the 23 test trading days.
