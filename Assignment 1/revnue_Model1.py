# %% Step 6

bid_LR, _ = runOpt(y_pred_test_CF, spot_test, up_test, down_test)
revenue_LR = revenue_calc(y_test,bid_LR, spot_test, up_test, down_test)

bid_Poly, _ = runOpt(y_pred_test_poly, spot_test, up_test, down_test)
revenue_Poly = revenue_calc(y_test, bid_Poly, spot_test, up_test, down_test)

bid_WR, _ = runOpt(y_pred_WR, spot_test, up_test, down_test)
revenue_WR = revenue_calc(y_test, bid_WR, spot_test, up_test, down_test)

bid_linL1, _ = runOpt(y_pred_linL1, spot_test, up_test, down_test)
revenue_linL1 = revenue_calc(y_test, bid_linL1, spot_test, up_test, down_test)

bid_linL2, _ = runOpt(y_pred_linL2, spot_test, up_test, down_test)
revenue_linL2 = revenue_calc(y_test, bid_linL2, spot_test, up_test, down_test)

bid_PolyL1, _ = runOpt(y_pred_polyL1, spot_test, up_test, down_test)
revenue_PolyL1 = revenue_calc(y_test, bid_PolyL1, spot_test, up_test, down_test)

bid_PolyL2, _ = runOpt(y_pred_polyL2, spot_test, up_test, down_test)
revenue_PolyL2 = revenue_calc(y_test, bid_PolyL2, spot_test, up_test, down_test)

bid_RLW1, _ = runOpt(y_pred_RLW1, spot_test, up_test, down_test)
revenue_RLW1 = revenue_calc(y_test, bid_RLW1, spot_test, up_test, down_test)

bid_RLW2, _ = runOpt(y_pred_RLW2, spot_test, up_test, down_test)
revenue_RLW2 = revenue_calc(y_test, bid_RLW2, spot_test, up_test, down_test)

# %%

# Barplot

regression_models = ["Linear", "Polyn.","Locally\nweighted", "Linear L1", "Linear L2",
                         "Polyn. L1","Polyn. L2", "Locally\nweighted L1", "Locally\nweighted L2"]
revenues = [revenue_LR, revenue_Poly, revenue_WR, revenue_linL1, revenue_linL2, 
                revenue_PolyL1, revenue_PolyL2, revenue_RLW1, revenue_RLW2]
  
fig = plt.figure(figsize = (10, 5))
 
plt.bar(regression_models, revenues, color ='blue', 
        width = 0.4)
plt.ylim(2870,2890)
 
plt.xlabel("Regression models")
plt.ylabel("Max. revenue in EUR")
plt.title("Averaged daily revenue of used models")
plt.show()
