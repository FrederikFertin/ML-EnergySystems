## Revenue Calculation
def revenue_calc(y_test,y_bid, ahead_rev, Up_rev, Down_rev):
   act_pow = y_test[:-2]
   bid_pow = y_bid[:-2]
   start= len(y_train)
   stop = start+len(bid_pow)-1
   revenue = 0
   for i in range(0,len(bid_pow)-1):
      revenue += ahead_rev[i]* y_pred[i]
      if (act_pow[i]-bid_pow[i])>0:
         revenue += Down_rev[i]*(act_pow[i]-bid_pow[i])
      else:
         revenue += Up_rev[i]*(act_pow[i]-bid_pow[i])
   revenue = round(revenue / len(bid_pow) *24, 2)
   return("Average daily revenue in EUR: " + str(revenue))



