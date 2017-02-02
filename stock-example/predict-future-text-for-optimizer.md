

# Improving Predictions for important classes using the optimizer

The Optimizer seeks to find relationships in the data that would allow it to improve prediction accuracy and to reduce the negative contribution from data that has negative impact on prediction accuracy.   

When predicting stocks for this usecase what we really care about is that when we think a stock will go up it really does go up by at least our 1% before it drops by more than 1%.    This means that predicting for class 1 the stock rising is more important than class 0 the stock falling simply because we will ignore class 0 predictions but if the system predicts a class 1 then we will buy the stock and will loose money when the system is wrong.   

As you can see from the two results below the version without the optimizer was accurate 63.8% of the time but it was only accurate 58% of the time when predicting for class 1.     The optimized version was accurate 65.4% of the time but class 1 received a much larger boost so it improved to 71.4% accurate.    Any level of accuracy above 50% can produce a profitable system. 

The randomizer is essentially a semi-random permutation based system so the results will vary from pass to pass depending on the data but  they can improve prediction accuracy  under specific conditions.   Part of the art of applying these systems is learning how to apply them to the current problem. 

------

**NOTE:  As of 2017-01-30 I broke the Optimizer when adding in another feature.     I want to implement  pre-analyzer before working on fixing the optimizer.  Please [let me know](http://BayesAnalytic.com) if you need the optimizer working and I will boost it's priority.** 

------



#### Example Output with the Optimizer

```
numRow=174  sucCnt=112 precis=0.6436782 failCnt=62 failPort=0.3563218
Summary By Class
Train Probability of being in any class
class=0,  cnt=2635  prob=0.6694215
class=1,  cnt=1298  prob=0.3305785
Num Train Row=1573 NumCol=8

RESULTS FOR TEST DATA
  Num Train Rows=174
class=0 ClassCnt=109 classProb=0.62643677, Predicted=167 Correct=107  recall=0.98165137  Prec=0.6407186 Lift=0.014281809
class=1 ClassCnt=65 classProb=0.37356323, Predicted=7 Correct=5  recall=0.07692308  Prec=0.71428573 Lift=0.3407225
Finished ClassifyTestFiles()
```

#### Example Output without the Optimizer

```
numRow=174  sucCnt=111 precis=0.63793105 failCnt=63 failPort=0.36206895
Summary By Class
Train Probability of being in any class
class=1,  cnt=520  prob=0.3305785
class=0,  cnt=1053  prob=0.6694215
Num Train Row=1573 NumCol=8

RESULTS FOR TEST DATA
  Num Train Rows=174
class=1 ClassCnt=65 classProb=0.37356323, Predicted=12 Correct=7  recall=0.10769231  Prec=0.5833333 Lift=0.20977008
class=0 ClassCnt=109 classProb=0.62643677, Predicted=162 Correct=104  recall=0.95412844  Prec=0.6419753 Lift=0.015538514
```

