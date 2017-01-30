# Quantized Classifier Background #

The documentation in is this repository is the only documentation that exists.   I invented Quantized classifier from scratch 5 years ago.  This is the first time I have published this work.   

> > Some help / Guidance  converting some one of these into a formal paper would be great.   One reason I have been working on the Genomic link listed below is the Galaxy team has several PHD who will co-publish results but that could be a 6 month task.

## Where did the concept Originate ## 

The concept originated basically from splunking around in intermediate ML statisics files generated while I was working on machine learning engines to predict and trade stock price movements.    I found that similar patterns kept re-emerging as I was applying ML and statistical learning machines to new problems at larger companies. 

The core problem many companies face when applying machine learning is engines that can deal with their bewildering scope of data.   Rapidly changing data with a need to make predictions in near real time.     This engine was specifically designed to support that use case. 

## Skewed Dataset & Overlapping Values Weakness  ## 

The Quantized probability style engine  great for the data sets where the underlying assumptions hold true.   It will not do as well as CNN style engines for image recognition.   There are specific conditions where the Quantized probability struggles and for those conditions we provide a  [quantized filter component](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/quant_filt.py) which acts more like a decision tree.

The Probability classifier will struggle when there are a lot of training records for a specific class of data and there are a relatively large number of features that share value range operlap with features that have much smaller populations.   A similar problem is faced by naive Bayes classifiers and can also affect KNN if the overlapping values are close enough.

You can think of the internals of the probability engine as a quarum of smaller classifiers.    When you have a given feature where Class A) has 10,000 records with a value such as 0.21  and Class B that has 500 records with a value of 0.21 then the base probability for that feature  is  95.24%  while class B has a base probability of 4.76%   This would strongly favor class A.  If the other features do not end up out voting  the input from this feature then a record will end up classified class A.  

To overcome this you need some other features that strongly favor class B.   Part of this can be accommodated by an optimizer which can discover that a given feature is yielding negative predictive value.   

In data set where class A has so many more records the optimizer will see more improvement by improving precision of class A even if class B suffers.   If class A is relatively less important and Class B is more important such as detecting Buy signal then you need to optimize for precision of class B even if total set precision for other classes suffer.    The optimizer provides -OptClassId=x feature to support this use case. 

One way to limit the impact of skewed populations is to try and establish data sets so training data has an even mixture but this can be difficult when doing anomaly detect such as in stock market only 1 out of 50 bars really are good buy points.    A better approach is to choose as many features as possible where the value ranges between signals are sufficiently different that the engine does not end classifying them in the same quanta group.   We are working on features to help identify current features that are likely to yield this kind of problem.   

Research continues into technical ways to minimize this impact.   One way to help minimize the impact is to figure out which features deliver interesting predictive value as early in the process as possible.    We are investigating options that test this on a feature by feature basis which can be turned on with a command line option -testFeatureValues with the results saved in a configuration file.  

## Quantized Filter ##

There are two aspects of the Quantized classifier the pure probabilistic portion and the quantized filter portion. I have been focusing on the probabilistic portion because it best fit the need of some early users. 

The quantized filter solves some problems that are difficult to solve using probabilistic models.   such as Skewed Dataset & Overlapping Values.  

The Quantized filter  [quant_filt.py](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/quant_filt.py) still uses the quantized concept but treats quanta more like a decision or splay tree where we can filter out some of the value influence described above. This was easy to prototype in python but it always takes more time to port to a strongly typed language.

There are applications where the the probability approach works best because it can still make a prediction for a row where one or more test / classification values have values way out of range where the QuantFilt approach currently gives up and says it can not classify those records. I am still thinking about how to combined the two to leverage both strengths without making the end user make that decision.

When running the results for a data set that giving the pure probability model problems yielding a maximum precision of 79% the quant_filt algorithm was able to deliver 95.8% precision.    In contrast the quant_filt algorithm did poorly on the stock test where the probability approached worked very well.  

The weakness of the current QuantFilt strategy is that it tries to find the most restrictive quant able to make a match but when it fails it moves to the next least restrictive set of quanta for the entire matching process.    What we really want is for it to back track only for the feature where the most restrictive match did not work reduce the restriction only for that feature and keep the more restrictive requirement for the others. This is complex because overly restrictive setting at any feature can cause failure in all subsequent features the the back track has to work it's way back incrementally.

At the current time the quant filter and quant probability models are separate and stand along.  Adding the ability to use them in combination will is likely the most valuable long term strategy.  It is unclear if it is better to first invest in supporting the back track capacity in the quant filter approach or to focus on identifying ways to use it to support feature couplets and triplets in the probability model where a miss just gives a zero output for that feature group. 



## Feature Combinations

When studying stock data i found that one challenge with features that provide predictive value is that many individual features have relatively little value on a single basis but when combined in  pairs and triplets  they can yield immense value if they each combination can be considered as a group.   

> > For one symbol I found that if the bar price was within 1% of the 30 day low on Mondays and Tuesdays while also being at least 3% above the 10 day low there was a 75% chance that stock would rise by at least 2% by the end of the week.     Each of these measures individually had limited value but together they held substantial value provided it was not within 2 weeks before a scheduled fed rate announcement.     
> >
> > Discovering these dual, tripplet and quad feature combinations that work well for predictive value is one area where additional R&D is planned.     I think it makes sense to integrate an approach that uses a varient of the [quantized filter concept (https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/quant_filt.py)  to identify feature groups that work better as mini-trees and integrate those as part of the larger probability model. 



# Citing Quantized Classifier

All documentation for Quantized classifier are either contained in the main repository or in the associated Wiki.    I will attempt to keep this list up to date as external references become available. 

# Internal Documents #

*  [main Readme](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/README.md)

*  [Notes on using for Genomic Disovery](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/docs/genomic-notes.md?at=default&fileviewer=file-view-default)

*  [Tutorial for use in Stock price prediction](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/wiki/stock-example/predict-future-stock-price-using-machine-learning.md)   This is the first in what I expect to be a series of articles.  I am actually more interested in using the optimizer to try and deduce valuable features and feature patterns using a series of technically derived indicators.   When I find a good and free source of stock articles I will try to integrate  a text processing sentiment indicator merged into the technical features.

*  [Conceptual approach for application in Text classification](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/docs/text-classification/overview-classification.md)  This one includes an extensive set of links that will eventually be moved to the Bibliography. 

*  [design-notes.md](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/docs/design-notes.md)

*  [bibliography.md](https://bitbucket.org/joexdobs/ml-classifier-gesture-recognition/src/default/docs/bibliography.md)

# Prior work that influenced design #

* [Precision VS Recall, a Net Trading Profit Perspective](http://bayesanalytic.com/precision-vs-recall-a-net-trading-profit-perspective/)

* [Applying KNN and Ensemble for stock price prediction](http://bayesanalytic.com/knn-and-ensemble-for-stock-price-prediction/)

* [Bayes Analytic Engine is really Bayes+ Morphed for prediction at high volumes](http://bayesanalytic.com/ok-bayes-analytic-is-really-bayes/)

* [Bayes Analytic Retail Marketing with Machine Learning](http://bayesanalytic.com/bayes-analytic-retail-marketing-machine-learning/)

* [Why did you need to build the Bayes Analytic engine when so many analytic engines are available on the market?](http://bayesanalytic.com/why-not-use-existing-trading-engines-r-or-other-analytic-tools/)

* [DEM – Digital Elevation Model work with Scala](http://bayesanalytic.com/main/technical-engineering/gis-geographic-information-systems/)

* [Sourcecode to save Forex Bar data to CSV via cAlgo bot](http://bayesanalytic.com/sourcecode-to-save-forex-bar-data-to-csv-via-calgo-bot/)

* [Scala fetch stock bar data from Yahoo finance service](http://bayesanalytic.com/main/technical-engineering/utilities-or-source-code-to-download-data/)

* [LuaJIT Access 20 Gig or More of Memory](http://bayesanalytic.com/main/technical-engineering/languages/lua-related-articles/)