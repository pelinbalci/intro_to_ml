# PCA

Two definitions of variation:
- The willingness/flexibility of an algorithm to learn.
- Technical term in statistics - the spread of a data distribution. 

PCA finds new coordinate system that's obtained from the old one by translation and rotation only. 
- Find the center of the data.
- This is the center of our new coordinate system. 
- The principal axis is the axis where we see the most variation. 
- y axis is orthagonal to that where less important directions of variation.


Make a `composite feature` that more directly probes the underlying phenomenon.

What is composite feature? ---> Principal Component.

We will use PCA for:
- dimensionality reduction
- unsupervised learning. 

##  How to get PCA?

💛Principal component is NOT a linear regression. We are not trying to predict anything.
 
💛We are trying to find a direction in the data so that we loose minimum information.

💛After finding the principal component, project all data to that new coordinate system.

💛For example you can down the dimensionality from 2 to 1 by projection. 

💛 Principal component of a dataset is the direction that has the largest variance because retains maximum amount of information in original data. 

💛 Calculate the sum of the distances between the data point and the principal component. --> Information Loss

💛 PCA is a general algorithm for feature transformation. 

💛 with PCA you can find that there are k things that drive the target value. 

💛 Use principal components as new features.

💛 Principal components are ranked with the variance, the mst variant the first principal component, the next one is second principal component, etc.

💛 All principal components are orthogonal to each other, they don't overlap. 

💛 max number of principal components = number of fetures. 


##  When to use it?

- Is there a latent feature?
- Dimensionality reduction
    - It is hard to visualize high dim data. 
    - reduce noise
    - make other algorithms (regression or classification) work better. 
    - eigenfaces 

