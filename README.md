Random forest, Java implementation
==================================
Random Forest is a bagging machine learning algorithm for combining multiple decision trees. The prediction is aggregated across all of trees.

The process of building Random Forest in this implementation:
<ul>
 <li>Generate a boostrap sample with replacement from the training data</li>
 <li>Build a tree for the boostrap data, by recursively repeating next steps
    <ul>
       <li>Randomly select variables from the full feature set</li>
       <li>Using information gain pick the best split-point among selected features</li>
       <li>Split the node into left and right child. </li>
       <li>Split the node into left and right child. </li>
       <li>Repeat it until the minimum node size _min_sample_leaf_ reached </li>
     </ul>
 
 </li>
</ul>

Tuned parameters
================
* **n_estimators** - Number of trees in the forest </li>
* **min_samples_leaf** - The minimum number of samples required to be at a leaf node
* **max_features** - The number of features to consider when looking for best split

Comparison
================
There is some comparision between Java Random Forest and sklearn's Random Forest on <a href="https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset">Spine</a> dataset.
You may look at it in notebook/spine-RandomForest.ipynb