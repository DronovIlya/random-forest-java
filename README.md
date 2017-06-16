# Random forest, Java implementation
Random Forest is a bagging machine learning algorithm for combining multiple decision trees. The prediction is aggregated across all of trees/

The process of building Random Forest in this implementation:
<ul>
 <li>(1) Generate a boostrap sample with replacement from the training data</li>
 <li>(2) Build a tree for the boostrap data, by recursively repeating next steps
    <ul>
       <li>(i) Randomly select variables from the full feature set</li>
       <li> (ii) Using _information gain_ pick the best split-point among selected features</li>
       <li> (iii) Split the node into left and right child. </li>
       <li> (iiii) Split the node into left and right child. </li>
       <li> (iiiii) Repeat it until the minimum node size _min_sample_leaf_ reached </li>
     </ul>
 
 </li>
</ul>

<b> Tuned parameters <b>
<ul>
 <li>**n_estimators** - Number of trees in the forest </li>
 <li>**min_samples_leaf** - The minimum number of samples required to be at a leaf node </li>
 <li>**max_features** - The number of features to consider when looking for best split
</ul>


<b> Comparison <b>
There is some comparision between Java Random Forest and sklearn's Random Forest on<a href="https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset">Spine</a>dataset.
You may look at it in notebook/spine-RandomForest.ipynb