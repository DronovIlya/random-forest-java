package ru.ml.rf.nodes;

/**
 * Decision node of decision tree, for the numerical variables. 
 * Decision node  splits a tree into two sub trees. Some common splitting
 * value for each feature, can be mean, median, or random from 
 * gaussian distribution ...etc. 
 */
public class DecisionNode extends TreeNode {
    
    private int splittingFeatureIndex;
    private double splittingFeatureValue;
    
    TreeNode left;
    TreeNode right;
    
    public DecisionNode(int splittingFeatureIndex, double splittingFeatureValue){
        super();
        this.splittingFeatureIndex = splittingFeatureIndex;
        this.splittingFeatureValue = splittingFeatureValue;
        this.left = null;
        this.right = null;
    }
    
    public void setLeftChild(TreeNode left){
        this.left = left;
    }
    
    public void setRightChild(TreeNode right){
        this.right = right;
    }
    
    public TreeNode getLeftChild(){
        return left;
    }
    
    public TreeNode getRightChild(){
        return right;
    }
    
    public int getSplittingFeatureIndex(){
        return splittingFeatureIndex;
    }
    
    public double getSplittingValue(){
        return splittingFeatureValue;
    }  
    
    public int getDepth(){
        return depth;
    }
    
    public void setDepth(int depth){
        this.depth = depth;
    }
}