from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image 

iris = load_iris()
clf = tree.DecisionTreeClassifier()
print ("data")
print (iris.data)

print ("target")
print (iris.target_names)
clf = clf.fit(iris.data, iris.target)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  

# tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,) 
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('arvore.jpg')
Image(graph.create_png()) 
print ("finalizado") 

