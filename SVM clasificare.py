import pandas
from sklearn import svm

names = ['Atrib1', 'Atrib2', 'Atrib3', 'Atrib4', 'Atrib5', 'Atrib6', 'Atrib7', 'Atrib8', 'Atrib9', 'Clasa']
dataset_train = pandas.read_csv(r"C:/Users/Andrei/Downloads/shuttle2.trn", names=names, delim_whitespace=True)

dataset_test = pandas.read_csv(r"C:/Users/Andrei/Downloads/shuttle2.tst", names=names, delim_whitespace=True)

array_train = dataset_train.values
X_train = array_train[:,:9]
Y_train = dataset_train.Clasa

array_test = dataset_test.values
X_test = array_test[:,:9]
Y_test = dataset_test.Clasa


for i in range(-5,7):
    model=svm.SVC(kernel='linear', C=pow(2,i), gamma = 1)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    acc=0
    for j in range(len(Y_test)):
        if Y_test[j]==predictions[j]:
            acc=acc+1
    print("Cost:", 2**(i))
    print(" Precizie:"+ str((acc/len(Y_test))*100) + '%')
    print("\n")

