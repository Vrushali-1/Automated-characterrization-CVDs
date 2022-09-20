from  django.http import HttpRequest
from django.http import HttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import pandas
from scipy.sparse.data import _data_matrix
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from django.conf import settings
from django.shortcuts import render
from django.templatetags.static import static
import os
from sklearn.svm import SVC

filename=""
GLOBAL_Entry = None

def index(request):
    return render(request,"index.html")
def login(request):
    return render(request,"login.html")
def logincheck(request):
    if(request.method=='POST'):
         username=request.POST['username']
         password=request.POST['password']
         user=auth.authenticate(username=username,password=password)
         if user is not None:
             auth.login(request,user)
             return render(request,"fileupload.html")
         else:
            messages.info(request,"UserName or Password is invalid")
            return render(request,"login.html")
    else:
        return render(request,"register.html")
def register(request):
    return render(request,"register.html")
def registerUser(request):
    if request.method=="POST":
        firstname=request.POST['firstname']
        lastname=request.POST['lastname']
        email=request.POST['email']
        username=request.POST['username']
        password1=request.POST['password1']
        password2=request.POST['password2']
        if(password1==password2):
            if(User.objects.filter(username=username).exists()):
                print('UserName already Exists')
                messages.info(request,'UserName already exists')
                return  render(request,"register.html")    
            elif(User.objects.filter(email=email).exists()):
                print('Email already taken')
                messages.info(request,'Email already exists')
                return  render(request,"register.html")
            else:    
                user=User.objects.create_user(username=username,password=password1,email=email,first_name=firstname,last_name=lastname)
                user.save()
                print('User Created Successfully')
                messages.info(request,'User Registered Successfully')
                return  render(request,"login.html")
        else:
            print("Password not matching")
            messages.info(request,'Password Not matching')
            return  render(request,"register.html")            
    else:
       return render(request,"login.html")
def uploadFile(request):
    if request.method=="POST":
        file1=request.FILES['file1']
        fs=FileSystemStorage()
        fs.save(file1.name,file1)
        messages.info(request,"File Uploaded Successfully")
    return render(request,"fileupload.html")
def getClassification(request):
    return render(request,"classification.html")
def classification(request):
    if request.method=="POST":
        file1=request.FILES['file1']
        global filename,GLOBAL_Entry
        filename=file1
        print(filename)
        path = "media\\"+str(filename)
        print(path)
        names = ['ML','V5']
        dataframe = pandas.read_csv(path, names=names)
        array = dataframe.values
        X = array[:,0:2]
        Y = array[:,1]
        seed = 7
        models = []
        models.append(('Random Forest ', RandomForestClassifier()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('Naive Byes', GaussianNB()))
        models.append(('SVM', SVC()))
        results = []
        names = []
        scoring = 'accuracy'
        for name,model in models:
            kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            results.append(cv_results.mean())
            names.append(name)
        for r in results:
            print(r)
        print(results[0])
        data1=str(results[0])
        data2=str(results[1])
        data3=str(results[2])
        data4=str(results[3])
        name1=names[0]
        name2=names[1]
        name3=names[2]
        name4=names[3]
        return  render(request,'classificationresults.html',{'data1':data1,'data2':data2,'data3':data3,'data4':data4,'name1':name1,'name2':name2,'name3':name3,'name4':name4})
        # return render(request,'index.html')

def nbClassification(request):
    import pandas  as pd
    from sklearn.naive_bayes import GaussianNB
    from sklearn import model_selection
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import seaborn as sns 

    path = "media\\"+str(filename)
    print(path)
    dataset = pd.read_csv(path)
    array = dataset.values
    #print(dataset.head())
    X = array[:,0:2]
    y = array[:,1]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    # print("confusion_matrix " ,cm )
    print(classification_report(y_test, y_pred))
    accuracyscore=metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:",accuracyscore)
    predictscore=y_pred.mean()
    print("Prediction ", predictscore)
    verdict=""
    if(predictscore>800 and predictscore<=850):
        verdict="Normal"
    if(predictscore>851 and predictscore<=900):
        verdict="HCM"
    if(predictscore>901 and predictscore<=950):
        verdict="HCM"
    if(predictscore>951):
        verdict="MI"
    print("Verdict ",verdict)
    return render(request,"predication.html",{'accuracyresult':accuracyscore,'predictionvalue':predictscore,'verdict':verdict})

   

