# Multi-Label Text Classification with BERT

This folder contains an app which utilises BERT (via the [simpletransformers](https://github.com/ThilinaRajapakse/simpletransformers) library) for Multi-Label Text Classification.

**Application Input**: CSV file containing input data (see below for data requirements)

**Application Output**: Evaluation metrics (see docs folder for more information)

This code base is adapted from 'Multi-Label Classification using BERT, RoBERTa, XLNet, XLM, and DistilBERT with Simple Transformers'. Blog:https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5 

## 1. Project Structure

The content of this repository is organized as follows:

* **data**: Contains the example dataset (filename: text_input.csv) used to demo the app - a processed version of the [Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) data is used for this purpose (see notebooks folder). 
* **docs**: Holds formal documentation for the application encompassing reasons for default hyperparameter and model choices
* **notebooks**: Jupyter notebooks. Contains notebook used to transform Toxic Comments data into format required for input into the text classifier app
* **src**: Folder containing the application source code
* **Dockerfile**: Text document containing all the commands required to build the image required to execute the application (inc. installation of required dependencies)
* **requirements.txt**: File containing a list of the Python packages required to be pip installed to run the application
* **setup.cfg**: Standard python configuration file e.g. pep8 configs, logging configuration. The Dockerfile builds this into the application to drive the logging configuration.

The application is built from two Docker images. A base image containing the core dependency set and a secondary layer containing the application itself. This is to support rapid rebuild and deployment of the platform where only the application code and/or input data has changed and not any of the core dependencies.

## 2. Target Environment and Dependencies

**Infrastructure Requirements**

Operating System: Linux - Ubuntu 16.04
Hardware: Volta/Turing (V/T) GPU 

Example - To demo this application, an Azure Linux Virtual Machine (NC6s_v3) containing a Tesla V100 GPU was used 

**Software Requirements**

Once you have set up your infrastructure, use the terminal (using ssh for a Virtual Machine) to execute the below commands and install the required software:

1. Docker -  Execute the below to install the latest version of Docker
	```bash
	sudo apt-get update && sudo apt-get install -y docker.io
	```
	
	Execute ```docker --version``` to check that Docker has been installed
	
2. git - Execute the below to install the latest version of git:
	```bash
	sudo apt-get install git
	```
**Input Data Structure Requirements and Considerations**

The Text Classifier app requires: 
* the first column of the dataframe to contain the text to be classified 
* any unwanted symbols/notation which results in the concatenation of words should be removed from the text column (e.g. /n representing a new line can concatenate the two words before and after the new line) 
* all columns, following the first column, to contain only the labels we will be training the model to predict 
* the app does not handle NULL values, the user must implement a chosen method for handling NULL values prior to use of the app 

 Important Considerations:
* the app can still be executed with data that has columns which contain multiple, mutually exclusive labels, however the multi-label classifier will not treat these labels as mutually exclusive 
* the text is truncated at 256 tokens (i.e. max_seq_length = 256 - parameter for multilabel classifier) to reduce training times, hence performance will be reduced if there are a large number of examples where label assignments cannot be inferred from the first 256 tokens   
* if it is important whether a value or 1 or 0 is assigned to a binary field (i.e. field with two possible values), this assignment should be carried out prior to loading data into the Text Classifier 

**Dependencies covered by Dockerfile**

An NVIDIA CUDA base image (see Dockerfile) is being used to enable GPU utilisation for model trainng.

The *requirements.txt* file is provided to define the Python dependencies. This file is used by the build process to install the dependencies into the application Docker image - see steps for build of application Docker image below.

## 3. Installation and Build

Follow the below steps to build the Docker image for the application:

1. Clone the bert-multilabel-text-classifier repository from Innersource into your working environment and authenticate using your Accenture credentials

	```bash
	git clone https://innersource.accenture.com/scm/uki_ds_sandbox/bert-multilabel-text-classifier.git
	```

2. Navigate into the bert-multilabel-text-classifier data folder

	```bash
	cd bert-multilabel-text-classifier/data
	```

3. (SKIP this step if using the demo data)
	Delete the demo data by executing ```rm -rf text_input.csv``` and replace with the data you would like to train and evaluate on.
	
	If using a Virtual Machine to run this application, and if your data is on a local machine, follow the instructions below using the command line on your local machine to securely copy the data to the Virutal Machine:
	
	a. On your local machine, navigate to directory containing the data to be copied
		
	b. Ensure that your user has write permissions on the bert-multilabel-text-classifier/data folder 
		
	c. Execute the below to securely copy the file containing the data from your local machine to the Virtual Machine 
		
	If you are using a username and password to access the virtual machine:
	```bash
	scp [data].csv [user]@hostname:/bert-multilabel-text-classifier/data/
	```
			
	If you are using a key to access the virtual machine:
	```bash
	scp -i "key.pem" [data].csv [user]@[hostname]:/bert-multilabel-text-classifier/data/
	```
			
	d. Rename the file to text_input.csv using ```mv [data].csv text_input.csv```


4. Navigate back to the bert-multilabel-text-classifier directory using ```cd ..``` and execute the below to build the application Docker image. The output will be a container named and tagged text_class:v1
	```bash
	docker build -t text_class:v1 -f Dockerfile .
	```

## 4. Execution

To run the sample application, execute the below to start and run a Docker container based on the image created in the **Installation and Build** step above:

```bash
docker run --runtime=nvidia text_class:v1
```

NOTE: the ```--runtime=nvidia``` option specifies use of the NVIDIA GPU by the application

Indicative run-times for the application, based upon size of input data, are given in the ______ file (see docs folder).

## 5. Future Areas of Investigation

* Use cases/scenarios for which other models may be more suitable (e.g. DistilBERT would be more suitable for cases where shorter training times are required)

* Investigation into more suitable parameters for specific use cases, as well as opportunities for introducing automated hyperparameter selection. Relevant hyperparametres to be considered for this are:
	- train_batch_size
    - learning_rate
    - num_train_epochs
    - max_seq_length

* Additional evaluation metrics (inc. visualisations) - e.g. ROC-AUC curves

* Extend capabilities of the application, for examples:
	- enable output of a pickled model for later use in making predictions
    - enable input of data into the app upon which to make predicitons following model training
