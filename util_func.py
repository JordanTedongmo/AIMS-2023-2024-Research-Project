#!/usr/bin/env python3
'''
T. A. Narh authored these utility functions.
Date: 13 Jun 2024
'''

import os
import cv2
import random
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import label_binarize
from itertools import cycle
from itertools import product
from matplotlib.colors import ListedColormap
from miceforest import ImputationKernel
import pickle

def createDirs(base_path='./', dirs='output_figures'):
	'''
	create a leaf directory
	
	Parameter:
	----------
		- path (str): path to dump directory
		- dirs (str): target directory
	''' 

	# Join the paths 
	full_path = os.path.join(base_path, dirs)

	# Check if the directory already exists
	if not os.path.exists(full_path):
		# If not, create the directory
		os.makedirs(full_path)
		print(f"Directory '{full_path}' created successfully.")
	else:
		print(f"Directory '{full_path}' already exists.")  

def load_random_images(directory, n=5):
	"""
	Load a random selection of images from a specified directory.

	Parameters:
	directory (str): The directory to search for image files.
	n (int): The number of random images to load. Default is 5.

	Returns:
	list: A list of images read from the selected files.
	"""
	
	# List all files in the given directory
	all_files = os.listdir(directory)
	
	# Filter the list to include only image files with specific extensions
	image_files = [f for f in all_files if f.endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp'))]
	
	# Randomly select 'n' image files from the filtered list
	selected_files = random.sample(image_files, n)
	
	# Read the selected image files and store them in a list
	images = [imread(os.path.join(directory, file)) for file in selected_files]
	
	return images 

def custom_svd(ndata):
	"""
	Perform singular value decomposition (SVD) on the given data.

	Parameters:
	ndata (numpy.ndarray): Input data of shape (batch_size, height, width, channels).

	Returns:
	tuple: Tuple containing U, S, and Vt matrices.
	"""
	reshaped_data = ndata.reshape(ndata.shape[0], -1, ndata.shape[-1])
	
	# Initialize lists to store the SVD results
	U_list = []
	S_list = []
	V_list = []
	
	for c in range(ndata.shape[-1]):
		if np.any(np.isnan(reshaped_data[..., c])):
		   
			# # Create an ImputationKernel instance
			# kernel = ImputationKernel(data=reshaped_data[...,c], save_all_iterations=True, random_state=1343)
			
			# print('Performing miceForest imputation ... \n') 
			# # Perform MICE imputation
			# kernel.mice(iterations=10, verbose=True)
			
			# # Retrieve the imputed dataset
			# data_imputed= kernel.complete_data(dataset=0, inplace=False)
			data_imputed = np.nan_to_num(reshaped_data[...,c])

		else:
			data_imputed = reshaped_data[..., c].copy()
			
		U, S, V = svd(data_imputed, full_matrices=False)
		sorted_indices = np.argsort(S)[::-1]
		S = S[sorted_indices]
		U = U[:, sorted_indices]
		V = V[sorted_indices, :]
		
		U_list.append(U)
		S_list.append(S)
		V_list.append(V)
		
	# Convert the lists to numpy arrays
	U_array = np.array(U_list)
	S_array = np.array(S_list)
	V_array = np.array(V_list)
		
	return U_array, S_array, V_array
	

def svdreconstruct(Ur, Sr, Vr, topN=5):
	"""
	Reconstruct the SVD using the top singular values and their respective U and V components.

	Parameters:
	U (numpy.ndarray): U matrix of shape (num_samples, num_samples).
	S (numpy.ndarray): S vector of singular values of shape (num_samples,).
	V (numpy.ndarray): V matrix of shape (num_features, num_samples).
	topN (int): Number of top singular values to use for reconstruction.

	Returns:
	numpy.ndarray: Reconstructed matrix of shape (num_samples, num_features).
	numpy.ndarray: Coefficients matrix used for reconstruction of shape (num_samples, num_samples).
	"""
	# Ensure topN does not exceed the length of Sr
	topN = min(topN, len(Sr))
	
	# Compute the coefficients matrix
	# print(np.diag(S))
	coeff = np.dot(Ur, np.diag(Sr))
	# print(coeff.shape)

	# Reconstruct using the top singular values and their respective U and V components
	reconstruct = np.dot(coeff[:, :topN], Vr[:topN, :])

	return reconstruct, coeff

def plot_subplotsq(original_data, model_data, residuals, set_title=['Image_0', 'Image_1', 'Image_2', 'Image_3']):
	"""
	Plots a 3x4 grid of subplots showing original data, model data, and residuals.

	Parameters:
	original_data (list of np.ndarray): List of 2D arrays representing the original data.
	model_data (list of np.ndarray): List of 2D arrays representing the model data.
	residuals (list of np.ndarray): List of 2D arrays representing the residuals.

	The function creates a figure with 3 rows and 4 columns of subplots. 
	The first row shows the original data, the second row shows the model data, 
	and the third row shows the residuals.
	"""
	
	# Create a figure with 3 rows and 4 columns of subplots
	fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True, sharex='col', sharey='row')
	plt.subplots_adjust(wspace=0, hspace=0.05, left=0.15)

	for i in range(4):
		# Plot original data
		im_orig = axes[0, i].imshow(original_data[i], cmap='hot')
		if i == 0:
			axes[0, i].set_ylabel('Original', fontsize=12)
		axes[0, i].set_title(f"{set_title[i]}")
		axes[0, i].set_yticks([])  # Remove y-ticks for original data plots

		# Plot model data
		im_model = axes[1, i].imshow(model_data[i], cmap='hot')
		if i == 0:
			axes[1, i].set_ylabel('Model', fontsize=12)
			axes[1, i].set_yticks([])  # Remove y-ticks for model data plots

		# Plot residuals
		im_res = axes[2, i].imshow(residuals[i], cmap='hot')
		axes[2, i].set_xticks([])  # Remove x-ticks for residual plots
		if i == 0:
			axes[2, i].set_ylabel('Residual', fontsize=12)
			axes[2, i].set_yticks([])  # Remove y-ticks for residual plots

	# Add colorbars for each row of subplots
	cbar_original = fig.colorbar(im_orig, ax=axes[0, :], location='right', shrink=0.9)
	cbar_original.set_label('Original Data', rotation=270, labelpad=10)

	cbar_model = fig.colorbar(im_model, ax=axes[1, :], location='right', shrink=0.9)
	cbar_model.set_label('Model Data', rotation=270, labelpad=10)

	cbar_residual = fig.colorbar(im_res, ax=axes[2, :], location='right', shrink=0.9)
	cbar_residual.set_label('Residual', rotation=270, labelpad=10)

	# Show the final plot
	plt.show()

def resize_data(data, new_size=(256, 256)):
	"""
	Resize the data to the specified new size.
	"""
	resized_data = np.zeros((data.shape[0], new_size[0], new_size[1], data.shape[3]))
	for i in range(data.shape[0]):
		for j in range(data.shape[3]):
			resized_data[i, :, :, j] = cv2.resize(data[i, :, :, j], new_size)
	return resized_data

def normalize_data(data):
	"""
	Normalize the data to be within the range [0, 1].
	"""
	data_min = np.min(data)
	data_max = np.max(data)
	if data_max > data_min:  # Prevent division by zero
		return (data - data_min) / (data_max - data_min)
	else:
		return np.zeros_like(data)

def load_resize_normalize_images(directory, target_shape=(128, 128), n=5):
	"""
	Function to load, resize, and normalize images from a directory
	"""
	all_files = os.listdir(directory)
	image_files = [f for f in all_files if f.endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp'))]
	selected_files = random.sample(image_files, n)
	images = []
	for file in selected_files:
		
		img = cv2.imread(os.path.join(directory, file))
		img_resized = cv2.resize(img, target_shape)
		img_normalized = normalize_data(img_resized)
		images.append(img_normalized)
		
	return np.array(images)  



class ArrayProcessor:
	def __init__(self, file_path):
		"""
		Initialize the ArrayProcessor object with a file path.

		Args:
		- file_path (str): Path to the file where arrays will be saved and loaded.
		"""
		self.file_path = file_path

	def save_arrays(self, arrays_dict):
		"""
		Save a dictionary of arrays to the specified file using pickle serialization.

		Args:
		- arrays_dict (dict): Dictionary where keys are array names and values are arrays (lists, numpy arrays, etc.).
		"""
		with open(self.file_path, 'wb') as file:
			pickle.dump(arrays_dict, file)
		print(f"Arrays saved to {self.file_path}")

	def load_arrays(self):
		"""
		Load arrays from the specified file using pickle deserialization.

		Returns:
		- arrays_dict (dict or None): Dictionary containing loaded arrays, or None if file not found.
		"""
		try:
			with open(self.file_path, 'rb') as file:
				arrays_dict = pickle.load(file)
			print(f"Arrays loaded from {self.file_path}")
			return arrays_dict
		except FileNotFoundError:
			print(f"File {self.file_path} not found.")
			return None


class ModelComparison:
	# Usage
	# df = pd.read_csv('your_dataset.csv')  # Assuming df is already defined
	# model_comp = ModelComparison(df, target='target')
	# model_comp.train_models()
	# model_comp.train_ensemble_models()
	# model_comp.plot_confusion_matrices()
	# model_comp.plot_classification_reports()
	# model_comp.plot_roc_curves()
	# model_comp.plot_decision_boundaries()

    def __init__(self, df, target, test_size=0.2, random_state=42, scaling_method='standard'):
        self.df = df
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaling_method = scaling_method
        
        # Separate features and target
        self.X = df.drop(target, axis=1)
        self.y = df[target]
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Standardize the features
        self.scaler = self._get_scaler(scaling_method)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'LightGBM': LGBMClassifier(),
            'XGBoost': XGBClassifier(),
            'SVM': SVC(probability=True),
            'Naive Bayes': GaussianNB(),
            'MLP': MLPClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gaussian Process': GaussianProcessClassifier()
        }
        self.ensemble_models = {}
        self.results = {}
        
    def _get_scaler(self, method):
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileTransformer(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(method, StandardScaler())

    def train_models(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'report': classification_report(self.y_test, y_pred, output_dict=True)
            }

    def train_ensemble_models(self):
        # Bagging
        bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=10, random_state=42)
        bagging_model.fit(self.X_train, self.y_train)
        y_pred_bagging = bagging_model.predict(self.X_test)
        self.ensemble_models['Bagging'] = {
            'model': bagging_model,
            'y_pred': y_pred_bagging,
            'report': classification_report(self.y_test, y_pred_bagging, output_dict=True)
        }

        
        # Boosting
        boosting_model = GradientBoostingClassifier()
        boosting_model.fit(self.X_train, self.y_train)
        y_pred_boosting = boosting_model.predict(self.X_test)
        self.ensemble_models['Boosting'] = {
            'model': boosting_model,
            'y_pred': y_pred_boosting,
            'report': classification_report(self.y_test, y_pred_boosting, output_dict=True)
        }

        # Stacking
        stacking_model = StackingClassifier(
            estimators=[
                ('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('svm', SVC(probability=True))
            ],
            final_estimator=LogisticRegression()
        )
        stacking_model.fit(self.X_train, self.y_train)
        y_pred_stacking = stacking_model.predict(self.X_test)
        self.ensemble_models['Stacking'] = {
            'model': stacking_model,
            'y_pred': y_pred_stacking,
            'report': classification_report(self.y_test, y_pred_stacking, output_dict=True)
        }

        # Average Voting
        voting_model = VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier()),
            ('svm', SVC(probability=True)),
            ('mlp', MLPClassifier()),
            ('lgbm', LGBMClassifier()),
            ('xgb', XGBClassifier())
        ], voting='soft')
        voting_model.fit(self.X_train, self.y_train)
        y_pred_voting = voting_model.predict(self.X_test)
        self.ensemble_models['Voting'] = {
            'model': voting_model,
            'y_pred': y_pred_voting,
            'report': classification_report(self.y_test, y_pred_voting, output_dict=True)
        }

    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate({**self.results, **self.ensemble_models}.items()):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt="d", ax=axes[idx], cmap="Blues")
            axes[idx].set_title(name)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()

    
    def get_confusion_matrices(self):
        Models_cm = {}

        for idx, (name, result) in enumerate({**self.results, **self.ensemble_models}.items()):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            Models_cm [name] = cm 
        return Models_cm 

    def plot_classification_reports(self):
        reports = {**self.results, **self.ensemble_models}
        for name, result in reports.items():
            print(f"Model: {name}")
            print(classification_report(self.y_test, result['y_pred']))
            print("-" * 60)


    def plot_roc_curves(self):
        plt.figure(figsize=(15, 10))
        
        if len(np.unique(self.y_test)) > 2:  # Multiclass case
            y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
            n_classes = y_test_bin.shape[1]
            
            # Compute macro-average ROC curve and AUC score
            fpr_macro = dict()
            tpr_macro = dict()
            roc_auc_macro = dict()
            
            for name, result in {**self.results, **self.ensemble_models}.items():
                model = result['model']
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self.X_test)
                else:
                    y_proba = model.decision_function(self.X_test)
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Aggregate micro-average ROC curve and AUC score
                fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                
                # Plot macro-average ROC curve
                fpr_macro[name] = fpr_micro
                tpr_macro[name] = tpr_micro
                roc_auc_macro[name] = roc_auc_micro
            
            for name in fpr_macro:
                plt.plot(fpr_macro[name], tpr_macro[name], lw=2, label=f'{name} (micro-average area = {roc_auc_macro[name]:.2f})', linestyle='--')
    
        else:  # Binary case
            for name, result in {**self.results, **self.ensemble_models}.items():
                model = result['model']
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                else:
                    y_proba = model.decision_function(self.X_test)
                
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        
    def plot_decision_boundaries(self):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
        axes = axes.ravel()
        
        X_train_reduced = self.X_train[:, :2]
        X_test_reduced = self.X_test[:, :2]
        
        cmap = ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        num_classes = len(np.unique(self.y_test))
        
        for idx, (name, result) in enumerate({**self.results, **self.ensemble_models}.items()):
            model = result['model']
            model.fit(X_train_reduced, self.y_train)
            
            x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
            y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
            
            # Plot scatter plot with four classes and legends
            sc = axes[idx].scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=self.y_test, s=50, edgecolor='k', cmap=cmap)
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) for i in range(num_classes)]
            axes[idx].legend(handles, [f'Class {i}' for i in range(num_classes)], loc='upper right')
            
            axes[idx].set_title(name)
            axes[idx].set_xlabel('Feature 1')
            axes[idx].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()