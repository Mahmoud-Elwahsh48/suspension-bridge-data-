import sys
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import gdown
import os
import logging

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QLabel, QPushButton, QFrame, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Suppress TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Constants
THRESHOLD_FILE = "thresholds.json"
DATA_FILE_ID = "1g9NCYEwmNzNHM_zTBKdJC2UFmD6UkidK"
LOCAL_DATA_PATH = 'Bridge data.csv'

class BridgeHealthApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bridge Health Monitoring System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize data and models
        self.data = None
        self.thresholds = None
        self.models = None
        
        # Load data and models
        self.load_data()
        self.load_models()
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create header
        self.create_header()
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_health_trend_tab()
        self.create_risk_indicators_tab()
        self.create_risk_distribution_tab()
        self.create_feature_correlations_tab()
        self.create_prediction_tab()
        
        # Create footer
        self.create_footer()
        
        # Set up auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(300000)  # 5 minutes in milliseconds
        
    def load_data(self):
        """Load and prepare the bridge data from Google Drive"""
        try:
            # Download the latest data from Google Drive
            url = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
            gdown.download(url, LOCAL_DATA_PATH, quiet=True)
            
            # Read the data
            data = pd.read_csv(LOCAL_DATA_PATH)
            data = data[~data['Date'].isna()]
            
            # Calculate corrosion
            def calculate_corrosion(t, T, RH, TOW, Precip, pH, Cl, material_type='steel'):
                if material_type == 'steel':
                    A_coeff = np.random.normal(85, 10)
                    n = np.random.normal(0.35, 0.05)
                else:
                    A_coeff = np.random.normal(65, 8)
                    n = np.random.normal(0.28, 0.03)

                env_factor = (
                    0.15 * T + 0.12 * RH + 0.10 * TOW + 
                    0.05 * Precip/1000.0 + 0.18 * (7 - pH) + 
                    0.30 * np.log(Cl + 1)
                )
                A = A_coeff * (1 + env_factor)
                C = A * (t ** n)
                return C/1000, A/1000

            # Apply corrosion calculation
            results = data.apply(lambda row: calculate_corrosion(
                t=row['BridgeAge'] * 0.8,
                T=row['Average_Temperature'],
                RH=row['Average_Humidity'],
                TOW=row['Mean_TOW'],
                Precip=row['TotalPrecipitation'],
                pH=row['Mean_PH'],
                Cl=row['Mean_Cl-'],
                material_type='steel'
            ), axis=1)
            
            data['CorrosionDepth'], data['AnnualRate'] = zip(*results)
            
            # Calculate additional metrics
            data['Wind-Induced Vibrations'] = data['Mean_Wind_Speed'] * np.sin(np.radians(data['Wind_Direction']))
            data['Traffic Load'] = data['AverageDailyTraffic'] * data['Structure_Length']
            
            # Calculate Structural Health
            data['Structural Health'] = 100 - (
                (data['BridgeAge'] * 0.1) +
                (data['Wind-Induced Vibrations'] * 0.2) +
                (data['AnnualRate'] * 0.3) +
                ((data['Traffic Load'] / 1e6) * 0.4))
            
            # Load dynamic thresholds
            self.load_thresholds()
            if not self.thresholds:
                raise Exception("Dynamic thresholds not found")
                
            # Calculate risk categories using dynamic thresholds
            data['Vibration Risk'] = data['Wind-Induced Vibrations'].apply(
                lambda x: self.categorize_risk(x, 'Wind_Induced_Vibrations', self.thresholds['Wind_Induced_Vibrations Risk']))
            data['Corrosion Risk'] = data['AnnualRate'].apply(
                lambda x: self.categorize_risk(x, 'Corrosion', self.thresholds['Corrosion Risk']))
            data['Traffic Load Risk'] = data['Traffic Load'].apply(
                lambda x: self.categorize_risk(x, 'Traffic_Load', self.thresholds['Traffic_Load Risk']))
            data['Structural Health Risk'] = data['Structural Health'].apply(
                lambda x: self.categorize_risk(x, 'Structural_Health', self.thresholds['Structural_Health Risk']))
            
            self.data = data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create empty DataFrame if loading fails
            self.data = pd.DataFrame()
    
    def load_thresholds(self):
        """Load dynamic thresholds from file"""
        if os.path.exists(THRESHOLD_FILE):
            with open(THRESHOLD_FILE, "r") as f:
                self.thresholds = json.load(f)
    
    def load_models(self):
        """Load all required models and preprocessing objects"""
        try:
            self.models = {
                'classification_model': tf.keras.models.load_model('classification_model_dynamic.h5'),
                'regression_model': tf.keras.models.load_model('regression_model_dynamic.h5'),
                'scaler': joblib.load('scaler.pkl'),
                'regression_scaler': joblib.load('regression_scaler.pkl'),
                'label_encoders': joblib.load('label_encoders.pkl')
            }
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = None
    
    def categorize_risk(self, value, risk_type, thresholds):
        """Categorize risk based on dynamic threshold values"""
        if risk_type == 'Structural_Health':
            # For structural health, higher is better (reverse order)
            if value >= thresholds[2]:
                return 3  # Good
            elif value >= thresholds[1]:
                return 2  # Fair
            elif value >= thresholds[0]:
                return 1  # Poor
            else:
                return 0  # Critical
        else:
            # For other risks, higher is worse
            if value < thresholds[0]:
                return 0  # Low
            elif value < thresholds[1]:
                return 1  # Moderate
            elif value < thresholds[2]:
                return 2  # High
            else:
                return 3  # Critical
    
    def get_risk_labels_and_colors(self, risk_value, risk_type):
        """Return appropriate label and color based on risk type and value"""
        if risk_type == 'Structural Health':
            labels = ['Critical', 'Poor', 'Fair', 'Good']
            colors = ['#FF0000', '#FFA500', '#1E90FF', '#32CD32']  # Red, Orange, Blue, Green
        else:
            labels = ['Low', 'Moderate', 'High', 'Critical']
            colors = ['#32CD32', '#1E90FF', '#FFA500', '#FF0000']  # Green, Blue, Orange, Red
        
        return labels[risk_value], colors[risk_value]
    
    def create_header(self):
        """Create the application header"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        
        title = QLabel("🏗️ Bridge Health Monitoring System")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        
        refresh_btn = QPushButton("Refresh Data")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self.refresh_data)
        
        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(refresh_btn)
        
        self.main_layout.addWidget(header)
    
    def create_footer(self):
        """Create the application footer"""
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        
        last_update = QLabel(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        last_update.setFont(QFont('Arial', 8))
        
        footer_layout.addWidget(last_update)
        footer_layout.addStretch()
        
        self.main_layout.addWidget(footer)
    
    def create_dashboard_tab(self):
        """Create the dashboard tab with key metrics"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Key metrics section
        metrics_section = QWidget()
        metrics_layout = QHBoxLayout(metrics_section)
        
        # Add metrics cards
        if not self.data.empty:
            last_row = self.data.iloc[-1]
            
            # Structural Health
            sh_value = f"{last_row['Structural Health']:.1f}"
            sh_risk_value = last_row['Structural Health Risk']
            sh_label, sh_color = self.get_risk_labels_and_colors(sh_risk_value, 'Structural Health')
            metrics_layout.addWidget(self.create_metric_card("Structural Health", sh_value, sh_label, sh_color))
            
            # Corrosion Risk
            cr_value = f"{last_row['AnnualRate']:.2f} mm/yr"
            cr_risk_value = last_row['Corrosion Risk']
            cr_label, cr_color = self.get_risk_labels_and_colors(cr_risk_value, 'Corrosion')
            metrics_layout.addWidget(self.create_metric_card("Corrosion Risk", cr_value, cr_label, cr_color))
            
            # Traffic Load Risk
            tl_value = f"{last_row['Traffic Load']/1e6:.2f}M"
            tl_risk_value = last_row['Traffic Load Risk']
            tl_label, tl_color = self.get_risk_labels_and_colors(tl_risk_value, 'Traffic Load')
            metrics_layout.addWidget(self.create_metric_card("Traffic Load Risk", tl_value, tl_label, tl_color))
            
            # Vibration Risk
            vr_value = f"{last_row['Wind-Induced Vibrations']:.2f}"
            vr_risk_value = last_row['Vibration Risk']
            vr_label, vr_color = self.get_risk_labels_and_colors(vr_risk_value, 'Vibration')
            metrics_layout.addWidget(self.create_metric_card("Vibration Risk", vr_value, vr_label, vr_color))
        
        scroll_layout.addWidget(metrics_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "🏠 Dashboard")
    
    def create_health_trend_tab(self):
        """Create a dedicated tab for Structural Health Trend"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Structural Health Trend
        trend_section = QWidget()
        trend_layout = QVBoxLayout(trend_section)
        
        trend_label = QLabel("Structural Health Trend Over Time")
        trend_label.setFont(QFont('Arial', 12, QFont.Bold))
        trend_layout.addWidget(trend_label)
        
        if not self.data.empty:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plot the structural health data
            ax.plot(self.data['Structural Health'], color='blue', linewidth=2)
            ax.set_title("Structural Health Trend", fontsize=14)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Structural Health Index", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add threshold lines
            if self.thresholds:
                ax.axhline(y=self.thresholds['Structural_Health Risk'][0], color='red', linestyle='--', 
                          label='Critical/Poor Boundary')
                ax.axhline(y=self.thresholds['Structural_Health Risk'][1], color='orange', linestyle='--', 
                          label='Poor/Fair Boundary')
                ax.axhline(y=self.thresholds['Structural_Health Risk'][2], color='green', linestyle='--', 
                          label='Fair/Good Boundary')
                ax.legend()
            
            canvas = FigureCanvas(fig)
            trend_layout.addWidget(canvas)
        
        scroll_layout.addWidget(trend_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "📉 Health Trend")
    
    def create_risk_indicators_tab(self):
        """Create a compact risk indicators visualization with prominent threshold lines"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Risk Indicators section
        risk_section = QWidget()
        risk_layout = QVBoxLayout(risk_section)
        
        risk_label = QLabel("Risk Indicators with Dynamic Thresholds")
        risk_label.setFont(QFont('Arial', 12, QFont.Bold))
        risk_layout.addWidget(risk_label)
        
        if not self.data.empty and self.thresholds:
            fig = Figure(figsize=(8, 4))  # Compact figure size
            ax = fig.add_subplot(111)
            
            # Plot original data with thinner lines
            line_width = 1.2
            ax.plot(self.data['AnnualRate'], label='Corrosion (mm/yr)', 
                    color='red', linewidth=line_width, alpha=0.8)
            ax.plot(self.data['Wind-Induced Vibrations'], label='Vibrations', 
                    color='blue', linewidth=line_width, alpha=0.8)
            ax.plot(self.data['Traffic Load']/1e6, label='Traffic (millions)', 
                    color='green', linewidth=line_width, alpha=0.8)
            
            # Add threshold zones with transparency
            self.add_threshold_zones(ax, 'Corrosion', 'red')
            self.add_threshold_zones(ax, 'Vibration', 'blue')
            self.add_threshold_zones(ax, 'Traffic', 'green')
            
            # Styling
            ax.set_title("Risk Indicators with Threshold Zones", fontsize=10)
            ax.set_xlabel("Time", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Compact legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=3, fontsize=8, framealpha=0.5)
            
            # Tight layout to minimize empty space
            fig.tight_layout()
            
            canvas = FigureCanvas(fig)
            risk_layout.addWidget(canvas)
        
        scroll_layout.addWidget(risk_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "⚠️ Risk Indicators")

    def add_threshold_zones(self, ax, risk_type, color):
        """Add colored threshold zones to the plot"""
        if risk_type == 'Corrosion':
            thresholds = self.thresholds['Corrosion Risk']
            label_prefix = 'Corrosion'
        elif risk_type == 'Vibration':
            thresholds = self.thresholds['Wind_Induced_Vibrations Risk']
            label_prefix = 'Vibration'
        elif risk_type == 'Traffic':
            thresholds = [x/1e6 for x in self.thresholds['Traffic_Load Risk']]
            label_prefix = 'Traffic'
        else:
            return
        
        # Add colored zones between thresholds
        alpha_values = [0.1, 0.2, 0.3]  # Increasing opacity for higher risk
        zone_labels = ['Low', 'Moderate', 'High', 'Critical']
        
        # Fill between zones
        y_min, y_max = ax.get_ylim()
        
        # Low risk zone (below first threshold)
        ax.axhspan(ymin=y_min, ymax=thresholds[0], 
                   facecolor=color, alpha=alpha_values[0], 
                   label=f'{label_prefix} {zone_labels[0]}')
        
        # Moderate risk zone
        ax.axhspan(ymin=thresholds[0], ymax=thresholds[1], 
                   facecolor=color, alpha=alpha_values[1],
                   label=f'{label_prefix} {zone_labels[1]}')
        
        # High risk zone
        ax.axhspan(ymin=thresholds[1], ymax=thresholds[2], 
                   facecolor=color, alpha=alpha_values[2],
                   label=f'{label_prefix} {zone_labels[2]}')
        
        # Critical risk zone
        ax.axhspan(ymin=thresholds[2], ymax=y_max, 
                   facecolor=color, alpha=0.4,
                   label=f'{label_prefix} {zone_labels[3]}')
        
        # Add threshold lines
        line_styles = [':', '--', '-']
        for i, threshold in enumerate(thresholds):
            ax.axhline(y=threshold, color=color, 
                      linestyle=line_styles[i], 
                      linewidth=1.5, alpha=0.8)
        
    def create_risk_distribution_tab(self):
        """Create a dedicated tab for Risk Category Distribution"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Risk Distribution section
        risk_dist_section = QWidget()
        risk_dist_layout = QVBoxLayout(risk_dist_section)
        
        risk_dist_label = QLabel("Risk Category Distribution")
        risk_dist_label.setFont(QFont('Arial', 12, QFont.Bold))
        risk_dist_layout.addWidget(risk_dist_label)
        
        if not self.data.empty:
            # Create a grid for pie charts
            pie_grid = QWidget()
            pie_grid_layout = QHBoxLayout(pie_grid)
            
            # Vibration Risk pie chart
            vibration_fig = self.create_pie_chart(
                self.data, 'Vibration Risk', 
                ['Low', 'Moderate', 'High', 'Critical'],
                ['#32CD32', '#1E90FF', '#FFA500', '#FF0000'],
                "Vibration Risk Distribution"
            )
            pie_grid_layout.addWidget(vibration_fig)
            
            # Corrosion Risk pie chart
            corrosion_fig = self.create_pie_chart(
                self.data, 'Corrosion Risk', 
                ['Low', 'Moderate', 'High', 'Critical'],
                ['#32CD32', '#1E90FF', '#FFA500', '#FF0000'],
                "Corrosion Risk Distribution"
            )
            pie_grid_layout.addWidget(corrosion_fig)
            
            risk_dist_layout.addWidget(pie_grid)
            
            # Second row of pie charts
            pie_grid2 = QWidget()
            pie_grid_layout2 = QHBoxLayout(pie_grid2)
            
            # Traffic Load Risk pie chart
            traffic_fig = self.create_pie_chart(
                self.data, 'Traffic Load Risk', 
                ['Low', 'Moderate', 'High', 'Critical'],
                ['#32CD32', '#1E90FF', '#FFA500', '#FF0000'],
                "Traffic Load Risk Distribution"
            )
            pie_grid_layout2.addWidget(traffic_fig)
            
            # Structural Health pie chart
            health_fig = self.create_pie_chart(
                self.data, 'Structural Health Risk', 
                ['Critical', 'Poor', 'Fair', 'Good'],
                ['#FF0000', '#FFA500', '#1E90FF', '#32CD32'],
                "Structural Health Distribution"
            )
            pie_grid_layout2.addWidget(health_fig)
            
            risk_dist_layout.addWidget(pie_grid2)
        
        scroll_layout.addWidget(risk_dist_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "📊 Risk Distribution")
    
    def create_feature_correlations_tab(self):
        """Create a dedicated tab for Feature Correlations"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Feature Correlations section
        corr_section = QWidget()
        corr_layout = QVBoxLayout(corr_section)
        
        corr_label = QLabel("Feature Correlations")
        corr_label.setFont(QFont('Arial', 12, QFont.Bold))
        corr_layout.addWidget(corr_label)
        
        if not self.data.empty:
            numeric_cols = self.data.select_dtypes(include=np.number).columns
            corr_matrix = self.data[numeric_cols].corr()
            
            fig = Figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            
            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list(
                'RdBu', ['#FF0000', '#FFFFFF', '#0000FF'])
            
            sns.heatmap(corr_matrix, ax=ax, cmap=cmap, center=0, 
                        annot=True, fmt=".2f", linewidths=.5)
            ax.set_title("Feature Correlation Matrix", fontsize=14)
            
            canvas = FigureCanvas(fig)
            corr_layout.addWidget(canvas)
        
        scroll_layout.addWidget(corr_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "🔗 Feature Correlations")
    
    def create_metric_card(self, title, value, risk_label, color):
        """Create a metric card widget"""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setLineWidth(1)
        card.setFixedHeight(120)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setFont(QFont('Arial', 10, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setFont(QFont('Arial', 14, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        
        risk_label = QLabel(risk_label)
        risk_label.setFont(QFont('Arial', 10, QFont.Bold))
        risk_label.setAlignment(Qt.AlignCenter)
        risk_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(risk_label)
        
        return card
    
    def create_pie_chart(self, data, column, labels, colors, title):
        """Create a pie chart widget"""
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        
        counts = data[column].value_counts().reindex(range(4), fill_value=0)
        ax.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(title)
        ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        
        canvas = FigureCanvas(fig)
        return canvas
    
    def create_prediction_tab(self):
        """Create the prediction tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Prediction section
        pred_section = QWidget()
        pred_layout = QVBoxLayout(pred_section)
        
        pred_label = QLabel("Bridge Health Prediction")
        pred_label.setFont(QFont('Arial', 12, QFont.Bold))
        pred_layout.addWidget(pred_label)
        
        if not self.data.empty:
            last_row = self.data.iloc[-1]
            last_date = pd.to_datetime(last_row['Date']).strftime('%Y-%m-%d %H:%M:%S')
            
            timestamp_label = QLabel(f"Data timestamp: {last_date}")
            timestamp_label.setFont(QFont('Arial', 10))
            pred_layout.addWidget(timestamp_label)
            
            predict_btn = QPushButton("Predict Current Status")
            predict_btn.clicked.connect(self.run_prediction)
            pred_layout.addWidget(predict_btn)
            
            # Add prediction results placeholder
            self.prediction_results = QWidget()
            self.prediction_results.setVisible(False)
            pred_layout.addWidget(self.prediction_results)
        
        scroll_layout.addWidget(pred_section)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "🔍 Status Prediction")
    
    def run_prediction(self):
        """Run the prediction and display results"""
        if self.data.empty or not self.models:
            return
        
        last_row = self.data.iloc[-1]
        
        # Prepare for prediction
        features_scaled, features_reg_scaled = self.prepare_input_for_models(
            pd.DataFrame([last_row]), 
            self.models['scaler'], 
            self.models['regression_scaler']
        )
        
        try:
            # Make predictions
            class_pred = self.models['classification_model'].predict(features_scaled)
            degradation_pred = self.models['regression_model'].predict(features_reg_scaled).flatten()
            
            current_degradation = degradation_pred[0]
            pred_health = 100 - current_degradation
            
            # Get predicted risk categories
            pred_vibration_value = np.argmax(class_pred[0][0])
            pred_corrosion_value = np.argmax(class_pred[1][0])
            pred_traffic_value = np.argmax(class_pred[2][0])
            pred_health_value = self.categorize_risk(pred_health, 'Structural_Health', 
                                                   self.thresholds['Structural_Health Risk'])
            
            # Get labels and colors
            pred_vibration_label, pred_vibration_color = self.get_risk_labels_and_colors(
                pred_vibration_value, 'Vibration')
            pred_corrosion_label, pred_corrosion_color = self.get_risk_labels_and_colors(
                pred_corrosion_value, 'Corrosion')
            pred_traffic_label, pred_traffic_color = self.get_risk_labels_and_colors(
                pred_traffic_value, 'Traffic Load')
            pred_health_label, pred_health_color = self.get_risk_labels_and_colors(
                pred_health_value, 'Structural Health')
            
            # Get current risk labels
            current_vibration_label, current_vibration_color = self.get_risk_labels_and_colors(
                last_row['Vibration Risk'], 'Vibration')
            current_corrosion_label, current_corrosion_color = self.get_risk_labels_and_colors(
                last_row['Corrosion Risk'], 'Corrosion')
            current_traffic_label, current_traffic_color = self.get_risk_labels_and_colors(
                last_row['Traffic Load Risk'], 'Traffic Load')
            current_health_label, current_health_color = self.get_risk_labels_and_colors(
                last_row['Structural Health Risk'], 'Structural Health')
            
            # Create results display
            results_layout = QVBoxLayout(self.prediction_results)
            
            # Add title
            results_title = QLabel("📊 Current Status vs Predictions")
            results_title.setFont(QFont('Arial', 12, QFont.Bold))
            results_layout.addWidget(results_title)
            
            # Create cards grid
            cards_grid = QWidget()
            grid_layout = QHBoxLayout(cards_grid)
            
            # Structural Health Card
            sh_card = self.create_prediction_card(
                "Structural Health", "#1f77b4",
                f"{pred_health:.1f}", "(Predicted)",
                f"{last_row['Structural Health']:.1f}", "(Actual)",
                pred_health_label, pred_health_color,
                current_health_label, current_health_color,
                pred_health - last_row['Structural Health']
            )
            grid_layout.addWidget(sh_card)
            
            # Corrosion Card
            cr_card = self.create_prediction_card(
                "Corrosion", "#ff7f0e",
                pred_corrosion_label, "(Predicted)",
                f"{last_row['AnnualRate']:.4f} mm/yr", "(Actual)",
                pred_corrosion_label, pred_corrosion_color,
                current_corrosion_label, current_corrosion_color
            )
            grid_layout.addWidget(cr_card)
            
            # Traffic Load Card
            tl_card = self.create_prediction_card(
                "Traffic Load", "#2ca02c",
                pred_traffic_label, "(Predicted)",
                f"{last_row['Traffic Load']/1e6:.2f}M", "(Actual)",
                pred_traffic_label, pred_traffic_color,
                current_traffic_label, current_traffic_color
            )
            grid_layout.addWidget(tl_card)
            
            # Vibration Card
            vib_card = self.create_prediction_card(
                "Vibration", "#d62728",
                pred_vibration_label, "(Predicted)",
                f"{last_row['Wind-Induced Vibrations']:.2f}", "(Actual)",
                pred_vibration_label, pred_vibration_color,
                current_vibration_label, current_vibration_color
            )
            grid_layout.addWidget(vib_card)
            
            results_layout.addWidget(cards_grid)
            self.prediction_results.setVisible(True)
            
        except Exception as e:
            print(f"Prediction error: {e}")
    
    def create_prediction_card(self, title, title_color, 
                             pred_value, pred_label, 
                             actual_value, actual_label,
                             pred_risk_label, pred_risk_color,
                             current_risk_label, current_risk_color,
                             delta=None):
        """Create a prediction card widget"""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setLineWidth(1)
        card.setFixedHeight(180)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_lbl = QLabel(title)
        title_lbl.setFont(QFont('Arial', 10, QFont.Bold))
        title_lbl.setStyleSheet(f"color: {title_color};")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        
        # Predicted value
        pred_value_lbl = QLabel(f"{pred_value} {pred_label}")
        pred_value_lbl.setFont(QFont('Arial', 12, QFont.Bold))
        pred_value_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(pred_value_lbl)
        
        # Actual value
        actual_value_lbl = QLabel(f"{actual_value} {actual_label}")
        actual_value_lbl.setFont(QFont('Arial', 10))
        actual_value_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(actual_value_lbl)
        
        # Predicted risk
        pred_risk_lbl = QLabel(f"Predicted Risk: {pred_risk_label}")
        pred_risk_lbl.setFont(QFont('Arial', 9))
        pred_risk_lbl.setAlignment(Qt.AlignCenter)
        pred_risk_lbl.setStyleSheet(f"color: {pred_risk_color};")
        layout.addWidget(pred_risk_lbl)
        
        # Current risk
        current_risk_lbl = QLabel(f"Current Risk: {current_risk_label}")
        current_risk_lbl.setFont(QFont('Arial', 9))
        current_risk_lbl.setAlignment(Qt.AlignCenter)
        current_risk_lbl.setStyleSheet(f"color: {current_risk_color};")
        layout.addWidget(current_risk_lbl)
        
        # Delta (if provided)
        if delta is not None:
            delta_color = "green" if delta >= 0 else "red"
            delta_lbl = QLabel(f"Δ {delta:+.1f}")
            delta_lbl.setFont(QFont('Arial', 10, QFont.Bold))
            delta_lbl.setAlignment(Qt.AlignCenter)
            delta_lbl.setStyleSheet(f"color: {delta_color};")
            layout.addWidget(delta_lbl)
        
        return card
    
    def prepare_input_for_models(self, data, scaler, regression_scaler):
        """Prepare data for model prediction with all expected features"""
        features = data.copy()

        # Ensure we have all expected columns
        expected_features = scaler.feature_names_in_
        missing_features = set(expected_features) - set(features.columns)

        # Add missing features with default value 0
        for feature in missing_features:
            features[feature] = 0

        # Reorder columns to match training order
        features = features[expected_features]

        # Scale features for classification
        features_scaled = scaler.transform(features)

        # Prepare for regression (may need different features)
        expected_reg_features = regression_scaler.feature_names_in_
        missing_reg_features = set(expected_reg_features) - set(features.columns)

        for feature in missing_reg_features:
            features[feature] = 0

        features_reg = features[expected_reg_features]
        features_reg_scaled = regression_scaler.transform(features_reg)

        return features_scaled, features_reg_scaled
    
    def refresh_data(self):
        """Refresh the data and update the UI"""
        self.load_data()
        
        # Update footer timestamp
        footer = self.main_layout.itemAt(self.main_layout.count()-1).widget()
        footer.findChild(QLabel).setText(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Recreate tabs to reflect new data
        current_tab = self.tabs.currentIndex()
        self.tabs.clear()
        self.create_dashboard_tab()
        self.create_health_trend_tab()
        self.create_risk_indicators_tab()
        self.create_risk_distribution_tab()
        self.create_feature_correlations_tab()
        self.create_prediction_tab()
        self.tabs.setCurrentIndex(current_tab)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BridgeHealthApp()
    window.show()
    sys.exit(app.exec_())
