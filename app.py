"""
AgroSmart ML-Driven Crop Recommender System
Full-Fledged Flask Application for Internal Server Deployment
"""

# =====================================================
# IMPORTS
# =====================================================
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import joblib
import numpy as np
import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-GUI backend


# =====================================================
# FLASK APP INITIALIZATION
# =====================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(BASE_PATH, 'static'),
            static_url_path='/static',
            template_folder=os.path.join(BASE_PATH, 'templates'))

app.secret_key = 'agrismart_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# =====================================================
# LOAD ML MODELS (WITH ERROR HANDLING)
# =====================================================

# Fertilizer Recommendation Model
fertilizer_model = None
le_soil = None
le_crop = None
le_fertilizer = None

try:
    fertilizer_model = joblib.load(os.path.join(BASE_PATH, 'model', 'fertilizer_app.pkl'))
    fertilizer_data = pd.read_csv(os.path.join(BASE_PATH, 'data', 'fertilizer_recommendation.csv'))
    
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fertilizer = LabelEncoder()
    
    fertilizer_data['Soil Type'] = le_soil.fit_transform(fertilizer_data['Soil Type'])
    fertilizer_data['Crop Type'] = le_crop.fit_transform(fertilizer_data['Crop Type'])
    fertilizer_data['Fertilizer Name'] = le_fertilizer.fit_transform(fertilizer_data['Fertilizer Name'])
    
    print("[OK] Fertilizer recommendation model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading fertilizer model: {e}")
    fertilizer_model = None


# Crop Recommendation Model
crop_model = None
try:
    crop_model = joblib.load(os.path.join(BASE_PATH, 'model', 'crop_app.pkl'))
    print("[OK] Crop recommendation model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading crop model: {e}")
    crop_model = None


# Yield Prediction Models
yield_model = None
yield_preprocessor = None
try:
    yield_model = pickle.load(open(os.path.join(BASE_PATH, 'model', 'dtr.pkl'), 'rb'))
    yield_preprocessor = pickle.load(open(os.path.join(BASE_PATH, 'model', 'preprocesser.pkl'), 'rb'))
    print("[OK] Yield prediction model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading yield prediction model: {e}")
    yield_model = None


# =====================================================
# LOAD DATA FILES (WITH ERROR HANDLING)
# =====================================================

analysis_df = None
yield_df = None

try:
    analysis_df = pd.read_csv(os.path.join(BASE_PATH, 'data', 'analysis1_data.csv'))
    print("[OK] Analysis data loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading analysis data: {e}")

try:
    yield_df = pd.read_csv(os.path.join(BASE_PATH, 'data', 'yield_df.csv'))
    print("[OK] Yield data loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading yield data: {e}")


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def generate_chart(chart_type, data_x, data_y, title, xlabel, ylabel):
    """Generate matplotlib chart and return as base64"""
    try:
        plt.figure(figsize=(10, 6))
        if chart_type == 'bar':
            plt.bar(data_x, data_y, color='steelblue')
        elif chart_type == 'pie':
            plt.pie(data_y, labels=data_x, autopct='%1.1f%%', startangle=90)
        elif chart_type == 'line':
            plt.plot(data_x, data_y, marker='o', color='steelblue')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return chart_url
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None


# =====================================================
# ROUTES - CORE PAGES
# =====================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple validation (in production, use proper authentication)
        if username and password:
            session['loggedin'] = True
            session['username'] = username
            flash(f'Welcome {username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.', 'error')
        elif password != confirm_password:
            flash('Passwords do not match.', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
        else:
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# =====================================================
# ROUTES - USER PAGES
# =====================================================

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    username = session.get('username', 'Guest')
    return render_template('dashboard.html', username=username)


@app.route('/profile')
def profile():
    """User profile"""
    if 'loggedin' not in session:
        flash('Please login first.', 'warning')
        return redirect(url_for('login'))
    
    # Format: [username, password, email, phone, location]
    account = (
        session.get('username', 'User'),
        '••••••••',
        f"{session.get('username', 'user')}@example.com",
        '+91-XXXXXXXXXX',
        'India'
    )
    return render_template('profile.html', account=account)


# =====================================================
# ROUTES - CROP RECOMMENDATION
# =====================================================

@app.route('/crop-recommend', methods=['GET', 'POST'])
def crop_recommend():
    """Crop recommendation using ML model"""
    if request.method == 'POST':
        try:
            # Get form data
            nitrogen = float(request.form.get('nitrogen', 0))
            phosphorus = float(request.form.get('phosphorus', 0))
            potassium = float(request.form.get('potassium', 0))
            temperature = float(request.form.get('temperature', 0))
            humidity = float(request.form.get('humidity', 0))
            ph_value = float(request.form.get('phValue', request.form.get('pH', 0)))
            rainfall = float(request.form.get('rainfall', 0))
            
            # Validation
            if not (0 <= nitrogen <= 300):
                return render_template('crop-recommend.html', error="Nitrogen must be 0-300 kg/ha")
            if not (0 <= phosphorus <= 150):
                return render_template('crop-recommend.html', error="Phosphorus must be 0-150 kg/ha")
            if not (0 <= potassium <= 250):
                return render_template('crop-recommend.html', error="Potassium must be 0-250 kg/ha")
            if not (0 <= temperature <= 45):
                return render_template('crop-recommend.html', error="Temperature must be 0-45°C")
            if not (0 <= humidity <= 100):
                return render_template('crop-recommend.html', error="Humidity must be 0-100%")
            if not (4 <= ph_value <= 14):
                return render_template('crop-recommend.html', error="pH must be 4-14")
            if not (0 <= rainfall <= 2000):
                return render_template('crop-recommend.html', error="Rainfall must be 0-2000mm")
            
            # Predict
            if crop_model:
                features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
                probabilities = crop_model.predict_proba(features)[0]
                top_indices = probabilities.argsort()[-3:][::-1]
                crop_labels = crop_model.classes_
                recommendations = crop_labels[top_indices].tolist()
                
                return render_template('crop-recommend.html', 
                                     recommendations=recommendations,
                                     input_data={'nitrogen': nitrogen, 'phosphorus': phosphorus, 
                                               'potassium': potassium, 'temperature': temperature,
                                               'humidity': humidity, 'pH': ph_value, 'rainfall': rainfall})
            else:
                return render_template('crop-recommend.html', error="Model not available")
                
        except ValueError:
            return render_template('crop-recommend.html', error="Invalid input values")
        except Exception as e:
            return render_template('crop-recommend.html', error=f"Error: {str(e)}")
    
    return render_template('crop-recommend.html')


# =====================================================
# ROUTES - FERTILIZER RECOMMENDATION
# =====================================================

@app.route('/fertilizer-recommend', methods=['GET', 'POST'])
def fertilizer_recommend():
    """Fertilizer recommendation using ML model"""
    if request.method == 'POST':
        try:
            if not fertilizer_model:
                return render_template('fertilizer-recommend.html', error="Model not available")
            
            # Get form data
            temperature = float(request.form.get('temperature', 0))
            humidity = float(request.form.get('humidity', 0))
            soil_moisture = float(request.form.get('soilMoisture', request.form.get('soil_moisture', 0)))
            soil_type = request.form.get('soilType', request.form.get('soil_type', ''))
            crop_type = request.form.get('cropType', request.form.get('crop_type', ''))
            nitrogen = float(request.form.get('nitrogen', 0))
            potassium = float(request.form.get('potassium', 0))
            phosphorus = float(request.form.get('phosphorous', request.form.get('phosphorus', 0)))
            
            # Encode categorical features
            soil_encoded = le_soil.transform([soil_type])[0]
            crop_encoded = le_crop.transform([crop_type])[0]
            
            # Predict
            features = [[temperature, humidity, soil_moisture, soil_encoded, crop_encoded, 
                        nitrogen, potassium, phosphorus]]
            prediction = fertilizer_model.predict(features)[0]
            recommendation = le_fertilizer.inverse_transform([prediction])[0]
            
            return render_template('fertilizer-recommend.html', 
                                 recommendation=recommendation,
                                 input_data={'temperature': temperature, 'humidity': humidity,
                                           'soil_moisture': soil_moisture, 'soil_type': soil_type})
        except Exception as e:
            return render_template('fertilizer-recommend.html', error=f"Error: {str(e)}")
    
    return render_template('fertilizer-recommend.html')


# =====================================================
# ROUTES - YIELD PREDICTION
# =====================================================

@app.route('/yield-predict', methods=['GET', 'POST'])
def yield_predict():
    """Yield prediction using ML model"""
    areas = []
    items = []
    
    if yield_df is not None:
        areas = yield_df['Area'].unique().tolist()
        items = yield_df['Item'].unique().tolist()
    
    if request.method == 'POST':
        try:
            if not yield_model or not yield_preprocessor:
                return render_template('yield-predict.html', 
                                     error="Model not available",
                                     areas=areas, items=items)
            
            year = int(request.form.get('Year', request.form.get('year', 2010)))
            rainfall = float(request.form.get('average_rain_fall_mm_per_year', request.form.get('rainfall', 0)))
            pesticides = float(request.form.get('pesticides_tonnes', request.form.get('pesticides', 0)))
            temperature = float(request.form.get('avg_temp', request.form.get('temperature', 0)))
            area = request.form.get('Area', request.form.get('area', ''))
            item = request.form.get('Item', request.form.get('item', ''))
            
            # Validate
            if not (1990 <= year <= 2013):
                return render_template('yield-predict.html', error="Year must be 1990-2013",
                                     areas=areas, items=items)
            
            # Prepare and transform features
            features = np.array([[year, rainfall, pesticides, temperature, area, item]], dtype=object)
            transformed = yield_preprocessor.transform(features)
            
            # Predict
            prediction = yield_model.predict(transformed)[0]
            
            return render_template('yield-predict.html',
                                 prediction=round(prediction, 2),
                                 areas=areas, items=items,
                                 input_data={'year': year, 'area': area, 'item': item})
        except Exception as e:
            return render_template('yield-predict.html', 
                                 error=f"Error: {str(e)}",
                                 areas=areas, items=items)
    
    return render_template('yield-predict.html', areas=areas, items=items)


# =====================================================
# ROUTES - DATA ANALYSIS
# =====================================================

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    """Analyze agricultural data"""
    if analysis_df is None:
        return render_template('analysis.html', error="Data not available")
    
    states = analysis_df['state'].unique().tolist() if 'state' in analysis_df.columns else []
    years = sorted(analysis_df['year'].unique().tolist()) if 'year' in analysis_df.columns else []
    
    if request.method == 'POST':
        try:
            state = request.form.get('state')
            year = int(request.form.get('year'))
            
            # Filter data
            filtered = analysis_df[(analysis_df['state'] == state) & (analysis_df['year'] == year)]
            
            if filtered.empty:
                return render_template('analysis.html', 
                                     states=states, years=years,
                                     error=f"No data for {state} in {year}")
            
            # Create charts
            chart1_url = None
            chart2_url = None
            chart3_url = None
            
            if 'crop_type' in filtered.columns and 'cost_of_production_per_hectare' in filtered.columns:
                chart1_url = generate_chart('bar', 
                                          filtered['crop_type'].values,
                                          filtered['cost_of_production_per_hectare'].values,
                                          f'Cost of Production - {state} ({year})',
                                          'Crop Type', 'Cost (₹/ha)')
            
            if 'crop_type' in filtered.columns and 'cultivation_area_hectares' in filtered.columns:
                area_grouped = filtered.groupby('crop_type')['cultivation_area_hectares'].sum()
                chart2_url = generate_chart('pie',
                                          area_grouped.index.values,
                                          area_grouped.values,
                                          f'Cultivation Area - {state} ({year})',
                                          'Crop Type', 'Area (ha)')
            
            if 'crop_type' in filtered.columns and 'rainfall_mm' in filtered.columns:
                chart3_url = generate_chart('bar',
                                          filtered['crop_type'].values,
                                          filtered['rainfall_mm'].values,
                                          f'Rainfall Impact - {state} ({year})',
                                          'Crop Type', 'Rainfall (mm)')
            
            return render_template('analysis.html',
                                 states=states, years=years,
                                 chart1=chart1_url, chart2=chart2_url, chart3=chart3_url,
                                 selected_state=state, selected_year=year)
        except Exception as e:
            return render_template('analysis.html',
                                 states=states, years=years,
                                 error=f"Error: {str(e)}")
    
    return render_template('analysis.html', states=states, years=years)


# =====================================================
# ROUTES - ADDITIONAL PAGES
# =====================================================

@app.route('/weather-forecast')
def weather_forecast():
    """Weather forecast page"""
    return render_template('weather-forecast.html')


@app.route('/help')
def help_page():
    """Help and support page"""
    return render_template('help.html')


# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    flash('An internal server error occurred.', 'error')
    return redirect(url_for('index')), 500


# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'crop': 'loaded' if crop_model else 'not_loaded',
            'fertilizer': 'loaded' if fertilizer_model else 'not_loaded',
            'yield': 'loaded' if yield_model else 'not_loaded'
        }
    })


# =====================================================
# CONTEXT PROCESSORS
# =====================================================

@app.context_processor
def inject_user():
    """Inject user data into template context"""
    return dict(
        username=session.get('username', 'Guest'),
        loggedin=session.get('loggedin', False)
    )


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("AgroSmart ML-Driven Crop Recommender System")
    print("="*60)
    print(f"Base Path: {BASE_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\nStarting Flask development server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # For production, use:
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    
    # For development:
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
