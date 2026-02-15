import xgboost as xgb
import pandas as pd
import json 
import os

class LoanScreener:
    def __init__(self, models_path='Models'):
        # Load the Gatekeeper classifier
        self.gatekeeper = xgb.Booster()
        self.gatekeeper.load_model(os.path.join(models_path, 'best_gatekeeper_model.json'))
        # Extract feature names directly from the trained XGBoost model
        self.gatekeeper_features = self.gatekeeper.feature_names
        # Load the Default Probability Model
        self.risk_model = xgb.Booster()
        self.risk_model.load_model(os.path.join(models_path, 'default_risk_model_v1.json')) 
        # Load risk features from the valid JSON file
        with open(os.path.join(models_path, 'default_model_features.json'), 'r') as f: 
            self.risk_features = json.load(f)

    def screen_customer(self, data):
        """
        Inference engine utilizing XGBoost categorical support.
        """
        # 1. Prepare Gatekeeper Data
        gk_features = self.gatekeeper_features
        gk_df = data[gk_features].copy()
        
        # PEPE Style: Explicitly cast categorical columns to 'category' Dtype
        cat_cols = ['title', 'zip_code', 'addr_state']
        for col in cat_cols:
            if col in gk_df.columns:
                gk_df[col] = gk_df[col].astype('category')
        
        # 2. Gatekeeper Inference
        gk_input = xgb.DMatrix(gk_df, enable_categorical=True)
        prob_pass = self.gatekeeper.predict(gk_input)[0]

        # Threshold check for eligibility
        if prob_pass < 0.5:
            return {"Status": "REJECTED", "Reason": "Fails eligibility criteria"}

        # 3. Risk Assessment (41 features)
        risk_input = xgb.DMatrix(data[self.risk_features])
        prob_default = float(self.risk_model.predict(risk_input)[0])
        
        # Assign bank grade using the internal method
        grade, rate = self.assign_grade_and_rate(prob_default)
        return {
            "Status": "APPROVED",
            "Probability of Default": f"{prob_default:.2%}",
            "Assigned Grade": grade,
            "Interest Rate": rate
        }
    
    def assign_grade_and_rate(self, prob_default):
        # Convert probability to percentage for grading
        pd_pct = prob_default * 100
        
        if pd_pct < 2.5:
            return "A", "6.5%"
        elif pd_pct < 7.5:
            return "B", "10.0%"
        elif pd_pct < 15.0:
            return "C", "14.0%"
        elif pd_pct < 25.0:
            return "D", "19.5%"
        else:
            return "E", "24.9%"

if __name__ == "__main__":
    print("Initializing Loan Screener Engine...")
    try:
        screener = LoanScreener()
        print("Pipeline is fully operational and ready to connect to the UI!")
    except Exception as e:
        print(f"Error loading models: {e}")