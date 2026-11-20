import os
import joblib
import shap
import numpy as np
import pandas as pd


class HousePricePredictor:
    def __init__(self, default_radius=10.0, fallback_ratio=None):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "catboost_conformal_house_price_model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError("❌ 找不到 model_features.pkl，請確認已 push 到 GitHub")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)
        self.default_radius = default_radius
        self.fallback_ratio = fallback_ratio

        extracted_radius = self._extract_conformal_radius(self.model)
        self.conformal_radius = extracted_radius

        # 自動尋找實際的模型物件
        actual_model = self.model
        if isinstance(self.model, dict):
            for key in ["base_model", "model", "regressor", "estimator", "catboost", "predictor"]:
                if key in self.model and hasattr(self.model[key], "predict"):
                    actual_model = self.model[key]
                    break

            if actual_model is self.model:
                for _, v in self.model.items():
                    if hasattr(v, "predict"):
                        actual_model = v
                        break

        if hasattr(actual_model, "model"):
            actual_model = actual_model.model

        self.actual_model = actual_model
        self.explainer = self._init_explainer(actual_model)

    def _init_explainer(self, model_obj):
        try:
            return shap.TreeExplainer(model_obj)
        except Exception:
            try:
                return shap.Explainer(model_obj.predict, pd.DataFrame([ {c: 0 for c in self.model_features} ]))
            except Exception:
                return None

    def _extract_conformal_radius(self, model_obj):
        candidate_keys = [
            "q_hat_90", "q90", "q_hat", "conformal_q", "quantile_90",
            "radius_90", "margin_90", "margin", "interval_radius", "delta_90"
        ]
        if isinstance(model_obj, dict):
            for k in candidate_keys:
                if k in model_obj and isinstance(model_obj[k], (int, float, np.number)):
                    return float(model_obj[k])
            for sub_key in ["conformal", "quantiles", "intervals", "meta"]:
                if isinstance(model_obj.get(sub_key), dict):
                    for k in candidate_keys + ["0.9", 0.9, "90"]:
                        if k in model_obj[sub_key] and isinstance(model_obj[sub_key][k], (int, float, np.number)):
                            return float(model_obj[sub_key][k])
        else:
            for attr in candidate_keys:
                if hasattr(model_obj, attr):
                    val = getattr(model_obj, attr)
                    if isinstance(val, (int, float, np.number)):
                        return float(val)
        return None

    def _align_features(self, case_dict):
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        missing_cols = {col: 0 for col in self.model_features if col not in df.columns}
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        return df[self.model_features]

    def _feature_to_human(self, feature, value):
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"
        HUMAN_MAP = {
            "building_area_sqm": f"建物總面積約 {value:.1f} 平方公尺",
            "main_area": f"主建物面積約 {value:.1f} 坪",
            "balcony_area": f"陽台面積約 {value:.1f} 坪",
            "building_age": f"屋齡約 {int(value)} 年",
            "floor": f"位於第 {int(value)} 樓",
            "total_floors": f"建物總樓層 {int(value)} 樓",
            "has_parking": "具備車位" if value == 1 else "未附車位",
            "has_elevator": "設有電梯" if value == 1 else "未設電梯",
        }
        return HUMAN_MAP.get(feature, feature)

    def predict(self, case_dict):
        X = self._align_features(case_dict)
        pred = float(self.actual_model.predict(X)[0])

        explanation_lines = []
        if self.explainer is not None:
            shap_values = self.explainer(X)
            shap_vals = shap_values.values[0]
            feature_names = X.columns
            base_value = float(np.mean(self.explainer.expected_value)) if hasattr(self.explainer, "expected_value") else pred

            explanation_lines.append(f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整：")
            idx_sorted = np.argsort(np.abs(shap_vals))[-10:][::-1]
            for i in idx_sorted:
                feat = feature_names[i]
                val = X.iloc[0][feat]
                shap_val = float(shap_vals[i])
                if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                    continue
                human_text = self._feature_to_human(feat, val)
                direction = "推升" if shap_val > 0 else "下修"
                explanation_lines.append(f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪")

        explanation_lines.append(f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪。")

        radius = self.conformal_radius
        if radius is None and self.fallback_ratio is not None:
            radius = pred * float(self.fallback_ratio)
        if radius is None and self.default_radius is not None:
            radius = float(self.default_radius)

        lower_bound = None
        upper_bound = None
        if radius is not None:
            lower_bound = max(0.0, pred - radius)
            upper_bound = pred + radius
            explanation_lines.append(
                f"📊 90% 信賴區間：約 {lower_bound:.1f} ～ {upper_bound:.1f} 萬 / 坪（±{radius:.1f} 萬 / 坪）。"
            )

        return {
            "predicted_price": pred,
            "ci_90_lower": lower_bound,
            "ci_90_upper": upper_bound,
            "ci_90_radius": radius,
            "explanation": "\n".join(explanation_lines),
        }