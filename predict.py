import os
import joblib
import shap
import numpy as np
import pandas as pd


class HousePricePredictor:
    # 若模型檔內未附帶 Split Conformal 校準半徑（q_hat），使用的預設半徑（萬/坪）
    DEFAULT_Q90_RADIUS = 12.0

    def __init__(self, q90_radius: float | None = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "catboost_conformal_house_price_model.pkl")
        feature_path = os.path.join(base_dir, "model_features.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型檔：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"❌ 找不到 model_features.pkl，請確認檔案位置：{feature_path}")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        # 自動尋找實際的模型物件與 Conformal 90% 區間半徑
        actual_model = self.model
        extracted_radius = None

        if isinstance(self.model, dict):
            for key in ["base_model", "model", "regressor", "estimator", "catboost", "cb", "lgb", "predictor"]:
                if key in self.model and hasattr(self.model[key], "predict"):
                    actual_model = self.model[key]
                    break

            if actual_model is self.model:
                for _, v in self.model.items():
                    if hasattr(v, "predict"):
                        actual_model = v
                        break

            # 嘗試從模型字典中擷取 Split Conformal 90% 區間半徑 / 分位數
            for q_key in [
                "ci_90_radius",
                "q_90",
                "q90",
                "quantile_90",
                "conformal_q90",
                "q_hat",
                "q",
                "quantile",
                "conformal_radius",
                "radius",
            ]:
                if q_key in self.model and self.model[q_key] is not None:
                    try:
                        extracted_radius = float(self.model[q_key])
                        break
                    except (TypeError, ValueError):
                        pass

        if hasattr(actual_model, "model"):
            actual_model = actual_model.model

        self._predictor_model = actual_model
        if q90_radius is not None:
            self.q90_radius = float(q90_radius)
        elif extracted_radius is not None:
            self.q90_radius = extracted_radius
        else:
            self.q90_radius = self.DEFAULT_Q90_RADIUS

        # 初始化 SHAP 解釋器
        self.explainer = shap.TreeExplainer(actual_model)

    def _align_features(self, case_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame([case_dict])
        df = pd.get_dummies(df)

        # 使用字典一次性補齊缺少的欄位，避免 Pandas DataFrame 碎片化警告
        missing_cols = {col: 0 for col in self.model_features if col not in df.columns}
        if missing_cols:
            missing_df = pd.DataFrame(missing_cols, index=df.index)
            df = pd.concat([df, missing_df], axis=1)

        return df[self.model_features]

    def _feature_to_human(self, feature: str, value) -> str:
        if feature.startswith("district_"):
            return f"位於「{feature.replace('district_', '')}」"
        if feature.startswith("building_type_"):
            return f"建物型態為「{feature.replace('building_type_', '')}」"
        if feature.startswith("main_use_"):
            return f"主要用途為「{feature.replace('main_use_', '')}」"

        HUMAN_MAP = {
            "building_area_sqm": f"建物總面積約 {float(value):.1f} 平方公尺",
            "main_area": f"主建物面積約 {float(value):.1f} 坪",
            "balcony_area": f"陽台面積約 {float(value):.1f} 坪",
            "building_age": f"屋齡約 {int(value)} 年",
            "floor": f"位於第 {int(value)} 樓",
            "total_floors": f"建物總樓層 {int(value)} 樓",
            "has_parking": "具備車位" if value == 1 else "未附車位",
            "has_elevator": "設有電梯" if value == 1 else "未設電梯",
        }

        return HUMAN_MAP.get(feature, feature)

    def predict(self, case_dict: dict) -> dict:
        X = self._align_features(case_dict)

        if hasattr(self._predictor_model, "predict"):
            pred = float(self._predictor_model.predict(X)[0])
        elif hasattr(self.model, "predict"):
            pred = float(self.model.predict(X)[0])
        else:
            pred = float(list(self.model.values())[0].predict(X)[0])

        # 計算 90% 預測/信賴區間（單價：萬/坪，下界不低於 0）
        ci_radius = float(self.q90_radius)
        ci_lower = max(0.0, pred - ci_radius)
        ci_upper = pred + ci_radius

        shap_values = self.explainer(X)
        shap_vals = shap_values.values[0]
        feature_names = X.columns
        base_value = float(np.mean(self.explainer.expected_value))

        explanation_lines = [
            f"📌 模型基準單價約為 {base_value:.1f} 萬 / 坪，以下條件使價格進行調整："
        ]

        idx_sorted = np.argsort(np.abs(shap_vals))[-10:][::-1]
        for i in idx_sorted:
            feat = feature_names[i]
            val = X.iloc[0][feat]
            shap_val = float(shap_vals[i])
            if val == 0 and feat.startswith(("district_", "building_type_", "main_use_")):
                continue
            human_text = self._feature_to_human(feat, val)
            direction = "推升" if shap_val > 0 else "下修"
            explanation_lines.append(
                f"👉 {human_text}，使單價約 {direction} {abs(shap_val):.1f} 萬 / 坪"
            )

        explanation_lines.append(
            f"\n➡️ 綜合以上因素後，模型推估本案合理單價約為 {pred:.1f} 萬 / 坪"
            f"（90% 預測區間：{ci_lower:.1f} ～ {ci_upper:.1f} 萬 / 坪，±{ci_radius:.1f} 萬 / 坪）。"
        )

        return {
            "predicted_price": pred,
            "ci_90_lower": ci_lower,
            "ci_90_upper": ci_upper,
            "ci_90_radius": ci_radius,
            "confidence_pct": 90,
            "explanation": "\n".join(explanation_lines),
        }