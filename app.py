import streamlit as st

try:
    from predict import HousePricePredictor
except ImportError:
    from prediction import HousePricePredictor

# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="房價估價系統",
    page_icon="🏠",
    layout="wide",
)

# 避免 st.metric 數值過長時被省略號（....）截斷
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏠 房價估價系統｜依據 114 年不動產成交資料")
st.caption("CatBoost + Split Conformal Prediction + 可解釋 AI（SHAP）")

# =========================
# 行政區對照表（與 model_features.pkl 完全對齊）
# =========================
CITY_TOWN_MAP = {
    "臺北市": [
        "士林區", "大同區", "大安區", "中山區", "中正區", "內湖區",
        "文山區", "北投區", "松山區", "信義區", "南港區", "萬華區",
    ],
    "新北市": [
        "八里區", "三芝區", "三重區", "三峽區", "土城區", "中和區",
        "五股區", "平溪區", "永和區", "石門區", "石碇區", "汐止區",
        "坪林區", "板橋區", "林口區", "金山區", "泰山區", "烏來區",
        "貢寮區", "淡水區", "深坑區", "新店區", "新莊區", "瑞芳區",
        "萬里區", "樹林區", "雙溪區", "蘆洲區", "鶯歌區",
    ],
    "桃園市": [
        "八德區", "大園區", "大溪區", "中壢區", "平鎮區", "桃園區",
        "復興區", "新屋區", "楊梅區", "龍潭區", "龜山區", "蘆竹區", "觀音區",
    ],
    "新竹市": ["全區"],
    "新竹縣": [
        "北埔鄉", "寶山鄉", "峨眉鄉", "新埔鎮", "新豐鄉", "橫山鄉",
        "湖口鄉", "竹北市", "竹東鎮", "芎林鄉", "關西鎮",
    ],
    "苗栗縣": [
        "三義鄉", "三灣鄉", "大湖鄉", "公館鄉", "竹南鎮", "西湖鄉",
        "卓蘭鎮", "南庄鄉", "後龍鎮", "苑裡鎮", "苗栗市", "泰安鄉",
        "通霄鎮", "造橋鄉", "獅潭鄉", "銅鑼鄉", "頭份市", "頭屋鄉",
    ],
    "臺中市": [
        "大甲區", "大安區", "大肚區", "大里區", "大雅區", "中區",
        "太平區", "北屯區", "北區", "外埔區", "石岡區", "后里區",
        "西屯區", "西區", "沙鹿區", "和平區", "東區", "東勢區",
        "南屯區", "南區", "烏日區", "神岡區", "梧棲區", "清水區",
        "新社區", "潭子區", "龍井區", "豐原區", "霧峰區",
    ],
    "南投縣": [
        "中寮鄉", "仁愛鄉", "水里鄉", "名間鄉", "竹山鎮", "信義鄉",
        "南投市", "埔里鎮", "草屯鎮", "國姓鄉", "魚池鄉", "鹿谷鄉", "集集鎮",
    ],
    "彰化縣": [
        "二水鄉", "二林鎮", "大村鄉", "大城鄉", "北斗鎮", "永靖鄉",
        "田中鎮", "田尾鄉", "竹塘鄉", "伸港鄉", "秀水鄉", "和美鎮",
        "社頭鄉", "芬園鄉", "花壇鄉", "芳苑鄉", "員林市", "埔心鄉",
        "埔鹽鄉", "埤頭鄉", "鹿港鎮", "溪州鄉", "溪湖鎮", "彰化市",
        "福興鄉", "線西鄉",
    ],
    "雲林縣": [
        "二崙鄉", "口湖鄉", "土庫鎮", "大埤鄉", "元長鄉", "斗六市",
        "斗南鎮", "水林鄉", "北港鎮", "古坑鄉", "四湖鄉", "西螺鎮",
        "東勢鄉", "林內鄉", "虎尾鎮", "崙背鄉", "麥寮鄉", "莿桐鄉",
        "台西鄉", "褒忠鄉",
    ],
    "嘉義市": ["全區"],
    "嘉義縣": [
        "大林鎮", "大埔鄉", "中埔鄉", "六腳鄉", "太保市", "水上鄉",
        "布袋鎮", "民雄鄉", "朴子市", "竹崎鄉", "東石鄉", "梅山鄉",
        "鹿草鄉", "番路鄉", "新港鄉", "溪口鄉", "義竹鄉",
    ],
    "臺南市": [
        "七股區", "下營區", "大內區", "山上區", "中西區", "仁德區",
        "六甲區", "北門區", "北區", "左鎮區", "永康區", "玉井區",
        "白河區", "安平區", "安定區", "安南區", "西港區", "佳里區",
        "官田區", "東山區", "東區", "南化區", "南區", "後壁區",
        "柳營區", "將軍區", "麻豆區", "善化區", "新化區", "新市區",
        "新營區", "楠西區", "學甲區", "歸仁區", "關廟區", "鹽水區",
    ],
    "高雄市": [
        "三民區", "大社區", "大寮區", "大樹區", "小港區", "仁武區",
        "內門區", "六龜區", "左營區", "永安區", "田寮區", "甲仙區",
        "杉林區", "岡山區", "林園區", "阿蓮區", "前金區", "前鎮區",
        "美濃區", "苓雅區", "茂林區", "茄萣區", "梓官區", "鳥松區",
        "湖內區", "新興區", "楠梓區", "路竹區", "鼓山區", "旗山區",
        "旗津區", "鳳山區", "橋頭區", "燕巢區", "彌陀區", "鹽埕區",
    ],
    "屏東縣": [
        "九如鄉", "三地門鄉", "內埔鄉", "竹田鄉", "牡丹鄉", "車城鄉",
        "里港鄉", "佳冬鄉", "來義鄉", "東港鎮", "枋山鄉", "枋寮鄉",
        "林邊鄉", "長治鄉", "南州鄉", "屏東市", "恆春鎮", "春日鄉",
        "崁頂鄉", "泰武鄉", "琉球鄉", "高樹鄉", "新埤鄉", "新園鄉",
        "獅子鄉", "萬丹鄉", "萬巒鄉", "滿州鄉", "瑪家鄉", "潮州鎮",
        "麟洛鄉", "鹽埔鄉",
    ],
    "宜蘭縣": [
        "三星鄉", "大同鄉", "五結鄉", "冬山鄉", "壯圍鄉", "宜蘭市",
        "南澳鄉", "員山鄉", "頭城鎮", "礁溪鄉", "羅東鎮", "蘇澳鎮",
    ],
    "花蓮縣": [
        "玉里鎮", "光復鄉", "吉安鄉", "秀林鄉", "花蓮市",
        "富里鄉", "新城鄉", "瑞穗鄉", "萬榮鄉", "壽豐鄉", "鳳林鎮", "豐濱鄉",
    ],
    "臺東縣": [
        "大武鄉", "太麻里鄉", "成功鎮", "池上鄉", "卑南鄉",
        "東河鄉", "長濱鄉", "海端鄉", "鹿野鄉", "台東市", "關山鎮",
    ],
    "基隆市": [
        "七堵區", "中山區", "中正區", "仁愛區", "安樂區", "信義區", "暖暖區",
    ],
    "澎湖縣": [
        "七美鄉", "白沙鄉", "西嶼鄉", "馬公市", "望安鄉", "湖西鄉",
    ],
    "金門縣": [
        "金沙鎮", "金城鎮", "金湖鎮", "金寧鄉", "烈嶼鄉",
    ],
}

# 建物型態完整選項（對應模型特徵欄位名稱）
BUILDING_TYPES = [
    "住宅大樓(11層含以上有電梯)",
    "華廈(10層含以下有電梯)",
    "公寓(5樓含以下無電梯)",
    "透天厝",
    "套房(1房1廳1衛)",
    "店面(店鋪)",
    "辦公商業大樓",
    "廠辦",
    "其他",
]

# 主要用途完整選項（對應模型特徵欄位名稱）
MAIN_USES = [
    "住家用",
    "住商用",
    "住商辦用",
    "住辦用",
    "住工用",
    "商業用",
    "商辦用",
    "辦公用",
    "工商用",
    "工業用",
    "農業用",
    "其他",
]


def _fmt_num(val: float) -> str:
    """若四捨五入到小數一位後為整數（如 113.0）則顯示 113，否則顯示一位小數（如 93.8）"""
    rounded = round(float(val), 1)
    return f"{int(rounded)}" if rounded.is_integer() else f"{rounded:.1f}"


# =========================
# 載入模型
# =========================
@st.cache_resource
def load_predictor():
    return HousePricePredictor()


predictor = load_predictor()

# =========================
# 側邊欄輸入
# =========================
st.sidebar.header("📋 房屋基本資料")

city = st.sidebar.selectbox("縣市", list(CITY_TOWN_MAP.keys()))
town = st.sidebar.selectbox("鄉鎮市區", CITY_TOWN_MAP[city])
district = city if town == "全區" else f"{city}{town}"
st.sidebar.caption(f"📍 模型對應行政區：{district}")

building_type = st.sidebar.selectbox("建物型態", BUILDING_TYPES)
main_use = st.sidebar.selectbox("主要用途", MAIN_USES)
building_age = st.sidebar.number_input("屋齡（年）", min_value=0, max_value=80, value=15)

# 坪數與平方公尺連動參考（1 坪 ≈ 3.305785 平方公尺）
building_area_ping = st.sidebar.number_input(
    "建物總面積（坪）", min_value=3.0, max_value=300.0, value=30.0, step=0.5
)
building_area_sqm = round(building_area_ping * 3.305785, 2)
st.sidebar.caption(f"📐 換算建物總面積：{building_area_sqm:.2f} 平方公尺")

main_area = st.sidebar.number_input("主建物面積（坪）", min_value=1.0, max_value=200.0, value=20.0, step=0.5)
balcony_area = st.sidebar.number_input("陽台面積（坪）", min_value=0.0, max_value=50.0, value=2.5, step=0.5)

total_floors = st.sidebar.number_input("總樓層數", min_value=1, max_value=100, value=12)
floor = st.sidebar.number_input("所在樓層", min_value=1, max_value=int(total_floors), value=min(5, int(total_floors)))

has_parking = st.sidebar.radio("是否有車位", ["有", "無"], horizontal=True)
has_elevator = st.sidebar.radio("是否有電梯", ["有", "無"], horizontal=True)

# =========================
# 組合輸入資料
# =========================
case_dict = {
    "district": district,
    "building_type": building_type,
    "main_use": main_use,
    "building_age": building_age,
    "building_area_sqm": building_area_sqm,
    "main_area": main_area,
    "balcony_area": balcony_area,
    "floor": floor,
    "total_floors": total_floors,
    "has_parking": 1 if has_parking == "有" else 0,
    "has_elevator": 1 if has_elevator == "有" else 0,
}

# =========================
# 主畫面
# =========================
st.subheader("📊 預測結果")

if "result" not in st.session_state:
    st.session_state.result = None

if st.button("🚀 開始估價", type="primary"):
    with st.spinner("模型預測與 SHAP 特徵歸因計算中，請稍候..."):
        st.session_state.result = predictor.predict(case_dict)

if st.session_state.result is not None:
    result = st.session_state.result
    pred_price = result["predicted_price"]
    ci_lower = result.get("ci_90_lower")
    ci_upper = result.get("ci_90_upper")
    ci_radius = result.get("ci_90_radius")
    confidence_pct = result.get("confidence_pct", 90)

    # 計算總價參考（萬）
    est_total_price = pred_price * building_area_ping

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="💰 預估單價",
            value=f"{_fmt_num(pred_price)} 萬/坪",
        )
    with col2:
        if ci_lower is not None and ci_upper is not None:
            st.metric(
                label=f"📊 {confidence_pct}% 預測區間",
                value=f"{_fmt_num(ci_lower)}~{_fmt_num(ci_upper)}萬/坪",
                delta=f"±{_fmt_num(ci_radius)}萬/坪" if ci_radius is not None else None,
                delta_color="off",
            )
        else:
            st.metric(label=f"📊 {confidence_pct}% 預測區間", value="尚未設定")
    with col3:
        st.metric(
            label="🏷️ 參考總價（依輸入總坪數）",
            value=f"{est_total_price:,.0f} 萬",
        )

    if ci_lower is not None and ci_upper is not None:
        total_lower = ci_lower * building_area_ping
        total_upper = ci_upper * building_area_ping
        st.markdown("#### 📈 90% 信賴/預測區間數值明細")
        st.write(
            f"- **預估單價**：**{_fmt_num(pred_price)} 萬/坪**\n"
            f"- **90% 區間下界（5% 分位）**：**{_fmt_num(ci_lower)} 萬/坪**（參考總價約 **{total_lower:,.0f} 萬**）\n"
            f"- **90% 區間上界（95% 分位）**：**{_fmt_num(ci_upper)} 萬/坪**（參考總價約 **{total_upper:,.0f} 萬**）\n"
            f"- **區間半徑（±）**：**±{_fmt_num(ci_radius)} 萬/坪**"
        )

    st.markdown("### 📝 中文估價說明（SHAP 歸因分析）")
    st.text(result["explanation"])

    if ci_lower is not None and ci_upper is not None:
        st.success(
            f"💰 模型預估單價約 **{_fmt_num(pred_price)} 萬/坪**，"
            f"{confidence_pct}% 預測區間為 **{_fmt_num(ci_lower)}~{_fmt_num(ci_upper)}萬/坪**。"
        )
else:
    st.info("👈 請先填寫左側房屋資料，並點擊「🚀 開始估價」")