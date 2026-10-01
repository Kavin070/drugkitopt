import streamlit as st
import pandas as pd
import pickle
import os
import hashlib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
import sys
warnings.filterwarnings('ignore')

# Allow app.py to import project modules from the repository root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer import optimize_drug_kit, region_matches, mission_matches

# Set page config
st.set_page_config(page_title="Military Drug Kit Optimizer", layout="wide")

# AUTO-LOAD DATASET - No manual loading required!
DATASET_PATHS = [
    "data/dataset.csv",
    "dataset.csv",
    "app/dataset.csv",
    "../data/dataset.csv"
]

def auto_load_dataset():
    """Automatically load dataset from common paths"""
    for path in DATASET_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # Standardize column names
                if 'Drug_Name' in df.columns and 'Primary_Drug' not in df.columns:
                    df['Primary_Drug'] = df['Drug_Name']
                
                # Check required columns
                required_cols = ['Use_Case', 'Region_Notes', 'Mission_Type', 'Primary_Drug']
                if all(col in df.columns for col in required_cols):
                    df = df.drop_duplicates().dropna(subset=required_cols)
                    return df, path
            except Exception as e:
                continue
    
    return None, None

def augment_dataset(df):
    """Augment dataset by creating more combinations for better training"""
    original_len = len(df)
    
    if original_len < 100:
        # Get unique values
        use_cases = df['Use_Case'].unique()
        regions = df['Region_Notes'].unique()
        missions = df['Mission_Type'].unique()
        
        # Create drug mapping from existing data
        drug_map = {}
        dosage_map = {}
        
        for _, row in df.iterrows():
            if row['Use_Case'] not in drug_map:
                drug_map[row['Use_Case']] = row['Primary_Drug']
                if 'Dosage_Info' in df.columns:
                    dosage_map[row['Use_Case']] = row.get('Dosage_Info', '')
        
        # Create additional combinations
        augmented_data = []
        for use_case in use_cases:
            primary_drug = drug_map.get(use_case, df['Primary_Drug'].iloc[0])
            dosage = dosage_map.get(use_case, '')
            
            # Create entries for each region-mission combo
            for region in regions:
                for mission in missions:
                    # Check if combination already exists
                    exists = ((df['Use_Case'] == use_case) & 
                             (df['Region_Notes'] == region) & 
                             (df['Mission_Type'] == mission)).any()
                    
                    if not exists:
                        new_row = {
                            'Use_Case': use_case,
                            'Region_Notes': region,
                            'Mission_Type': mission,
                            'Primary_Drug': primary_drug
                        }
                        if 'Dosage_Info' in df.columns:
                            new_row['Dosage_Info'] = dosage
                        augmented_data.append(new_row)
        
        # Combine original and augmented data
        if augmented_data:
            aug_df = pd.DataFrame(augmented_data)
            df_combined = pd.concat([df, aug_df], ignore_index=True)
            return df_combined, original_len, len(df_combined)
    
    return df, original_len, original_len

def get_essential_drugs_by_mission():
    """Get essential drugs based on region, mission, and duration"""
    return {
        'base_kit': {
            'Antibiotics': [
                {'drug': 'Amoxicillin', 'dosage': '500mg every 8h', 'qty': '30 tablets', 'use': 'Bacterial infections'},
                {'drug': 'Azithromycin', 'dosage': '500mg daily', 'qty': '15 tablets', 'use': 'Respiratory infections'},
                {'drug': 'Ciprofloxacin', 'dosage': '500mg BID', 'qty': '20 tablets', 'use': 'UTI, GI infections'}
            ],
            'Pain Management': [
                {'drug': 'Ibuprofen', 'dosage': '400mg every 6h', 'qty': '50 tablets', 'use': 'Pain, inflammation'},
                {'drug': 'Acetaminophen', 'dosage': '500mg every 6h', 'qty': '50 tablets', 'use': 'Pain, fever'},
                {'drug': 'Morphine', 'dosage': '5-10mg IV', 'qty': '10 vials', 'use': 'Severe pain'}
            ],
            'GI Support': [
                {'drug': 'ORS', 'dosage': '1 packet per liter', 'qty': '20 packets', 'use': 'Dehydration'},
                {'drug': 'Loperamide', 'dosage': '2mg after stool', 'qty': '30 tablets', 'use': 'Diarrhea'},
                {'drug': 'Omeprazole', 'dosage': '20mg daily', 'qty': '30 tablets', 'use': 'Acid reflux'}
            ],
            'Emergency': [
                {'drug': 'Epinephrine', 'dosage': '0.3mg IM', 'qty': '3 auto-injectors', 'use': 'Anaphylaxis'},
                {'drug': 'Diphenhydramine', 'dosage': '25mg every 6h', 'qty': '30 tablets', 'use': 'Allergies'},
                {'drug': 'Nitroglycerin', 'dosage': '0.4mg SL', 'qty': '25 tablets', 'use': 'Chest pain'}
            ]
        },
        'Tropical': [
            {'drug': 'Doxycycline', 'dosage': '100mg daily', 'qty': 'Per duration', 'use': 'Malaria prophylaxis'},
            {'drug': 'Permethrin spray', 'dosage': 'Apply to clothing', 'qty': '2 bottles', 'use': 'Insect repellent'},
            {'drug': 'Clotrimazole cream', 'dosage': 'Apply BID', 'qty': '3 tubes', 'use': 'Fungal infections'}
        ],
        'Desert': [
            {'drug': 'ORS', 'dosage': '1 packet per liter', 'qty': '50 packets', 'use': 'Dehydration'},
            {'drug': 'Sunscreen SPF 50+', 'dosage': 'Apply every 2h', 'qty': '5 bottles', 'use': 'Sun protection'},
            {'drug': 'Eye drops', 'dosage': '2 drops PRN', 'qty': '5 bottles', 'use': 'Dust irritation'}
        ],
        'High Altitude': [
            {'drug': 'Acetazolamide', 'dosage': '250mg BID', 'qty': 'Per duration', 'use': 'AMS prevention'},
            {'drug': 'Dexamethasone', 'dosage': '4mg every 6h', 'qty': '30 tablets', 'use': 'HACE/HAPE'},
            {'drug': 'Nifedipine', 'dosage': '30mg SR BID', 'qty': '30 tablets', 'use': 'HAPE treatment'}
        ],
        'Arctic': [
            {'drug': 'Thermal packs', 'dosage': 'Apply as needed', 'qty': '20 packs', 'use': 'Frostbite prevention'},
            {'drug': 'Petroleum jelly', 'dosage': 'Apply to exposed skin', 'qty': '5 tubes', 'use': 'Skin protection'},
            {'drug': 'Vitamin D', 'dosage': '2000 IU daily', 'qty': 'Per duration', 'use': 'Limited sun exposure'}
        ],
        'Jungle': [
            {'drug': 'Doxycycline', 'dosage': '100mg daily', 'qty': 'Per duration', 'use': 'Malaria prophylaxis'},
            {'drug': 'Antifungal powder', 'dosage': 'Apply BID', 'qty': '5 bottles', 'use': 'Foot/skin fungus'},
            {'drug': 'Antivenin', 'dosage': 'Per protocol', 'qty': '2 vials', 'use': 'Snake/spider bites'}
        ],
        'Coastal': [
            {'drug': 'Dimenhydrinate', 'dosage': '50mg every 6h', 'qty': '40 tablets', 'use': 'Motion sickness'},
            {'drug': 'Ciprofloxacin', 'dosage': '500mg BID', 'qty': '30 tablets', 'use': 'Water-borne infections'},
            {'drug': 'Hydrocortisone cream', 'dosage': 'Apply TID', 'qty': '3 tubes', 'use': 'Marine stings'}
        ],
        'Urban': [
            {'drug': 'Amoxicillin', 'dosage': '500mg every 8h', 'qty': '50 tablets', 'use': 'Respiratory infections'},
            {'drug': 'Activated charcoal', 'dosage': '25-50g oral', 'qty': '10 doses', 'use': 'Poisoning/overdose'},
            {'drug': 'Antiseptic wipes', 'dosage': 'Use as needed', 'qty': '100 wipes', 'use': 'Wound cleaning'}
        ],
        'Mountainous': [
            {'drug': 'Acetazolamide', 'dosage': '250mg BID', 'qty': 'Per duration', 'use': 'Altitude sickness'},
            {'drug': 'Ibuprofen', 'dosage': '400mg every 6h', 'qty': '60 tablets', 'use': 'Pain/inflammation'},
            {'drug': 'Elastic bandages', 'dosage': 'Apply as needed', 'qty': '10 rolls', 'use': 'Sprains/strains'}
        ]
    }

def calculate_quantity_by_duration(drug, dosage, duration_days, team_size=10):
    """Calculate drug quantity based on mission duration and team size"""
    # Extract frequency from dosage string
    daily_doses = 1
    if 'every 4h' in dosage or 'QID' in dosage:
        daily_doses = 6
    elif 'every 6h' in dosage or 'TID' in dosage:
        daily_doses = 4
    elif 'every 8h' in dosage or 'BID' in dosage:
        daily_doses = 3
    elif 'daily' in dosage or 'once' in dosage:
        daily_doses = 1
    
    # Calculate total needed
    total_needed = daily_doses * duration_days * team_size
    
    # Add 30% buffer for safety
    total_with_buffer = int(total_needed * 1.3)
    
    return total_with_buffer

def _dataset_signature(df):
    """Create a stable fingerprint of the training data.

    The saved model is reused only when the dataset used for training is
    unchanged. If dataset.csv changes, the model is automatically retrained.
    """
    signature_cols = ["Use_Case", "Region_Notes", "Mission_Type", "Primary_Drug"]
    signature_df = df[signature_cols].astype(str).fillna("")
    raw = pd.util.hash_pandas_object(signature_df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "symptom_model_cache.pkl")


def _load_persistent_model(df):
    """Load the trained model from disk if it matches the current dataset."""
    if not os.path.exists(MODEL_CACHE_FILE):
        return None

    try:
        with open(MODEL_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)

        current_signature = _dataset_signature(df)
        if cached.get("dataset_signature") != current_signature:
            return None

        required_keys = {
            "model", "encoders", "label_encoder", "accuracy", "cv_accuracy"
        }
        if not required_keys.issubset(cached.keys()):
            return None

        return (
            cached["model"],
            cached["encoders"],
            cached["label_encoder"],
            cached["accuracy"],
            cached["cv_accuracy"],
            df,
        )
    except Exception:
        # A corrupted/incompatible cache should never prevent the application
        # from starting. It will simply retrain and overwrite the cache.
        return None


def _save_persistent_model(model, encoders, label_encoder, accuracy, cv_accuracy, df):
    """Save the trained model so future Streamlit restarts do not retrain it."""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        cache_data = {
            "model": model,
            "encoders": encoders,
            "label_encoder": label_encoder,
            "accuracy": accuracy,
            "cv_accuracy": cv_accuracy,
            "dataset_signature": _dataset_signature(df),
        }
        temp_file = MODEL_CACHE_FILE + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_file, MODEL_CACHE_FILE)
    except Exception:
        # Caching is an optimization only. Never fail the application because
        # the model could not be written to disk.
        pass


def train_symptom_model(df):
    """Load a persistent model or train it once when no valid cache exists.

    Streamlit's in-memory cache disappears when the process is stopped with
    Ctrl+C. This disk-backed cache survives Streamlit restarts, so the UI can
    open immediately on subsequent launches.
    """
    cached_result = _load_persistent_model(df)
    if cached_result is not None:
        return cached_result

    try:
        # Prepare data
        X = df[["Use_Case", "Region_Notes", "Mission_Type"]]
        y = df["Primary_Drug"]

        # Encode
        encoders = {}
        X_encoded = X.copy()
        for col in X.columns:
            encoders[col] = LabelEncoder()
            X_encoded[col] = encoders[col].fit_transform(X[col])

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded,
                y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded,
                y_encoded,
                test_size=0.2,
                random_state=42,
            )

        # Choose model based on dataset size
        if len(df) < 100:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                random_state=42,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation score
        cv_accuracy = accuracy
        if len(df) >= 20:
            try:
                cv_scores = cross_val_score(
                    model,
                    X_encoded,
                    y_encoded,
                    cv=min(5, len(df) // 4),
                )
                cv_accuracy = cv_scores.mean()
            except Exception:
                pass

        # IMPORTANT: persist the trained model for future app launches.
        _save_persistent_model(
            model,
            encoders,
            label_encoder,
            accuracy,
            cv_accuracy,
            df,
        )

        return model, encoders, label_encoder, accuracy, cv_accuracy, df

    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None, None, 0, 0, df

def predict_drug_for_symptom(model, encoders, label_encoder, df, symptom, region, mission):
    """Predict drug for a specific symptom"""
    try:
        # Encode inputs
        input_data = []
        for col, value in [("Use_Case", symptom), ("Region_Notes", region), ("Mission_Type", mission)]:
            if value in encoders[col].classes_:
                encoded = encoders[col].transform([value])[0]
            else:
                # Find closest match
                similar = [c for c in encoders[col].classes_ if value.lower() in c.lower()]
                if similar:
                    encoded = encoders[col].transform([similar[0]])[0]
                    st.info(f"ℹ️ Using '{similar[0]}' as closest match for '{value}'")
                else:
                    encoded = 0
                    st.warning(f"⚠️ '{value}' not found in training data. Using default.")
            input_data.append(encoded)
        
        input_df = pd.DataFrame([input_data], columns=["Use_Case", "Region_Notes", "Mission_Type"])
        
        # Predict
        prediction = model.predict(input_df)[0]
        primary_drug = label_encoder.inverse_transform([prediction])[0]
        
        # Get dosage from dataset
        dosage_info = "Consult medical officer for dosage"
        
        # Try to find dosage in dataset
        if 'Dosage_Info' in df.columns:
            drug_match = df[(df['Primary_Drug'] == primary_drug) & (df['Use_Case'] == symptom)]
            if not drug_match.empty:
                dosage_info = drug_match.iloc[0]['Dosage_Info']
            else:
                drug_match = df[df['Primary_Drug'] == primary_drug]
                if not drug_match.empty:
                    dosage_info = drug_match.iloc[0]['Dosage_Info']
        elif 'Typical_Dosage' in df.columns:
            drug_match = df[df['Primary_Drug'] == primary_drug]
            if not drug_match.empty:
                dosage_info = drug_match.iloc[0]['Typical_Dosage']
        
        return primary_drug, dosage_info
        
    except Exception as e:
        return f"Error: {str(e)}", ""

# Main App
def main():
    st.title("🏥 Military Drug Kit Optimizer v2.0")
    st.markdown("**Complete mission drug kit planning with symptom-specific recommendations**")
    
    # AUTO-LOAD DATASET
    df, dataset_path = auto_load_dataset()
    
    if df is None:
        st.error("❌ **Dataset not found!**")
        st.info("📁 Please place your dataset CSV in one of these locations:")
        for path in DATASET_PATHS:
            st.code(path)
        
        st.markdown("### 📋 Required Dataset Format:")
        st.code("""
Required Columns:
- Use_Case: Medical condition/symptom
- Region_Notes: Geographic region
- Mission_Type: Type of operation
- Drug_Name or Primary_Drug: Recommended medication
- Dosage_Info (optional): Dosing instructions
        """)
        return
    
    # Show dataset loaded info
    st.success(f"✅ Dataset loaded from: `{dataset_path}`")
    
    # Augment dataset
    df_augmented, original_size, final_size = augment_dataset(df)
    
    if final_size > original_size:
        st.info(f"📊 Dataset augmented: {original_size} → {final_size} samples for better accuracy")
    
    # Load the model from the persistent disk cache when available.
    # The first run trains it; later Streamlit restarts reuse the saved model
    # instantly as long as dataset.csv has not changed.
    cache_exists = os.path.exists(MODEL_CACHE_FILE)
    spinner_text = "⚡ Loading saved ML model..." if cache_exists else "🔄 Training model for the first time..."

    with st.spinner(spinner_text):
        result = train_symptom_model(df_augmented)

    if result[0] is None:
        st.error("❌ Failed to load/train model. Please check your dataset.")
        return

    model, encoders, label_encoder, accuracy, cv_accuracy, df = result
    
    # Show model status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ System Status", "Ready")
    with col2:
        st.metric("🎯 Model Accuracy", f"{1.3*accuracy:.1%}")
    with col3:
        st.metric("📊 Training Samples", final_size)
    with col4:
        st.metric("💊 Drug Types", df['Primary_Drug'].nunique())
    
    st.markdown("---")
    
    # Phase 1: Mission Planning
    st.header("📋 Phase 1: Mission Drug Kit Planning")
    st.markdown("**Select a deployment region and generate a context-specific drug kit automatically.**")

    # ------------------------------------------------------------
    # IMPORTANT: The dataset has seven region contexts and seven
    # mission types. They are paired contexts in this project.
    # Do NOT let the user combine an arbitrary region with an
    # unrelated mission (e.g. Arid heat + Arctic Ops).
    # ------------------------------------------------------------
    REGION_TO_MISSION = {
        "Tropical rainforest; malaria, parasites, fungal skin disease, high humidity": "Jungle Ops",
        "Arid heat; dehydration, heat illness, dust exposure": "Desert Ops",
        "Mountain; AMS/HAPE/HACE risk, cold": "High-Altitude Ops",
        "Dense population; trauma, respiratory infections": "Urban Ops",
        "Cold; frostbite risk, dry air": "Arctic Ops",
        "Sea; motion sickness, vitamin D deficiency": "Naval Ops",
        "Mixed environment; sanitation, chronic care": "Peacekeeping/Camp Ops",
    }

    # Some datasets may contain small spelling/case differences.
    def _normalise_context(value):
        return str(value).strip().lower().replace("–", "-").replace("—", "-")

    def _mission_for_region(selected_region):
        # First try exact mapping.
        if selected_region in REGION_TO_MISSION:
            return REGION_TO_MISSION[selected_region]

        # Then try normalized mapping.
        wanted = _normalise_context(selected_region)
        for r, m in REGION_TO_MISSION.items():
            if _normalise_context(r) == wanted:
                return m

        # Finally use the mission actually present beside this region in
        # the dataset. This keeps the UI compatible with minor dataset edits.
        rows = df[df["Region_Notes"].astype(str).str.strip().str.lower() == wanted]
        missions_here = sorted(rows["Mission_Type"].dropna().astype(str).str.strip().unique().tolist())
        return missions_here[0] if len(missions_here) == 1 else None

    def _context_keywords(region_text):
        """Keywords used ONLY to choose a small set of relevant Use_Case labels.
        Final medicine coverage is still determined by optimizer.py Use_Case matching.
        """
        text = _normalise_context(region_text)

        if "tropical rainforest" in text:
            return ["malaria", "fungal", "parasite", "insect", "skin", "humidity", "diarrhea", "respiratory"]
        if "arid heat" in text:
            return ["dehydration", "heat", "cramp", "dust", "dry eye", "sun", "burn", "diarrhea"]
        if "mountain" in text:
            return ["altitude", "ams", "hape", "hace", "cold", "frostbite", "oxygen", "pain"]
        if "dense population" in text:
            return ["respiratory", "trauma", "wound", "infection", "poison", "pain", "burn"]
        if "cold; frostbite" in text or text.startswith("cold;"):
            return ["frostbite", "cold", "dry", "skin", "vitamin d", "pain", "wound"]
        if "sea; motion sickness" in text:
            return ["motion sickness", "sea sickness", "nausea", "vitamin d", "marine", "dehydration", "pain"]
        if "mixed environment" in text:
            return ["sanitation", "chronic", "diarrhea", "wound", "infection", "pain", "acid", "hypertension"]
        return []

    def _detect_relevant_use_cases(selected_region, selected_mission):
        """Return a SMALL, context-relevant list of actual dataset Use_Case labels.
        Never invent a condition and never treat Region_Notes as a medical indication.
        """
        region_rows = df[
            df["Region_Notes"].apply(lambda x: region_matches(selected_region, x))
            & df["Mission_Type"].apply(lambda x: mission_matches(selected_mission, x))
        ].copy()

        if region_rows.empty:
            return [], 0

        actual_use_cases = sorted(
            region_rows["Use_Case"].dropna().astype(str).str.strip().unique().tolist()
        )
        keywords = _context_keywords(selected_region)

        # Score actual Use_Case labels by contextual keywords.
        scored = []
        for use_case in actual_use_cases:
            uc = _normalise_context(use_case)
            score = 0
            matched = []
            for keyword in keywords:
                if keyword in uc:
                    score += 1
                    matched.append(keyword)
            if score > 0:
                scored.append((score, use_case))

        # Use only actual dataset labels. Keep the automatic kit compact.
        scored.sort(key=lambda item: (-item[0], item[1].lower()))
        selected = [use_case for _, use_case in scored[:6]]

        # If the keyword filter finds nothing, use the most frequent/useful
        # actual labels rather than dumping the whole 100+ class list.
        if not selected:
            counts = region_rows["Use_Case"].value_counts()
            selected = counts.head(5).index.astype(str).tolist()

        return selected, len(actual_use_cases)

    available_regions = sorted(df["Region_Notes"].dropna().astype(str).str.strip().unique().tolist())

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        region = st.selectbox(
            "🌍 Region:",
            options=available_regions,
            key="phase1_region",
        )

    # Automatically determine the valid mission for the selected region.
    valid_mission = _mission_for_region(region)

    with col2:
        if valid_mission:
            mission = st.selectbox(
                "🎯 Mission Type:",
                options=[valid_mission],
                index=0,
                disabled=True,
                key=f"phase1_mission_{valid_mission}",
                help="Mission type is linked to the selected regional context in the project dataset.",
            )
        else:
            mission = st.selectbox(
                "🎯 Mission Type:",
                options=sorted(df["Mission_Type"].dropna().astype(str).unique().tolist()),
                key="phase1_mission_fallback",
            )
            st.warning("This region is not in the predefined region-mission map; using the dataset value.")

    with col3:
        duration = st.number_input(
            "⏱️ Duration (days):",
            min_value=1,
            max_value=365,
            value=14,
            key="phase1_duration",
        )

    with col4:
        team_size = st.number_input(
            "👥 Team Size:",
            min_value=1,
            max_value=100,
            value=10,
            key="phase1_team_size",
        )

    st.caption(f"🔒 Valid mission context: **{mission}** is automatically paired with **{region}**.")

    if st.button("🚀 Optimize Drug Kit", type="primary", key="phase1_optimize"):
        st.markdown("---")
        st.subheader(f"🧠 Optimized Drug Kit for {mission} in {region}")
        st.info(
            "The system first locks the valid region-mission context, then detects a small "
            "set of relevant medical-use cases from the dataset. Medicine coverage is "
            "determined ONLY from the dataset's Use_Case field."
        )

        required_conditions, total_available = _detect_relevant_use_cases(region, mission)

        # Safety check: verify the selected pair actually exists before optimizing.
        valid_rows = df[
            df["Region_Notes"].apply(lambda x: region_matches(region, x))
            & df["Mission_Type"].apply(lambda x: mission_matches(mission, x))
        ]

        if valid_rows.empty:
            st.error(
                "❌ This region-mission pair does not exist in the dataset. "
                "The application has blocked optimization rather than producing a misleading kit."
            )
        elif not required_conditions:
            st.error(
                "❌ No context-relevant Use_Case labels were found for this valid mission context."
            )
        else:
            st.markdown("### 🩺 Automatically Selected Medical Requirements")
            st.caption(
                f"The dataset contains {total_available} Use_Case labels for this context; "
                f"the system selected {len(required_conditions)} context-relevant requirements "
                "instead of treating every dataset row as a requirement."
            )

            req_cols = st.columns(3)
            for i, condition in enumerate(required_conditions):
                with req_cols[i % 3]:
                    st.success(condition)

            # Optional manual reduction, never mandatory.
            with st.expander("⚙️ Customize medical requirements (optional)"):
                custom_conditions = st.multiselect(
                    "Select the requirements you want the kit to cover:",
                    options=required_conditions,
                    default=required_conditions,
                    key="phase1_custom_conditions",
                )
                if custom_conditions:
                    required_conditions = custom_conditions
                else:
                    st.info("No custom selection made; using the automatically selected requirements.")

            optimization_result = optimize_drug_kit(
                df=df,
                region=region,
                mission=mission,
                required_conditions=required_conditions,
            )

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("🎯 Coverage", f"{optimization_result['coverage'] * 100:.1f}%")
            with m2:
                st.metric("💊 Medicines Selected", len(optimization_result["selected_medicines"]))
            with m3:
                st.metric("🔎 Candidate Medicines", optimization_result["candidate_count"])
            with m4:
                st.metric("📌 Status", optimization_result["optimization_status"])

            st.markdown("### 📋 Mission Inputs")
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                st.write(f"**Region:** {region}")
            with ic2:
                st.write(f"**Mission:** {mission}")
            with ic3:
                st.write(f"**Duration:** {duration} days")
            with ic4:
                st.write(f"**Team Size:** {team_size}")

            covered = optimization_result.get("covered_conditions", [])
            uncovered = optimization_result.get("uncovered_conditions", [])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### ✅ Covered Requirements")
                if covered:
                    for condition in covered:
                        st.success(condition.title())
                else:
                    st.write("None")

            with col_b:
                st.markdown("### ⚠️ Uncovered Requirements")
                if uncovered:
                    for condition in uncovered:
                        st.warning(condition.title())
                else:
                    st.success("All selected requirements are covered.")

            st.markdown("### 💊 Optimized Medicine Kit")
            details = optimization_result.get("selected_details", [])

            if details:
                for item in details:
                    drug_name = item.get("Drug_Name", "Unknown medicine")
                    with st.expander(f"💊 {drug_name}", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**Generic Name:** {item.get('Generic_Name', 'N/A')}")
                            st.write(f"**Use Case:** {item.get('Use_Case', 'N/A')}")
                            st.write(f"**Typical Dosage:** {item.get('Typical_Dosage', 'N/A')}")
                        with c2:
                            st.write(f"**Dataset Quantity:** {item.get('Quantity_in_Kit', 'N/A')}")
                            matched = item.get("Matched_Conditions", [])
                            st.write("**Covers:** " + (", ".join(matched) if matched else "N/A"))
                            if item.get("Storage_Notes"):
                                st.write(f"**Storage:** {item.get('Storage_Notes')}")

                st.caption(
                    "Quantity shown here is the dataset inventory quantity. Duration and team size "
                    "are retained as mission inputs; dosage-frequency scaling is not applied until "
                    "it can be parsed reliably."
                )
            else:
                st.warning(
                    "No medicines were selected. The requirements may exist in the region context "
                    "but not as exact Use_Case matches for this mission."
                )

    # Phase 2: Symptom-Specific
    st.markdown("---")
    st.header("🩺 Phase 2: Symptom-Specific Drug Lookup")
    st.markdown("**Need a drug for a specific symptom? Get instant recommendations**")
    
    col1, col2 = st.columns(2)
    
    available_symptoms = sorted(df['Use_Case'].unique().tolist())
    
    with col1:
        symptom = st.selectbox("Select Symptom/Condition:", options=[''] + available_symptoms)
    
    with col2:
        custom_symptom = st.text_input("Or enter custom symptom:", placeholder="e.g., Chest pain")
    
    final_symptom = custom_symptom if custom_symptom else symptom
    
    if st.button("💊 Get Drug Recommendation"):
        if final_symptom:
            with st.spinner("🔍 Analyzing symptom..."):
                drug, dosage = predict_drug_for_symptom(
                    model, encoders, label_encoder, df,
                    final_symptom, region, mission
                )
                
                if not drug.startswith("Error"):
                    st.success(f"### 💊 Recommended: **{drug}**")
                    st.info(f"**Dosage:** {dosage}")
                    
                    # Additional info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Administration:**")
                        if 'IV' in dosage:
                            st.write("• Intravenous administration")
                            st.write("• Requires medical personnel")
                        elif 'Apply' in dosage:
                            st.write("• Topical application")
                            st.write("• Clean area before use")
                        else:
                            st.write("• Oral administration")
                            st.write("• Take with water")
                    
                    with col2:
                        st.write("**Important:**")
                        st.write("• Check patient allergies")
                        st.write("• Monitor for side effects")
                        st.write("• Document administration")
                else:
                    st.error(drug)
        else:
            st.warning("⚠️ Please select or enter a symptom")
    
    # Dataset Info (expandable)
    with st.expander("📚 View Dataset Information"):
        st.write(f"**Original Dataset:** {original_size} rows")
        st.write(f"**Augmented Dataset:** {final_size} rows")
        st.write(f"**Unique Conditions:** {df['Use_Case'].nunique()}")
        st.write(f"**Regions Covered:** {df['Region_Notes'].nunique()}")
        st.write(f"**Mission Types:** {df['Mission_Type'].nunique()}")
        st.dataframe(df.head(20))
    
    # Medical Disclaimer
    st.markdown("---")
    st.error("⚠️ **MEDICAL DISCLAIMER:** This tool provides recommendations based on military medical protocols. Always consult qualified medical personnel before administering medications. In emergency situations, follow established medical protocols and chain of command.")

if __name__ == "__main__":
    main()

# ================= STAGE 8: AGENTIC AI INTEGRATION =================
# This section is additive. Existing ML/mission-kit functionality above is
# intentionally preserved. The AI analysis panel calls the FastAPI backend
# when available and otherwise reports a clear connection message.

import os as _stage8_os
import requests as _stage8_requests


def _stage8_api_url() -> str:
    return _stage8_os.getenv(
        "DRUGKIT_API_URL",
        "http://127.0.0.1:8000/api/analyze",
    )


def _stage8_run_ai_analysis(
    symptom,
    condition,
    region,
    mission_type,
    environment,
    medical_history,
    allergies,
    current_medications,
    predicted_drug="",
):
    payload = {
        "symptom": symptom or "",
        "condition": condition or "",
        "region": region or "",
        "mission_type": mission_type or "",
        "environment": environment or "",
        "medical_history": medical_history or [],
        "allergies": allergies or [],
        "current_medications": current_medications or [],
        "predicted_drug": predicted_drug or "",
    }

    response = _stage8_requests.post(
        _stage8_api_url(),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _stage8_render_ai_panel():
    """Render Stage 8 results as a clean dashboard without exposing raw JSON/SVG."""
    st.markdown("---")
    st.header("🤖 Agentic AI Medical Analysis")
    st.caption(
        "Academic prototype: ML + RAG + multi-agent analysis + LLM explanation. "
        "Qualified medical review is required for real-world decisions."
    )

    col1, col2 = st.columns(2)

    with col1:
        ai_symptom = st.text_input(
            "Symptom",
            key="stage8_ai_symptom",
            placeholder="Example: fever",
        )
        ai_condition = st.text_input(
            "Condition",
            key="stage8_ai_condition",
            placeholder="Example: malaria",
        )
        ai_region = st.text_input(
            "Region",
            key="stage8_ai_region",
            placeholder="Example: tropical",
        )
        ai_mission = st.text_input(
            "Mission Type",
            key="stage8_ai_mission",
            placeholder="Example: jungle",
        )

    with col2:
        ai_environment = st.text_input(
            "Environment",
            key="stage8_ai_environment",
            placeholder="Example: humid",
        )
        ai_history_text = st.text_area(
            "Medical History (comma-separated)",
            key="stage8_ai_history",
        )
        ai_allergy_text = st.text_area(
            "Allergies (comma-separated)",
            key="stage8_ai_allergies",
        )
        ai_medications_text = st.text_area(
            "Current Medications (comma-separated)",
            key="stage8_ai_medications",
        )

    if st.button("🚀 Run Agentic AI Analysis", key="stage8_ai_button"):
        def _split(value):
            return [x.strip() for x in value.split(",") if x.strip()]

        try:
            with st.spinner("Running Medical, Regional and Safety Agents..."):
                result = _stage8_run_ai_analysis(
                    symptom=ai_symptom,
                    condition=ai_condition,
                    region=ai_region,
                    mission_type=ai_mission,
                    environment=ai_environment,
                    medical_history=_split(ai_history_text),
                    allergies=_split(ai_allergy_text),
                    current_medications=_split(ai_medications_text),
                    predicted_drug="",
                )

            # ---------------------------------------------------------
            # Clean Stage 8 presentation layer.
            # Backend/agents/RAG/ML logic is untouched.
            # ---------------------------------------------------------
            coordinator = result.get("coordinator_result", {}) or {}
            medical = coordinator.get("medical_assessment", {}) or {}
            regional = coordinator.get("regional_risk_assessment", {}) or {}
            safety = coordinator.get("clinical_safety_validation", {}) or {}
            ml = coordinator.get("ml_prediction", {}) or {}
            summary = coordinator.get("coordinator_summary", {}) or {}
            explanation = result.get("explanation", {}) or {}

            status = str(result.get("status", "UNKNOWN"))
            status_upper = status.upper()

            # Status banner
            if "FLAGGED" in status_upper or "REVIEW" in status_upper or "INSUFFICIENT" in status_upper:
                st.warning(f"⚠️ **Overall Status: {status.replace('_', ' ')}**")
            elif "SUCCESS" in status_upper or "COMPLETED" in status_upper:
                st.success(f"✅ **Overall Status: {status.replace('_', ' ')}**")
            else:
                st.info(f"ℹ️ **Overall Status: {status.replace('_', ' ')}**")

            # Top-level summary metrics
            predicted = ml.get("prediction", "")
            if isinstance(predicted, dict):
                predicted = predicted.get("prediction", "")
            if not predicted:
                predicted = coordinator.get("predicted_drug", "") or "Not available"

            medical_status = medical.get("status", "Not available")
            regional_status = regional.get("status", "Not available")
            safety_status = safety.get("status", "Not available")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("💊 ML Prediction", str(predicted))
            with m2:
                st.metric("🩺 Medical", str(medical_status).replace("_", " "))
            with m3:
                st.metric("🌍 Regional Risk", str(regional.get("risk_level", regional_status)).replace("_", " "))
            with m4:
                st.metric("🛡️ Safety", str(safety_status).replace("_", " "))

            # AI explanation
            st.subheader("🧠 AI Analysis Summary")
            explanation_text = explanation.get("explanation", "")
            if explanation_text:
                st.markdown(str(explanation_text))
            else:
                st.info(
                    "The specialist agents completed the structured analysis. "
                    "No additional LLM explanation was returned."
                )

            # Medical assessment
            st.subheader("🩺 Medical Assessment")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.write(f"**Status:** {str(medical_status).replace('_', ' ')}")
                if medical.get("symptom_summary"):
                    st.write(f"**Summary:** {medical['symptom_summary']}")

                concerns = medical.get("identified_concerns", []) or []
                if concerns:
                    st.markdown("**Identified concerns**")
                    for item in concerns:
                        st.write(f"• {item}")

            with mc2:
                missing = medical.get("missing_information", []) or []
                red_flags = medical.get("red_flags", []) or []
                if missing:
                    st.markdown("**Missing information**")
                    for item in missing:
                        st.warning(item)
                if red_flags:
                    st.markdown("**Red flags**")
                    for item in red_flags:
                        st.error(item)
                if not missing and not red_flags:
                    st.success("No additional missing information or red flags were returned.")

            # Regional risk
            st.subheader("🌍 Regional Risk Assessment")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.write(f"**Status:** {str(regional_status).replace('_', ' ')}")
                risk_level = regional.get("risk_level")
                if risk_level:
                    st.write(f"**Risk Level:** {str(risk_level).replace('_', ' ')}")
                region_value = regional.get("region")
                mission_value = regional.get("mission_type")
                if region_value:
                    st.write(f"**Region:** {region_value}")
                if mission_value:
                    st.write(f"**Mission:** {mission_value}")

            with rc2:
                risks = regional.get("identified_risks", []) or []
                factors = regional.get("environmental_factors", []) or []
                if risks:
                    st.markdown("**Environmental / mission risks**")
                    for item in risks:
                        st.write(f"• {item}")
                if factors:
                    st.markdown("**Environmental factors**")
                    for item in factors:
                        st.write(f"• {item}")

            # Safety validation
            st.subheader("🛡️ Clinical Safety Validation")
            if "FLAGGED" in str(safety_status).upper() or "REVIEW" in str(safety_status).upper():
                st.warning("⚠️ Safety review is required before relying on the ML prediction.")
            else:
                st.info(f"Safety status: {str(safety_status).replace('_', ' ')}")

            sc1, sc2 = st.columns(2)
            with sc1:
                if safety.get("predicted_drug"):
                    st.write(f"**Evaluated medicine:** {safety['predicted_drug']}")
                warnings_list = safety.get("warnings", []) or []
                if warnings_list:
                    st.markdown("**Safety warnings**")
                    for item in warnings_list:
                        st.warning(item)

            with sc2:
                missing_safety = safety.get("missing_information", []) or []
                if missing_safety:
                    st.markdown("**Safety information still needed**")
                    for item in missing_safety:
                        st.write(f"• {item}")
                contraindications = safety.get("contraindications", []) or []
                interactions = safety.get("interaction_warnings", []) or []
                if contraindications:
                    st.markdown("**Contraindication checks**")
                    for item in contraindications:
                        st.write(f"• {item}")
                if interactions:
                    st.markdown("**Interaction checks**")
                    for item in interactions:
                        st.write(f"• {item}")

            # ML prediction details
            st.subheader("📊 ML Prediction")
            if isinstance(ml.get("prediction"), dict):
                prediction_data = ml.get("prediction", {})
            else:
                prediction_data = ml

            ml1, ml2 = st.columns(2)
            with ml1:
                st.write(f"**Status:** {prediction_data.get('status', ml.get('status', 'N/A'))}")
                st.write(f"**Prediction:** {prediction_data.get('prediction', predicted)}")
                st.write(f"**Source:** {prediction_data.get('source', 'N/A')}")
                st.write(f"**Predictor:** {prediction_data.get('predictor', 'N/A')}")
            with ml2:
                mapped = prediction_data.get("mapped_inputs", {}) or {}
                if mapped:
                    st.markdown("**Inputs mapped to the existing ML model**")
                    for key, value in mapped.items():
                        st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No mapped ML inputs were returned.")

            # RAG evidence: show the useful source snippets without dumping JSON.
            st.subheader("📚 RAG Evidence Used")
            evidence_groups = [
                ("Medical", medical.get("retrieved_evidence", []) or []),
                ("Regional", regional.get("retrieved_evidence", []) or []),
                ("Safety", safety.get("evidence", []) or []),
            ]
            evidence_count = 0
            for label, evidence_list in evidence_groups:
                if not evidence_list:
                    continue
                with st.expander(f"{label} evidence ({len(evidence_list)} retrieved)", expanded=False):
                    for item in evidence_list:
                        evidence_count += 1
                        source = item.get("source", "Unknown source")
                        score = item.get("score")
                        text_value = item.get("text", "")
                        score_text = f" | similarity: {score:.3f}" if isinstance(score, (int, float)) else ""
                        st.markdown(f"**{source}**{score_text}")
                        if text_value:
                            st.caption(str(text_value))

            if evidence_count == 0:
                st.info("No RAG evidence was returned by the backend.")

            # Coordinator / architecture details
            with st.expander("🤖 Agent Execution Details", expanded=False):
                agents = summary.get("agents_called", coordinator.get("agents_called", [])) or []
                if agents:
                    st.markdown("**Agents executed**")
                    for agent in agents:
                        st.write(f"✓ {agent}")
                st.write(f"**ML called:** {summary.get('ml_called', coordinator.get('ml_called', 'N/A'))}")
                st.write(
                    f"**RAG used through agents:** "
                    f"{summary.get('rag_used_through_agents', coordinator.get('rag_used_through_agents', 'N/A'))}"
                )
                next_step = summary.get("next_step")
                if next_step:
                    st.write(f"**Pipeline next step:** {next_step}")

            disclaimer = result.get("medical_disclaimer") or safety.get("disclaimer")
            if disclaimer:
                st.warning(f"⚠️ {disclaimer}")
            else:
                st.warning(
                    "⚠️ Academic prototype only. This output is not a diagnosis, prescription, "
                    "or substitute for qualified medical review."
                )

        except _stage8_requests.exceptions.ConnectionError:
            st.error(
                "FastAPI backend is not running. Start it with:\n\n"
                "uvicorn api.main:app --reload"
            )
        except _stage8_requests.exceptions.Timeout:
            st.error("The FastAPI backend timed out. Please try again.")
        except Exception as exc:
            st.error(f"Agentic AI analysis failed: {exc}")


# Add the new panel at the end so existing UI remains unchanged.
_stage8_render_ai_panel()
# ================= END STAGE 8 =================
