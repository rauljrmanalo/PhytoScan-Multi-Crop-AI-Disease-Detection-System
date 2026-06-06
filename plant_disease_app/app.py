from flask import Flask, render_template, request, jsonify
import os
import json
import random
import time
from werkzeug.utils import secure_filename
from models.model_loader import load_all_models, get_classifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load all trained VGG16 models at startup.
# Crops without a model file automatically fall back to mock_classify().
load_all_models()

# ─────────────────────────────────────────────────────────────
# DISEASE DATABASE
# Replace the mock classifier below with your real ML model calls
# ─────────────────────────────────────────────────────────────
DISEASE_DB = {
    # ── RICE ── 4 classes matching Kaggle dataset (loki4514/rice-leaf-diseases-detection)
    # Folder names in dataset → disease IDs here:
    #   "Bacterial Blight" → "bacterial_blight"
    #   "Blast"            → "rice_blast"
    #   "Brown Spot"       → "brown_spot"
    #   "Tungro"           → "tungro"
    "rice": {
        "diseases": [
            {
                "id": "bacterial_leaf_blight",
                "name": "Bacterial Leaf Blight",
                "pathogen": "Xanthomonas oryzae pv. oryzae",
                "severity": "High",
                "confidence_range": (60, 92),
                "symptoms": "Water-soaked to yellowish stripes on leaf margins that turn white to yellow; seedling wilting (kresek symptom) in severe cases.",
                "treatment": [
                    "Apply copper-based bactericides: Copper oxychloride at 3 g/L water",
                    "Use Streptomycin sulfate 90% + Tetracycline 10% at 1 g per 10L",
                    "Remove and destroy severely infected plant material",
                    "Drain fields and avoid irrigation during active outbreak"
                ],
                "prevention": [
                    "Plant resistant varieties: IRBB series, IR64, MR219",
                    "Avoid physical injury to plants during transplanting",
                    "Use balanced NPK fertilization — excess nitrogen worsens susceptibility",
                    "Practice field sanitation: remove stubble and crop debris",
                    "Avoid waterlogging and deep flooding in susceptible varieties"
                ],
                "color": "#f39c12"
            },
            {
                "id": "brown_spot",
                "name": "Brown Spot",
                "pathogen": "Cochliobolus miyabeanus",
                "severity": "Medium",
                "confidence_range": (55, 88),
                "symptoms": "Oval to circular brown spots (1–14 mm) with yellow halos on leaves. Grain discoloration (pecky rice) reduces market value significantly.",
                "treatment": [
                    "Apply Mancozeb 75 WP (2.5 g/L) or Propiconazole 25 EC (1 mL/L)",
                    "Use Iprodione 50 WP (1.5 g/L) for moderate-severe infections",
                    "Seed treatment with Carboxin + Thiram at 2 g/kg seed",
                    "Foliar spray at tillering and again at panicle initiation stages"
                ],
                "prevention": [
                    "Maintain adequate soil potassium and silicon — deficiency worsens susceptibility",
                    "Use certified, treated seeds from reputable source",
                    "Avoid water stress, especially during grain filling stage",
                    "Apply organic matter (compost) to improve soil nutrient retention",
                    "Correct soil pH to 5.5–6.5 to maximize nutrient availability"
                ],
                "color": "#8e44ad"
            },
            {
                "id": "healthy",
                "name": "Healthy Plant",
                "pathogen": "None",
                "severity": "None",
                "confidence_range": (80, 99),
                "symptoms": "Vigorous green growth, absence of lesions, spots, or discoloration. Panicles develop normally with proper grain filling.",
                "treatment": [
                    "No treatment required",
                    "Continue standard monitoring protocols"
                ],
                "prevention": [
                    "Maintain balanced fertilization",
                    "Ensure proper water management based on growth stage",
                    "Keep fields free of weeds to reduce competition",
                    "Regularly scout for early signs of pests or diseases"
                ],
                "color": "#27ae60"
            },
            {
                "id": "leaf_blast",
                "name": "Leaf Blast",
                "pathogen": "Magnaporthe oryzae",
                "severity": "High",
                "confidence_range": (65, 95),
                "symptoms": "Diamond-shaped lesions with gray or white centers and dark brown borders on leaves, nodes, and panicles. Can cause neck rot and total panicle loss.",
                "treatment": [
                    "Apply Tricyclazole (0.6 g/L) or Isoprothiolane (1.5 mL/L) immediately",
                    "Spray Azoxystrobin 23 SC (1 mL/L) for systemic protection",
                    "Repeat application after 10–14 days if disease pressure is high",
                    "Drain fields for 3–5 days to reduce leaf wetness and humidity",
                    "Remove and destroy all infected debris after harvest"
                ],
                "prevention": [
                    "Plant blast-resistant varieties: IR64, IRRI 6, MR219, Ciherang",
                    "Avoid excessive nitrogen — split applications instead of single large dose",
                    "Maintain proper plant spacing (20×20 cm) for air circulation",
                    "Use certified disease-free seeds and seed treatment",
                    "Apply silicon fertilizer to strengthen cell walls against fungal penetration"
                ],
                "color": "#e74c3c"
            },
            {
                "id": "leaf_scald",
                "name": "Leaf Scald",
                "pathogen": "Microdochium oryzae",
                "severity": "Medium",
                "confidence_range": (50, 85),
                "symptoms": "Zonate lesions that start at the leaf tips or edges and spread downwards, often appearing as alternating light and dark bands (scalded look).",
                "treatment": [
                    "Apply Propiconazole 25 EC (1 mL/L) or Hexaconazole 5 SC (2 mL/L)",
                    "Treat seeds with Captan or Thiram before sowing",
                    "Foliar applications during late tillering to booting stages if symptoms appear"
                ],
                "prevention": [
                    "Use clean, pathogen-free seeds",
                    "Avoid excessive application of nitrogen fertilizers",
                    "Destroy crop residues after harvest to reduce overwintering fungi",
                    "Plant at wider spacing to improve canopy ventilation"
                ],
                "color": "#d35400"
            },
            {
                "id": "narrow_brown_spot",
                "name": "Narrow Brown Leaf Spot",
                "pathogen": "Cercospora janseana",
                "severity": "Low to Medium",
                "confidence_range": (60, 88),
                "symptoms": "Short, linear, narrow brown lesions running parallel to the leaf veins. Often appears late in the season.",
                "treatment": [
                    "Apply Propiconazole 25 EC (1 mL/L) at the booting and heading stages",
                    "Apply a foliar fungicide if the disease pressure is unusually high early in the season"
                ],
                "prevention": [
                    "Use resistant or tolerant rice varieties",
                    "Plant early to avoid the peak disease pressure later in the season",
                    "Maintain proper soil fertility; potassium deficiency can exacerbate symptoms"
                ],
                "color": "#a0522d"
            },
            {
                "id": "neck_blast",
                "name": "Neck Blast",
                "pathogen": "Magnaporthe oryzae",
                "severity": "Critical",
                "confidence_range": (70, 96),
                "symptoms": "Brown to black lesions form on the neck of the panicle, causing it to weaken and break over ('blanking' or 'whiteheads'). Complete loss of the panicle is common.",
                "treatment": [
                    "Apply Tricyclazole 75 WP (0.6 g/L) immediately at late booting to early heading",
                    "Application must occur before the panicle fully emerges for maximum efficacy",
                    "Use Azoxystrobin + Difenoconazole for broad-spectrum control if applied preventatively"
                ],
                "prevention": [
                    "Plant blast-resistant varieties suited for your region",
                    "Avoid late planting to escape highly favorable weather for the fungus",
                    "Do not over-apply nitrogen fertilizers, especially at panicle initiation",
                    "Manage water levels to avoid drought stress during the reproductive phase"
                ],
                "color": "#c0392b"
            },
            {
                "id": "rice_hispa",
                "name": "Rice Hispa",
                "pathogen": "Dicladispa armigera (Insect Pest)",
                "severity": "Medium to High",
                "confidence_range": (65, 92),
                "symptoms": "Adults scrape the upper surface of leaves, leaving white parallel streaks. Grubs mine into the leaf tissue, causing irregular translucent white patches. Leaves may eventually wither.",
                "treatment": [
                    "Spray Chlorpyrifos 20 EC (2.5 mL/L) or Quinalphos 25 EC (2 mL/L)",
                    "Apply Cypermethrin 10 EC (1 mL/L) for heavy adult infestations",
                    "Sweep nets can be used physically to collect and destroy adult beetles"
                ],
                "prevention": [
                    "Clip the top 2-3 cm of leaves before transplanting to remove eggs",
                    "Avoid excessive nitrogen which attracts the pest",
                    "Maintain weed-free bunds as weeds serve as alternate hosts",
                    "Encourage natural predators like spiders and eulophid wasps"
                ],
                "color": "#34495e"
            },
            {
                "id": "sheath_blight",
                "name": "Sheath Blight",
                "pathogen": "Rhizoctonia solani",
                "severity": "High",
                "confidence_range": (70, 94),
                "symptoms": "Oval, greenish-gray, water-soaked spots on leaf sheaths near the water line. Lesions expand and develop a bleached center with a dark brown border, moving upward.",
                "treatment": [
                    "Apply Validamycin 3 L (2 mL/L) or Hexaconazole 5 SC (2 mL/L)",
                    "Spray targeting the base of the plant where symptoms originate",
                    "Apply fungicides at the panicle initiation or booting stages if disease is progressing"
                ],
                "prevention": [
                    "Use wider plant spacing to reduce canopy humidity",
                    "Avoid high nitrogen rates and dense crop stands",
                    "Manage weeds on bunds that may harbor the fungus",
                    "Deep plow fields after harvest to bury the sclerotia (survival structures)"
                ],
                "color": "#16a085"
            },
            {
                "id": "tungro",
                "name": "Rice Tungro Disease",
                "pathogen": "Rice Tungro Spherical Virus (RTSV) + Rice Tungro Bacilliform Virus (RTBV)",
                "severity": "High",
                "confidence_range": (60, 90),
                "symptoms": "Yellow-orange leaf discoloration starting from the tip; stunted plant growth; reduced tillering; panicles may be absent or sterile. Often mistaken for nitrogen deficiency.",
                "treatment": [
                    "No direct cure for viral infection once established",
                    "Control green leafhopper vector: Buprofezin 25 WP (1 g/L) or Imidacloprid 17.8 SL (0.3 mL/L)",
                    "Remove and destroy infected plants promptly to reduce inoculum source",
                    "Apply carbofuran granules at transplanting to suppress leafhopper early",
                    "Avoid replanting in fields with active tungro outbreak for that season"
                ],
                "prevention": [
                    "Plant tungro-resistant varieties: TKM6, IR36, IR64, Matatag 9",
                    "Synchronize planting with neighboring farms to break leafhopper cycle",
                    "Use yellow sticky traps (25/ha) to monitor leafhopper populations",
                    "Avoid planting near ratoon rice or old infected fields",
                    "Maintain clean bunds and field borders to eliminate leafhopper refuges"
                ],
                "color": "#e67e22"
            }
        ],
        "food_security": {
            "global_production_mt": 520,
            "countries_dependent": 3500000000,
            "caloric_contribution_pct": 21,
            "annual_loss_pct": 30,
            "annual_loss_billion_usd": 75,
            "key_stat": "Rice feeds more than half of the world's population",
            "region": "Asia (90% of production)",
            "yield_potential_t_ha": 10,
            "average_yield_t_ha": 4.5
        }
    },
    "corn": {
        "diseases": [
            {
                "id": "northern_blight",
                "name": "Northern Corn Leaf Blight",
                "pathogen": "Exserohilum turcicum",
                "severity": "High",
                "confidence_range": (65, 92),
                "symptoms": "Long, cigar-shaped grayish-green to tan lesions (2.5–15 cm) running parallel to leaf margins.",
                "treatment": [
                    "Apply Propiconazole (Tilt) at 0.5 L/ha",
                    "Use Azoxystrobin + Propiconazole at tasseling stage",
                    "Apply fungicide when 50% of plants show symptoms on lower leaves",
                    "Repeat application every 14 days in severe cases"
                ],
                "prevention": [
                    "Plant resistant hybrids: P3144, DK9108",
                    "Practice crop rotation with non-host crops",
                    "Plow under crop debris after harvest",
                    "Avoid overhead irrigation",
                    "Ensure proper plant spacing for air circulation"
                ],
                "color": "#e67e22"
            },
            {
                "id": "gray_leaf_spot",
                "name": "Gray Leaf Spot",
                "pathogen": "Cercospora zeae-maydis",
                "severity": "Medium",
                "confidence_range": (60, 88),
                "symptoms": "Rectangular, gray to tan lesions bounded by leaf veins; lesions run parallel to veins.",
                "treatment": [
                    "Apply Strobilurin-based fungicides (Headline, Quadris)",
                    "Use Trifloxystrobin + Propiconazole at VT/R1 growth stages",
                    "Foliar applications most effective at silking stage"
                ],
                "prevention": [
                    "Use tolerant hybrids",
                    "Rotate with soybean or other non-grass crops",
                    "Reduce surface crop residue through tillage",
                    "Avoid continuous corn planting"
                ],
                "color": "#95a5a6"
            },
            {
                "id": "common_rust",
                "name": "Common Rust",
                "pathogen": "Puccinia sorghi",
                "severity": "Medium",
                "confidence_range": (70, 95),
                "symptoms": "Small, circular to elongated brick-red pustules scattered over both leaf surfaces.",
                "treatment": [
                    "Apply Mancozeb 75 WP (2 kg/ha) or Zineb 75 WP",
                    "Use Propiconazole (1 mL/L) for severe infections",
                    "Early application more effective before pustule rupture"
                ],
                "prevention": [
                    "Plant rust-resistant varieties",
                    "Early planting to avoid high-risk periods",
                    "Monitor fields weekly during cool/humid weather",
                    "Destroy volunteer corn plants"
                ],
                "color": "#c0392b"
            },
            {
                "id": "healthy",
                "name": "Healthy Plant",
                "pathogen": "None",
                "severity": "None",
                "confidence_range": (80, 99),
                "symptoms": "No disease symptoms. Vibrant dark green leaves with no lesions or discoloration.",
                "treatment": ["No treatment required"],
                "prevention": [
                    "Maintain current management practices",
                    "Scout fields every 7-10 days",
                    "Keep good records for season comparison"
                ],
                "color": "#27ae60"
            }
        ],
        "food_security": {
            "global_production_mt": 1200,
            "countries_dependent": 1000000000,
            "caloric_contribution_pct": 6,
            "annual_loss_pct": 20,
            "annual_loss_billion_usd": 60,
            "key_stat": "Corn is the most produced crop globally by volume",
            "region": "Americas (50% of production)",
            "yield_potential_t_ha": 15,
            "average_yield_t_ha": 5.8
        }
    },
    "banana": {
        "diseases": [
            {
                "id": "panama_disease",
                "name": "Panama Disease (Fusarium Wilt)",
                "pathogen": "Fusarium oxysporum f.sp. cubense",
                "severity": "Critical",
                "confidence_range": (70, 95),
                "symptoms": "Yellowing of older leaves, internal browning of pseudostem, plant collapse; TR4 strain affects Cavendish.",
                "treatment": [
                    "No chemical cure available — prevention is critical",
                    "Remove and destroy infected plants including corms",
                    "Apply lime to affected soil to raise pH above 7",
                    "Use biological control: Trichoderma spp. soil drench",
                    "Quarantine affected areas strictly"
                ],
                "prevention": [
                    "Use certified disease-free tissue culture planting material",
                    "Implement strict biosecurity protocols",
                    "Disinfect tools with 70% ethanol between plants",
                    "Avoid movement of soil from infected areas",
                    "Plant resistant varieties: FHIA-01, FHIA-17, Goldfinger"
                ],
                "color": "#e74c3c"
            },
            {
                "id": "black_sigatoka",
                "name": "Black Sigatoka",
                "pathogen": "Mycosphaerella fijiensis",
                "severity": "High",
                "confidence_range": (65, 90),
                "symptoms": "Small pale yellow streaks on leaves that develop into dark brown/black necrotic lesions with yellow halos.",
                "treatment": [
                    "Apply systemic fungicides: Propiconazole or Trifloxystrobin",
                    "Alternate between chemical groups to prevent resistance",
                    "Apply every 3 weeks during wet season",
                    "Remove severely infected leaves (leaf surgery)"
                ],
                "prevention": [
                    "Regular deleafing of infected leaves",
                    "Maintain good drainage in plantations",
                    "Use resistant varieties where available",
                    "Implement integrated disease management (IDM)",
                    "Monitor using Stover's disease assessment scale"
                ],
                "color": "#2c3e50"
            },
            {
                "id": "bunchy_top",
                "name": "Banana Bunchy Top",
                "pathogen": "Banana bunchy top virus (BBTV)",
                "severity": "High",
                "confidence_range": (60, 88),
                "symptoms": "Dark green streaks on leaf midribs and petioles; leaves narrow and erect giving a bunchy appearance.",
                "treatment": [
                    "No cure — infected plants must be destroyed",
                    "Inject infected plants with 2,4-D herbicide to kill quickly",
                    "Destroy plants before aphids can spread virus further",
                    "Report outbreaks to local agricultural authority"
                ],
                "prevention": [
                    "Source planting material only from virus-indexed sources",
                    "Control banana aphid (Pentalonia nigronervosa) vector with insecticides",
                    "Establish new plantations far from infected areas",
                    "Implement regular field scouting programs",
                    "Use reflective mulches to deter aphid landing"
                ],
                "color": "#8e44ad"
            },
            {
                "id": "healthy",
                "name": "Healthy Plant",
                "pathogen": "None",
                "severity": "None",
                "confidence_range": (80, 99),
                "symptoms": "Uniform dark green leaves with no streaks, lesions, or chlorosis.",
                "treatment": ["No treatment required"],
                "prevention": [
                    "Continue regular leaf pruning and field sanitation",
                    "Monitor for early aphid infestations",
                    "Maintain balanced nutrition"
                ],
                "color": "#27ae60"
            }
        ],
        "food_security": {
            "global_production_mt": 124,
            "countries_dependent": 400000000,
            "caloric_contribution_pct": 3,
            "annual_loss_pct": 35,
            "annual_loss_billion_usd": 25,
            "key_stat": "Banana is the 4th most important food crop in developing countries",
            "region": "Tropical regions (India leads production)",
            "yield_potential_t_ha": 70,
            "average_yield_t_ha": 20
        }
    },
    "chilli": {
        "diseases": [
            {
                "id": "anthracnose",
                "name": "Anthracnose",
                "pathogen": "Colletotrichum capsici",
                "severity": "High",
                "confidence_range": (65, 92),
                "symptoms": "Dark, sunken, water-soaked lesions on fruits; lesions expand with salmon-pink spore masses.",
                "treatment": [
                    "Apply Carbendazim 50 WP (1 g/L) or Mancozeb 75 WP (2.5 g/L)",
                    "Use Azoxystrobin + Difenoconazole for severe cases",
                    "Remove and destroy infected fruits immediately",
                    "Spray every 7-10 days during monsoon season"
                ],
                "prevention": [
                    "Use treated seeds: hot water treatment at 52°C for 30 min",
                    "Avoid overhead irrigation",
                    "Harvest fruits at proper maturity",
                    "Use resistant varieties: Indra, LCA 334",
                    "Maintain proper drainage and field hygiene"
                ],
                "color": "#e74c3c"
            },
            {
                "id": "leaf_curl",
                "name": "Chilli Leaf Curl Disease",
                "pathogen": "Chilli leaf curl virus (ChLCV) via Bemisia tabaci",
                "severity": "High",
                "confidence_range": (60, 88),
                "symptoms": "Upward or downward curling of leaves, thickening and crinkling; stunted growth and reduced yield.",
                "treatment": [
                    "No direct cure for viral infection",
                    "Control whitefly vector with Imidacloprid 17.8 SL (0.3 mL/L)",
                    "Use Thiamethoxam 25 WG for whitefly management",
                    "Remove and destroy infected plants to reduce inoculum"
                ],
                "prevention": [
                    "Use virus-free transplants",
                    "Install yellow sticky traps (20/ha) to monitor whiteflies",
                    "Use reflective silver mulch to repel whiteflies",
                    "Plant barrier crops of maize or sorghum around fields",
                    "Avoid planting near cucurbit or tomato crops"
                ],
                "color": "#f39c12"
            },
            {
                "id": "powdery_mildew",
                "name": "Powdery Mildew",
                "pathogen": "Leveillula taurica",
                "severity": "Medium",
                "confidence_range": (55, 85),
                "symptoms": "White powdery coating on lower leaf surface; corresponding yellow patches on upper surface; leaf drop.",
                "treatment": [
                    "Apply Wettable Sulfur 80 WP (3 g/L) or Triadimefon",
                    "Use Tebuconazole for moderate to severe infections",
                    "Spray in early morning or evening to prevent phytotoxicity"
                ],
                "prevention": [
                    "Ensure good air circulation through proper spacing",
                    "Avoid excessive nitrogen fertilization",
                    "Use resistant varieties where available",
                    "Apply neem oil (5 mL/L) as preventive spray"
                ],
                "color": "#bdc3c7"
            },
            {
                "id": "healthy",
                "name": "Healthy Plant",
                "pathogen": "None",
                "severity": "None",
                "confidence_range": (80, 99),
                "symptoms": "Deep green, flat leaves with no curling, spots, or powdery deposits.",
                "treatment": ["No treatment required"],
                "prevention": [
                    "Monitor for whitefly and aphid infestations weekly",
                    "Maintain optimal nutrition and irrigation",
                    "Practice crop rotation"
                ],
                "color": "#27ae60"
            }
        ],
        "food_security": {
            "global_production_mt": 42,
            "countries_dependent": 2000000000,
            "caloric_contribution_pct": 1,
            "annual_loss_pct": 40,
            "annual_loss_billion_usd": 10,
            "key_stat": "Chilli is the world's most consumed spice by weight",
            "region": "Asia (India, China lead production)",
            "yield_potential_t_ha": 25,
            "average_yield_t_ha": 8
        }
    },
    "onion": {
        "diseases": [
            {
                "id": "purple_blotch",
                "name": "Purple Blotch",
                "pathogen": "Alternaria porri",
                "severity": "High",
                "confidence_range": (65, 92),
                "symptoms": "Small white lesions with purple centers on leaves and stalks; lesions enlarge with yellow halos.",
                "treatment": [
                    "Apply Mancozeb 75 WP (2.5 g/L) or Iprodione 50 WP (1 g/L)",
                    "Use Tebuconazole 25.9 EC (1 mL/L) for severe cases",
                    "Spray at 10-day intervals starting from early infection",
                    "Apply fungicides in evening to avoid leaf burn"
                ],
                "prevention": [
                    "Use certified disease-free seed or sets",
                    "Practice crop rotation (3-year cycle)",
                    "Avoid overhead irrigation; use drip irrigation",
                    "Plant onions in well-drained, raised beds",
                    "Remove and destroy crop debris after harvest"
                ],
                "color": "#9b59b6"
            },
            {
                "id": "downy_mildew",
                "name": "Downy Mildew",
                "pathogen": "Peronospora destructor",
                "severity": "Medium",
                "confidence_range": (60, 88),
                "symptoms": "Pale green to yellow patches on leaves; violet-gray fuzzy growth (sporulation) in humid conditions; leaves collapse.",
                "treatment": [
                    "Apply Metalaxyl + Mancozeb (Ridomil Gold) at 2.5 g/L",
                    "Use Fosetyl-Al (Aliette) 80 WP at 2 g/L",
                    "Spray every 7 days during wet/cool conditions"
                ],
                "prevention": [
                    "Use tolerant varieties",
                    "Ensure wide row spacing for air circulation",
                    "Avoid irrigation that wets foliage",
                    "Apply protective fungicides before disease onset in wet seasons",
                    "Rotate with non-allium crops for 3+ years"
                ],
                "color": "#2980b9"
            },
            {
                "id": "fusarium_basal_rot",
                "name": "Fusarium Basal Rot",
                "pathogen": "Fusarium oxysporum f.sp. cepae",
                "severity": "High",
                "confidence_range": (55, 85),
                "symptoms": "Pink to reddish-brown rot at bulb base; roots turn pink then brown; yellowing from leaf tips downward.",
                "treatment": [
                    "Apply Carbendazim as soil drench (1 g/L)",
                    "Use Trichoderma harzianum biofungicide as soil amendment",
                    "Remove and destroy infected plants to prevent spread",
                    "Drench planting sites with Mancozeb"
                ],
                "prevention": [
                    "Treat planting sets with Thiram or Captan before planting",
                    "Avoid planting in poorly drained soils",
                    "Use disease-free sets from certified sources",
                    "Apply Trichoderma to soil 2 weeks before planting",
                    "Practice long crop rotations (5+ years)"
                ],
                "color": "#e67e22"
            },
            {
                "id": "healthy",
                "name": "Healthy Plant",
                "pathogen": "None",
                "severity": "None",
                "confidence_range": (80, 99),
                "symptoms": "Upright, firm dark-green tubular leaves with no lesions, yellowing, or rot at base.",
                "treatment": ["No treatment required"],
                "prevention": [
                    "Continue good irrigation management",
                    "Monitor for thrips and other pest vectors",
                    "Maintain balanced fertilization program"
                ],
                "color": "#27ae60"
            }
        ],
        "food_security": {
            "global_production_mt": 105,
            "countries_dependent": 3000000000,
            "caloric_contribution_pct": 1,
            "annual_loss_pct": 25,
            "annual_loss_billion_usd": 8,
            "key_stat": "Onion is the 2nd most important horticultural crop worldwide",
            "region": "Asia (China, India account for 60%)",
            "yield_potential_t_ha": 60,
            "average_yield_t_ha": 19
        }
    }
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def mock_classify(crop, image_path):
    """Fallback mock — used only when no trained model exists for this crop."""
    time.sleep(1.0)
    diseases = DISEASE_DB[crop]["diseases"]
    disease = random.choice(diseases)
    confidence = random.uniform(*disease["confidence_range"])
    affected_area = 0.0 if disease["id"] == "healthy" else random.uniform(10, 75)
    return {
        "disease_id": disease["id"],
        "confidence": round(confidence, 1),
        "affected_area_pct": round(affected_area, 1),
        "source": "mock"
    }


def classify(crop, image_path):
    """
    Main entry point for inference.
    Uses real VGG16 model if loaded, otherwise falls back to mock.
    """
    classifier = get_classifier(crop)
    if classifier is not None:
        result = classifier.predict(image_path)
        result["source"] = "vgg16"
        return result
    return mock_classify(crop, image_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    crop = request.form.get('crop', '').lower()

    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if crop not in DISEASE_DB:
        return jsonify({"error": f"Unknown crop: {crop}"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ── Run classification ──────────────────────────────────
    result = classify(crop, filepath)
    disease_info = next(
        (d for d in DISEASE_DB[crop]["diseases"] if d["id"] == result["disease_id"]),
        DISEASE_DB[crop]["diseases"][-1]
    )
    food_sec = DISEASE_DB[crop]["food_security"]

    # ── Build disease prevalence stats for chart ────────────
    prevalence = {
        d["name"]: round(random.uniform(5, 45), 1)
        for d in DISEASE_DB[crop]["diseases"]
        if d["id"] != "healthy"
    }

    # ── Yield loss comparison data ──────────────────────────
    yield_data = {
        "labels": ["Potential Yield", "Average Actual", "With Disease Loss"],
        "values": [
            food_sec["yield_potential_t_ha"],
            food_sec["average_yield_t_ha"],
            round(food_sec["average_yield_t_ha"] * (1 - food_sec["annual_loss_pct"] / 100), 1)
        ]
    }

    return jsonify({
        "crop": crop,
        "disease": {
            "name": disease_info["name"],
            "id": disease_info["id"],
            "pathogen": disease_info["pathogen"],
            "severity": disease_info["severity"],
            "confidence": result["confidence"],
            "affected_area_pct": result["affected_area_pct"],
            "symptoms": disease_info["symptoms"],
            "treatment": disease_info["treatment"],
            "prevention": disease_info["prevention"],
            "color": disease_info["color"],
            "top3": result.get("top3", [])
        },
        "model_source": result.get("source", "mock"),
        "food_security": food_sec,
        "stats": {
            "prevalence": prevalence,
            "yield_data": yield_data,
            "annual_loss_billion": food_sec["annual_loss_billion_usd"],
            "loss_pct": food_sec["annual_loss_pct"]
        }
    })


if __name__ == '__main__':
    print("🌱 Plant Disease Detection Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)
