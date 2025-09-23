/* === KrishiMitra Enhanced JavaScript with Full Translation Support (Corrected) === */

// Translation data (fixed keys and removed extra spaces)
const translations = {
    en: {
        "app-name": "Krishi Sahyog",
        "nav-logout":"Logout",
        "nav-home": "Home",
        "nav-chatbot": "Smart Assistant",
        "nav-diagnosis": "Plant Diagnosis", // Corrected key name
        "nav-features": "Features",
        "nav-advisory": "Advisory",
        "nav-contact": "Contact",
        "nav-soil":"Soil Analysis",
        "nav-dashboard": "Dashboard",
        "hero-title": "Smart Agricultural Advisory System",
        "hero-subtitle": "Empowering farmers with AI-powered insights, real-time data, and personalized recommendations",
        "hero-title-features": "Explore Our Smart Features",
        "hero-subtitle-features": "Providing comprehensive solutions for modern agriculture with cutting-edge technology.",
        "hero-title-advisory": "Personalized Farming Advisory",
        "hero-subtitle-advisory": "Get crop-specific recommendations based on soil health, weather, and more.",
        "hero-title-contact": "Get in Touch with KrishiMitra",
        "hero-subtitle-contact": "We'd love to hear from you. Reach out to our team for support or inquiries.",
        "hero-title-dashboard": "Your Farm Dashboard",
        "hero-subtitle-dashboard": "A quick overview of your farm's health and market insights.",
        "feature-plant-health": "Plant Health Diagnosis",
        "feature-plant-health-desc": "Upload plant images for AI-powered disease and pest detection using advanced computer vision.",
        "upload-plant-image": "Click to upload plant image",
        "btn-diagnose": "Diagnose Plant",
        "feature-soil-monitoring": "Soil Monitoring",
        "feature-soil-monitoring-desc": "Real-time soil pH, moisture, and temperature monitoring through IoT sensors.",
        "sensor-ph": "pH Level",
        "sensor-moisture": "Moisture",
        "sensor-temperature": "Temperature",
        "sensor-nitrogen": "Nitrogen",
        "sensor-phosphorus": "Phosphorus",
        "sensor-potassium": "Potassium",
        "btn-refresh": "Refresh Data",
        "feature-weather": "Weather Insights",
        "feature-weather-desc": "Get localized weather forecasts and agricultural alerts.",
        "btn-update-weather": "Update Weather",
        "feature-market-prices": "Market Prices",
        "feature-market-prices-desc": "Real-time crop prices and market trends.",
        "btn-update-prices": "Update Prices",
        "feature-crop-advisory": "Crop Advisory",
        "feature-crop-advisory-desc": "Get personalized crop recommendations based on your farm conditions.",
        "label-crop-type": "Crop Type:",
        "label-farm-size": "Farm Size (acres):",
        "btn-get-advice": "Get Advice",
        "feature-ai-assistant": "AI Assistant",
        "feature-ai-assistant-desc": "Chat with our AI assistant for instant farming advice and support.",
        "btn-chat": "Start Chat",
        "btn-voice": "Voice Assistant",
        "results-title": "Analysis Results",
        "chat-title": "KrishiMitra Assistant",
        "chat-welcome": "Welcome! How can I help you today?",
        "crop-rice": "Rice",
        "crop-wheat": "Wheat",
        "crop-potato": "Potato",
        "crop-tomato": "Tomato",
        "crop-onion": "Onion",
        "crop-corn": "Corn",
        "crop-soybean": "Soybean",
        "contact-name": "Name",
        "contact-email": "Email",
        "contact-subject": "Subject",
        "contact-message": "Message",
        "contact-submit": "Send Message",
        "card-title-sensor": "Live Sensor Data",
        "card-desc-sensor": "Real-time readings from your IoT sensors.",
        "card-title-diagnoses": "Recent Plant Diagnoses",
        "card-desc-diagnoses": "AI-powered disease detection history.",
        "card-title-market": "Current Market Prices",
        "card-desc-market": "Up-to-date crop prices from local markets.",
        "btn-fetch-diagnoses": "Fetch Diagnoses",
        "diagnoses-placeholder": "No recent diagnoses available.",
        "weather-location": "Location",
        "soil-analysis": "Soil Analysis",
        "crop-recommendations": "Crop Recommendations",
        "seasonal-advice": "Seasonal Advice",
        "market-trends": "Market Trends",
        "success": "Success",
        "error": "Error",
        "loading": "Loading...",
        "no-data": "No data available",
        "try-again": "Try Again",
        "disease-detected": "Disease Detected",
        "confidence": "Confidence",
        "treatment": "Treatment",
        "severity": "Severity",
        "recommendations": "Recommendations",
        "upload-success": "Image uploaded successfully",
        "analysis-complete": "Analysis complete",
        "unknown-soil": "Various",
        "schedule": "Suggested Schedule",
        "print": "Print Advisory",
        "save-pdf": "Save as PDF",
        "crop-rice-rec-1": "Use 20-25 kg seeds per acre for transplanting",
        "crop-rice-rec-2": "Apply 80:40:40 kg NPK per acre in split doses",
        "crop-rice-rec-3": "Maintain 2-3 cm water depth during vegetative stage",
        "crop-rice-rec-4": "Control weeds with pre-emergence herbicide",
        "crop-rice-rec-5": "Monitor for stem borer and leaf folder",
        "crop-wheat-rec-1": "Use 40-50 kg seeds per acre",
        "crop-wheat-rec-2": "Apply 120:60:40 kg NPK per acre",
        "crop-wheat-rec-3": "First irrigation at crown root initiation (20-25 DAS)",
        "crop-wheat-rec-4": "Second irrigation at late jointing stage",
        "crop-wheat-rec-5": "Control rust disease with propiconazole",
        "default-rec-1": "Test soil pH and nutrient levels before planting",
        "default-rec-2": "Use certified seeds from reliable sources",
        "default-rec-3": "Follow recommended spacing for optimal growth",
        "default-rec-4": "Implement integrated pest management practices",
        "default-rec-5": "Maintain proper irrigation schedule based on crop stage",
        "advisory-card-title": "Get Personalized Advice",
        "soil-type-label": "Soil Type",
        "soil-type-option-default": "Select Soil Type",
        "irrigation-type-label": "Irrigation Type",
        "irrigation-type-option-default": "Select Irrigation",
        "region-state-label": "Region/State",
        "specific-issues-label": "Specific Issues (if any)",
        "describe-placeholder": "Describe any specific problems you're facing...",
        "your-advisory-title": "Your Personalized Advisory",
        "seasonal-guide-title": "Seasonal Farming Guide",
        "kharif-season-title": "Kharif Season",
        "kharif-season-date": "(June - October)",
        "rabi-season-title": "Rabi Season",
        "rabi-season-date": "(November - March)",
        "zaid-season-title": "Zaid Season",
        "zaid-season-date": "(March - June)",
        "recommended-crops": "Recommended Crops",
        "key-activities": "Key Activities",
        "crop-specific-tips-title": "Crop-Specific Best Practices",
        "other-crops": "Other Crops",
        "veg-tab-btn": "Vegetables",
        "cash-tab-btn": "Cash Crops",
        "rice-guide-title": "Rice Cultivation Guide",
        "seed-selection": "Seed Selection",
        "water-management": "Water Management",
        "nutrient-management": "Nutrient Management",
        "pest-management": "Pest Management",
        "wheat-guide-title": "Wheat Cultivation Guide",
        "sowing-time": "Sowing Time",
        "seed-rate": "Seed Rate",
        "irrigation-title": "Irrigation",
        "disease-control": "Disease Control",
        "vegetable-guide-title": "Vegetable Cultivation Guide",
        "tomato": "Tomato",
        "potato": "Potato",
        "cole-crops": "Cole Crops",
        "cucurbits": "Cucurbits",
        "cash-crops-guide-title": "Cash Crops Guide",
        "mustard": "Mustard",
        "sunflower": "Sunflower",
        "groundnut": "Groundnut",
        "sugarcane": "Sugarcane",
        "plant-type": "Plant Type",
        "select-plant-default": "Select Plant Type",
        "label-plant-type": "Plant Type:",
        "sensor-ec":"Electrical Conductivity",
        "sensor-rainfall":"Rainfall"
    },
    hi: {
        "app-name": "कृषि सहियोग",
        "nav-logout":"लॉग आउट",
        "nav-home": "होम",
        "nav-chatbot":"स्मार्ट सहाय",
        "nav-diagnosis": "पौधे का निदान",
        "nav-soil":"मृदा विश्लेषण",
        "nav-features": "विशेषताएं",
        "nav-advisory": "सलाह",
        "nav-contact": "संपर्क",
        "nav-dashboard": "डैशबोर्ड",
        "hero-title": "स्मार्ट कृषि परामर्श प्रणाली",
        "hero-subtitle": "AI, डेटा और व्यक्तिगत सलाह से किसानों को सशक्त बनाना",
        "hero-title-features": "हमारी स्मार्ट विशेषताएं खोजें",
        "hero-subtitle-features": "अत्याधुनिक तकनीक के साथ आधुनिक कृषि के लिए व्यापक समाधान प्रदान करना।",
        "hero-title-advisory": "व्यक्तिगत कृषि परामर्श",
        "hero-subtitle-advisory": "मिट्टी के स्वास्थ्य, मौसम और अन्य के आधार पर फसल-विशिष्ट सिफारिशें प्राप्त करें।",
        "hero-title-contact": "कृषिमित्र से संपर्क करें",
        "hero-subtitle-contact": "हमें आपसे सुनकर खुशी होगी। समर्थन या पूछताछ के लिए हमारी टीम से संपर्क करें।",
        "hero-title-dashboard": "आपका फार्म डैशबोर्ड",
        "hero-subtitle-dashboard": "आपके खेत के स्वास्थ्य और बाजार की जानकारी का एक त्वरित अवलोकन।",
        "feature-plant-health": "पौधों का स्वास्थ्य निदान",
        "feature-plant-health-desc": "रोग और कीट पहचान के लिए पौधों की छवियां अपलोड करें।",
        "upload-plant-image": "पौधे की छवि अपलोड करें",
        "btn-diagnose": "निदान करें",
        "feature-soil-monitoring": "मिट्टी की निगरानी",
        "feature-soil-monitoring-desc": "IoT सेंसर से pH, नमी और तापमान की वास्तविक समय निगरानी।",
        "sensor-ph": "पीएच स्तर",
        "sensor-moisture": "नमी",
        "sensor-temperature": "तापमान",
        "sensor-nitrogen": "नाइट्रोजन",
        "sensor-phosphorus": "फास्फोरस",
        "sensor-potassium": "पोटाश",
        "btn-refresh": "डेटा रीफ्रेश करें",
        "feature-weather": "मौसम जानकारी",
        "feature-weather-desc": "स्थानीय मौसम पूर्वानुमान और कृषि अलर्ट।",
        "btn-update-weather": "मौसम अपडेट करें",
        "feature-market-prices": "बाजार भाव",
        "feature-market-prices-desc": "फसलों की वास्तविक समय कीमतें और रुझान।",
        "btn-update-prices": "कीमतें अपडेट करें",
        "feature-crop-advisory": "फसल परामर्श",
        "feature-crop-advisory-desc": "आपकी खेती की स्थिति के अनुसार सलाह प्राप्त करें।",
        "label-crop-type": "फसल का प्रकार:",
        "label-farm-size": "खेती का आकार (एकड़):",
        "btn-get-advice": "सलाह प्राप्त करें",
        "feature-ai-assistant": "AI सहायक",
        "feature-ai-assistant-desc": "तुरंत कृषि सलाह और समर्थन के लिए चैट करें।",
        "btn-chat": "चैट शुरू करें",
        "btn-voice": "वॉइस असिस्टेंट",
        "results-title": "विश्लेषण परिणाम",
        "chat-title": "कृषिमित्र सहायक",
        "chat-welcome": "स्वागत है! मैं आपकी कैसे मदद कर सकता हूं?",
        "crop-rice": "चावल",
        "crop-wheat": "गेहूं",
        "crop-potato": "आलू",
        "crop-tomato": "टमाटर",
        "crop-onion": "प्याज",
        "crop-corn": "मक्का",
        "crop-soybean": "सोयाबीन",
        "contact-name": "नाम",
        "contact-email": "ईमेल",
        "contact-subject": "विषय",
        "contact-message": "संदेश",
        "contact-submit": "संदेश भेजें",
        "card-title-sensor": "लाइव सेंसर डेटा",
        "card-desc-sensor": "आपके IoT सेंसर से वास्तविक समय रीडिंग।",
        "card-title-diagnoses": "हाल के पौधों के निदान",
        "card-desc-diagnoses": "AI-संचालित रोग पहचान इतिहास।",
        "card-title-market": "वर्तमान बाजार भाव",
        "card-desc-market": "स्थानीय बाजारों से अद्यतन फसल की कीमतें।",
        "btn-fetch-diagnoses": "निदान लाएं",
        "diagnoses-placeholder": "कोई हालिया निदान उपलब्ध नहीं।",
        "weather-location": "स्थान",
        "soil-analysis": "मिट्टी विश्लेषण",
        "crop-recommendations": "फसल सिफारिशें",
        "seasonal-advice": "मौसमी सलाह",
        "market-trends": "बाजार रुझान",
        "success": "सफलता",
        "error": "त्रुटि",
        "loading": "लोड हो रहा है...",
        "no-data": "कोई डेटा उपलब्ध नहीं",
        "try-again": "फिर कोशिश करें",
        "disease-detected": "रोग का पता चला",
        "confidence": "आत्मविश्वास",
        "treatment": "उपचार",
        "severity": "गंभीरता",
        "recommendations": "सिफारिशें",
        "upload-success": "छवि सफलतापूर्वक अपलोड",
        "analysis-complete": "विश्लेषण पूर्ण",
        "unknown-soil": "विभिन्न",
        "schedule": "सुझाया गया कार्यक्रम",
        "print": "परामर्श प्रिंट करें",
        "save-pdf": "पीडीएफ के रूप में सहेजें",
        "crop-rice-rec-1": "प्रत्यारोपण के लिए प्रति एकड़ 20-25 किलो बीज का उपयोग करें",
        "crop-rice-rec-2": "विभाजित खुराकों में प्रति एकड़ 80:40:40 किलो एनपीके लागू करें",
        "crop-rice-rec-3": "वनस्पति चरण के दौरान 2-3 सेमी पानी की गहराई बनाए रखें",
        "crop-rice-rec-4": "पूर्व-उभरने वाले शाकनाशी से खरपतवारों को नियंत्रित करें",
        "crop-rice-rec-5": "तना छेदक और पत्ती मोड़क की निगरानी करें",
        "crop-wheat-rec-1": "प्रति एकड़ 40-50 किलो बीज का उपयोग करें",
        "crop-wheat-rec-2": "प्रति एकड़ 120:60:40 किलो एनपीके लागू करें",
        "crop-wheat-rec-3": "क्राउन रूट दीक्षा (20-25 डीएएस) पर पहला सिंचाई करें",
        "crop-wheat-rec-4": "लेट जॉइंटिंग चरण में दूसरी सिंचाई करें",
        "crop-wheat-rec-5": "प्रोपीकोनाज़ोल से रस्ट रोग को नियंत्रित करें",
        "default-rec-1": "रोपण से पहले मिट्टी के पीएच और पोषक तत्वों के स्तर का परीक्षण करें",
        "default-rec-2": "विश्वसनीय स्रोतों से प्रमाणित बीजों का उपयोग करें",
        "default-rec-3": "इष्टतम विकास के लिए अनुशंसित रिक्ति का पालन करें",
        "default-rec-4": "एकीकृत कीट प्रबंधन प्रथाओं को लागू करें",
        "default-rec-5": "फसल के चरण के आधार पर उचित सिंचाई कार्यक्रम बनाए रखें",
        "advisory-card-title": "व्यक्तिगत सलाह प्राप्त करें",
        "soil-type-label": "मिट्टी का प्रकार",
        "soil-type-option-default": "मिट्टी का प्रकार चुनें",
        "irrigation-type-label": "सिंचाई का प्रकार",
        "irrigation-type-option-default": "सिंचाई चुनें",
        "region-state-label": "क्षेत्र/राज्य",
        "specific-issues-label": "विशिष्ट मुद्दे (यदि कोई हो)",
        "describe-placeholder": "आप जिन विशिष्ट समस्याओं का सामना कर रहे हैं, उनका वर्णन करें...",
        "your-advisory-title": "आपकी व्यक्तिगत सलाह",
        "seasonal-guide-title": "मौसमी खेती गाइड",
        "kharif-season-title": "खरीफ सीजन",
        "kharif-season-date": "(जून - अक्टूबर)",
        "rabi-season-title": "रबी सीजन",
        "rabi-season-date": "(नवंबर - মার্চ)",
        "zaid-season-title": "जायद सीजन",
        "zaid-season-date": "(मार्च - जून)",
        "recommended-crops": "अनुशंसित फसलें",
        "key-activities": "मुख्य गतिविधियां",
        "crop-specific-tips-title": "फसल-विशिष्ट सर्वोत्तम प्रथाएं",
        "other-crops": "अन्य फसलें",
        "veg-tab-btn": "सब्जियां",
        "cash-tab-btn": "नकदी फसलें",
        "rice-guide-title": "चावल की खेती का गाइड",
        "seed-selection": "बीज चयन",
        "water-management": "जल प्रबंधन",
        "nutrient-management": "पोषक तत्व प्रबंधन",
        "pest-management": "कीट प्रबंधन",
        "wheat-guide-title": "गेहूं की खेती का गाइड",
        "sowing-time": "बुवाई का समय",
        "seed-rate": "बीज दर",
        "irrigation-title": "सिंचाई",
        "disease-control": "रोग नियंत्रण",
        "vegetable-guide-title": "सब्जी की खेती का गाइड",
        "tomato": "टमाटर",
        "potato": "आलू",
        "cole-crops": "गोभी कुल की फसलें",
        "cucurbits": "कुकर्बिट्स",
        "cash-crops-guide-title": "नकदी फसलों का गाइड",
        "mustard": "सरसों",
        "sunflower": "सूरजमुखी",
        "groundnut": "मूंगफली",
        "sugarcane": "गन्ना",
        "plant-type": "पौधे का प्रकार",
        "select-plant-default": "पौधे का प्रकार चुनें",
        "label-plant-type": "पौधे का प्रकार:"
    },
    bn: {
        "app-name": "কৃষি সহায়ক",
        "nav-logout":"লগআউট",
        "nav-home": "হোম",
        "nav-diagnosis": "গাছের রোগ নির্ণয়",
        "nav-soil":"মাটি বিশ্লেষণ", 
        "nav-chatbot":"স্মার্ট সহকারী",
        "nav-features": "বৈশিষ্ট্য",
        "nav-advisory": "পরামর্শ",
        "nav-contact": "যোগাযোগ",
        "nav-dashboard": "ড্যাশবোর্ড",
        "hero-title": "স্মার্ট কৃষি পরামর্শ ব্যবস্থা",
        "hero-subtitle": "AI, তথ্য এবং ব্যক্তিগত পরামর্শের মাধ্যমে কৃষকদের ক্ষমতায়ন",
        "hero-title-features": "আমাদের স্মার্ট বৈশিষ্ট্যগুলি অন্বেষণ করুন",
        "hero-subtitle-features": "সর্বাধুনিক প্রযুক্তির মাধ্যমে আধুনিক কৃষির জন্য ব্যাপক সমাধান প্রদান করা।",
        "hero-title-advisory": "ব্যক্তিগত কৃষি পরামর্শ",
        "hero-subtitle-advisory": "মাটির স্বাস্থ্য, আবহাওয়া এবং আরও অনেক কিছুর উপর ভিত্তি করে ফসল-নির্দিষ্ট সুপারিশ পান।",
        "hero-title-contact": "কৃষিমিত্রের সাথে যোগাযোগ করুন",
        "hero-subtitle-contact": "আমরা আপনার কাছ থেকে শুনে খুশি হব। সমর্থন বা অনুসন্ধানের জন্য আমাদের দলের সাথে যোগাযোগ করুন।",
        "hero-title-dashboard": "আপনার ফার্ম ড্যাশবোর্ড",
        "hero-subtitle-dashboard": "আপনার ক্ষেত্রের স্বাস্থ্য এবং বাজারের তথ্যের একটি দ্রুত পর্যবেক্ষণ।",
        "feature-plant-health": "গাছের স্বাস্থ্য নির্ণয়",
        "feature-plant-health-desc": "রোগ এবং কীট শনাক্তকরণের জন্য গাছের ছবি আপলোড করুন।",
        "upload-plant-image": "গাছের ছবি আপলোড করুন",
        "btn-diagnose": "নির্ণয় করুন",
        "feature-soil-monitoring": "মাটির পর্যবেক্ষণ",
        "feature-soil-monitoring-desc": "IoT সেন্সরের মাধ্যমে pH, আর্দ্রতা এবং তাপমাত্রার রিয়েল-টাইম পর্যবেক্ষণ।",
        "sensor-ph": "pH স্তর",
        "sensor-moisture": "আর্দ্রতা",
        "sensor-temperature": "তাপমাত্রা",
        "sensor-nitrogen": "নাইট্রোজেন",
        "sensor-phosphorus": "ফসফরাস",
        "sensor-potassium": "পটাশিয়াম",
        "btn-refresh": "ডেটা রিফ্রেশ করুন",
        "feature-weather": "আবহাওয়ার তথ্য",
        "feature-weather-desc": "স্থানীয় আবহাওয়ার পূর্বাভাস এবং কৃষি সতর্কতা।",
        "btn-update-weather": "আবহাওয়া আপডেট করুন",
        "feature-market-prices": "বাজার মূল্য",
        "feature-market-prices-desc": "ফসলের রিয়েল-টাইম দাম এবং প্রবণতা।",
        "btn-update-prices": "মূল্য আপডেট করুন",
        "feature-crop-advisory": "ফসল পরামর্শ",
        "feature-crop-advisory-desc": "আপনার কৃষিক্ষেত্রের শর্ত অনুযায়ী ব্যক্তিগত পরামর্শ পান।",
        "label-crop-type": "ফসলের ধরন:",
        "label-farm-size": "ক্ষেত্রের আকার (একর):",
        "btn-get-advice": "পরামর্শ পান",
        "feature-ai-assistant": "AI সহকারী",
        "feature-ai-assistant-desc": "তাৎক্ষণিক কৃষি পরামর্শ এবং সহায়তার জন্য চ্যাট করুন।",
        "btn-chat": "চ্যাট শুরু করুন",
        "btn-voice": "ভয়েস সহকারী",
        "results-title": "বিশ্লেষণের ফলাফল",
        "chat-title": "কৃষিমিত্র সহকারী",
        "chat-welcome": "স্বাগতম! আমি কিভাবে আপনার সাহায্য করতে পারি?",
        "crop-rice": "ধান",
        "crop-wheat": "গম",
        "crop-potato": "আলু",
        "crop-tomato": "টমেটো",
        "crop-onion": "পেঁয়াজ",
        "crop-corn": "ভুট্টা",
        "crop-soybean": "সয়াবিন",
        "contact-name": "নাম",
        "contact-email": "ইমেইল",
        "contact-subject": "বিষয়",
        "contact-message": "বার্তা",
        "contact-submit": "বার্তা পাঠান",
        "card-title-sensor": "লাইভ সেন্সর ডেটা",
        "card-desc-sensor": "আপনার IoT সেন্সর থেকে রিয়েল-টাইম রিডিং।",
        "card-title-diagnoses": "সাম্প্রতিক গাছের নির্ণয়",
        "card-desc-diagnoses": "AI-চালিত রোগ শনাক্তকরণের ইতিহাস।",
        "card-title-market": "বর্তমান বাজার মূল্য",
        "card-desc-market": "স্থানীয় বাজার থেকে আপ-টু-ডেট ফসলের দাম।",
        "btn-fetch-diagnoses": "নির্ণয় আনুন",
        "diagnoses-placeholder": "কোনো সাম্প্রতিক নির্ণয় উপলব্ধ নেই।",
        "weather-location": "অবস্থান",
        "soil-analysis": "মাটি বিশ্লেষণ",
        "crop-recommendations": "ফসল সুপারিশ",
        "seasonal-advice": "মৌসুমি পরামর্শ",
        "market-trends": "বাজার প্রবণতা",
        "success": "সফল",
        "error": "ত্রুটি",
        "loading": "লোড হচ্ছে...",
        "no-data": "কোনো ডেটা উপলব্ধ নেই",
        "try-again": "আবার চেষ্টা করুন",
        "disease-detected": "রোগ শনাক্ত",
        "confidence": "আত্মবিশ্বাস",
        "treatment": "চিকিৎসা",
        "severity": "তীব্রতা",
        "recommendations": "সুপারিশ",
        "upload-success": "ছবি সফলভাবে আপলোড",
        "analysis-complete": "বিশ্লেষণ সম্পূর্ণ",
        "unknown-soil": "বিভিন্ন",
        "schedule": "প্রস্তাবিত সময়সূচী",
        "print": "পরামর্শ প্রিন্ট করুন",
        "save-pdf": "পিডিএফ হিসাবে সংরক্ষণ করুন",
        "crop-rice-rec-1": "প্রতিরোপণের জন্য প্রতি একর 20-25 কেজি বীজ ব্যবহার করুন",
        "crop-rice-rec-2": "বিভক্ত মাত্রায় প্রতি একর 80:40:40 কেজি এনপিকে প্রয়োগ করুন",
        "crop-rice-rec-3": "উদ্ভিজ্জ বৃদ্ধির সময় 2-3 সেমি পানির গভীরতা বজায় রাখুন",
        "crop-rice-rec-4": "আগাছা দমনের জন্য প্রি-ইমার্জেন্স হার্বিসাইড ব্যবহার করুন",
        "crop-rice-rec-5": "স্টেম বোরার এবং লিফ ফোল্ডারের জন্য পর্যবেক্ষণ করুন",
        "crop-wheat-rec-1": "প্রতি একর 40-50 কেজি বীজ ব্যবহার করুন",
        "crop-wheat-rec-2": "প্রতি একর 120:60:40 কেজি এনপিকে প্রয়োগ করুন",
        "crop-wheat-rec-3": "ক্রাউন রুট ইনিশিয়েশন (20-25 ডিএএস) পর্যায়ে প্রথম সেচ দিন",
        "crop-wheat-rec-4": "লেটারাল জয়েনিং পর্যায়ে দ্বিতীয় সেচ দিন",
        "crop-wheat-rec-5": "প্রোপিকোনাজল দিয়ে রাস্ট রোগ নিয়ন্ত্রণ করুন",
        "default-rec-1": "রোপণের আগে মাটির পিএইচ এবং পুষ্টির মাত্রা পরীক্ষা করুন",
        "default-rec-2": "নির্ভরযোগ্য উৎস থেকে সার্টিফাইড বীজ ব্যবহার করুন",
        "default-rec-3": "সর্বোত্তম বৃদ্ধির জন্য প্রস্তাবিত দূরত্ব বজায় রাখুন",
        "default-rec-4": "সমন্বিত কীটপতঙ্গ ব্যবস্থাপনা পদ্ধতি প্রয়োগ করুন",
        "default-rec-5": "ফসলের পর্যায় অনুযায়ী সঠিক সেচ সময়সূচী বজায় রাখুন",
        "advisory-card-title": "ব্যক্তিগত পরামর্শ পান",
        "soil-type-label": "মাটির ধরন",
        "soil-type-option-default": "মাটির ধরন নির্বাচন করুন",
        "irrigation-type-label": "সেচ পদ্ধতি",
        "irrigation-type-option-default": "সেচ পদ্ধতি নির্বাচন করুন",
        "region-state-label": "অঞ্চল/রাজ্য",
        "specific-issues-label": "নির্দিষ্ট সমস্যা (যদি থাকে)",
        "describe-placeholder": "আপনার সম্মুখীন হওয়া নির্দিষ্ট সমস্যাগুলি বর্ণনা করুন...",
        "your-advisory-title": "আপনার ব্যক্তিগত পরামর্শ",
        "seasonal-guide-title": "মৌসুমি চাষের গাইড",
        "kharif-season-title": "খরিফ সিজন",
        "kharif-season-date": "(জুন - অক্টোবর)",
        "rabi-season-title": "রবি সিজন",
        "rabi-season-date": "(নভেম্বর - মার্চ)",
        "zaid-season-title": "জাইদ সিজন",
        "zaid-season-date": "(মার্চ - জুন)",
        "recommended-crops": "প্রস্তাবিত ফসল",
        "key-activities": "মূল কার্যক্রম",
        "crop-specific-tips-title": "ফসল-নির্দিষ্ট সেরা অনুশীলন",
        "other-crops": "অন্যান্য ফসল",
        "veg-tab-btn": "সবজি",
        "cash-tab-btn": "অর্থকরী ফসল",
        "rice-guide-title": "ধান চাষের গাইড",
        "seed-selection": "বীজ নির্বাচন",
        "water-management": "জল ব্যবস্থাপনা",
        "nutrient-management": "পুষ্টি ব্যবস্থাপনা",
        "pest-management": "কীটপতঙ্গ ব্যবস্থাপনা",
        "wheat-guide-title": "গম চাষের গাইড",
        "sowing-time": "বপনের সময়",
        "seed-rate": "বীজের হার",
        "irrigation-title": "সেচ",
        "disease-control": "রোগ নিয়ন্ত্রণ",
        "vegetable-guide-title": "সবজি চাষের গাইড",
        "tomato": "টমেটো",
        "potato": "আলু",
        "cole-crops": "কোল ফসল",
        "cucurbits": "কুকুরবিট",
        "cash-crops-guide-title": "অর্থকরী ফসলের গাইড",
        "mustard": "সর্ষে",
        "sunflower": "সূর্যমুখী",
        "groundnut": "চিনাবাদাম",
        "sugarcane": "আখ",
        "plant-type": "গাছের প্রকার",
        "select-plant-default": "গাছের প্রকার নির্বাচন করুন",
        "label-plant-type": "গাছের প্রকার:",
        
    }
};

let currentLang = 'en';
let socket = null;

// Initialize WebSocket connection
function initializeWebSocket() {
    if (typeof io !== 'undefined') {
        socket = io();
        
        socket.on('connect', function() {
            console.log('Connected to KrishiMitra server');
        });
        
        socket.on('sensor_update', function(data) {
            updateSensorDisplay(data);
        });
        
        socket.on('weather_update', function(data) {
            updateWeatherDisplay(data);
        });
        
        socket.on('market_update', function(data) {
            updateMarketDisplay(data);
        });
    }
}

// Language Management
function changeLanguage(lang, event) {
    if (event) {
        event.preventDefault();
    }
    
    currentLang = lang;
    
    // Update all translatable elements
    document.querySelectorAll('[data-translate]').forEach(element => {
        const key = element.getAttribute('data-translate');
        if (translations[lang] && translations[lang][key]) {
            if (element.tagName === 'INPUT' && element.type === 'text') {
                element.placeholder = translations[lang][key];
            } else {
                element.textContent = translations[lang][key];
            }
        }
    });
    
    // Update language buttons
    document.querySelectorAll('.lang-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.lang-btn[onclick*="'${lang}'"]`).classList.add('active');
    
    // Update select options
    updateSelectOptions();
    
    // Re-fetch dynamic content if on dashboard
    if (window.location.pathname.includes('dashboard')) {
        fetchDiagnoses();
        updateMarketPrices();
    }
    
    // Update chat messages
    updateChatLanguage();
    
    console.log(`Language changed to: ${lang}`);
}

function updateSelectOptions() {
    // Update crop selection options
    const cropSelects = document.querySelectorAll('#cropType');
    cropSelects.forEach(select => {
        Array.from(select.options).forEach(option => {
            const key = option.getAttribute('data-translate');
            if (key && translations[currentLang][key]) {
                option.textContent = translations[currentLang][key];
            }
        });
    });
    
    // Update plant type selection options
    const plantTypeSelects = document.querySelectorAll('#plantType');
    plantTypeSelects.forEach(select => {
        Array.from(select.options).forEach(option => {
            const key = option.getAttribute('data-translate');
            if (key && translations[currentLang][key]) {
                option.textContent = translations[currentLang][key];
            }
        });
    });
    
    // Update other select options on advisory page
    const soilSelect = document.getElementById('soilType');
    if (soilSelect) {
        const options = soilSelect.options;
        options[0].textContent = translations[currentLang]['soil-type-option-default'] || 'Select Soil Type';
    }
    
    const irrigationSelect = document.getElementById('irrigation');
    if (irrigationSelect) {
        const options = irrigationSelect.options;
        options[0].textContent = translations[currentLang]['irrigation-type-option-default'] || 'Select Irrigation';
    }
}

function updateChatLanguage() {
    const chatWelcome = document.querySelector('[data-translate="chat-welcome"]');
    if (chatWelcome && translations[currentLang]['chat-welcome']) {
        chatWelcome.textContent = translations[currentLang]['chat-welcome'];
    }
}

// Plant Disease Detection
function handleImageUpload(input) {
    const file = input.files[0];
    if (file) {
        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        if (!allowedTypes.includes(file.type)) {
            showMessage('error', 'Please select a valid image file (JPEG, PNG, GIF)');
            return;
        }
        
        // Validate file size (max 5MB)
        if (file.size > 5 * 1024 * 1024) {
            showMessage('error', 'File size too large. Please select an image under 5MB');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            if (preview) {
                preview.innerHTML = `
                    <div class="image-preview-container">
                        <img src="${e.target.result}" style="max-width: 100%; max-height: 300px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <div class="image-info">
                            <small>📁 ${file.name} (${(file.size / 1024).toFixed(1)} KB)</small>
                        </div>
                    </div>
                `;
            }
        };
        reader.readAsDataURL(file);
        
        showMessage('success', translations[currentLang]['upload-success'] || 'Image uploaded successfully');
    }
}

function diagnosePlant() {
    const fileInput = document.getElementById('plantImage');
    const file = fileInput?.files[0];
    const plantTypeSelect = document.getElementById('plantType');
    const plantType = plantTypeSelect?.value;
    
    if (!file) {
        showMessage('error', 'Please select an image to diagnose');
        return;
    }
    
    if (!plantType) {
        showMessage('error', 'Please select a plant type');
        return;
    }
    
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    if (resultsContent) {
        resultsContent.innerHTML = `
            <div class="loading-container">
                <div class="spinner"></div>
                <p>${translations[currentLang]['loading'] || 'Analyzing image...'}</p>
            </div>
        `;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('plant_type', plantType);  // Add plant type to form data
    
    fetch('/api/upload-image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && resultsContent) {
            const result = data.result;
            resultsContent.innerHTML = `
                <div class="diagnosis-result">
                    <div class="result-header">
                        <h4>🔬 ${translations[currentLang]['disease-detected'] || 'Disease Analysis'}</h4>
                    </div>
                    <div class="result-grid">
                        <div class="result-item">
                            <strong>${translations[currentLang]['plant-type'] || 'Plant Type'}:</strong>
                            <span class="plant-type">${result.plant_type || 'Unknown'}</span>
                        </div>
                        <div class="result-item">
                            <strong>${translations[currentLang]['disease-detected'] || 'Disease'}:</strong>
                            <span class="disease-name">${result.disease}</span>
                        </div>
                        <div class="result-item">
                            <strong>${translations[currentLang]['confidence'] || 'Confidence'}:</strong>
                            <span class="confidence-score">${result.confidence.toFixed(1)}%</span>
                        </div>
                        <div class="result-item">
                            <strong>${translations[currentLang]['severity'] || 'Severity'}:</strong>
                            <span class="severity-${result.severity}">${result.severity}</span>
                        </div>
                    </div>
                    <div class="treatment-section">
                        <h5>💊 ${translations[currentLang]['treatment'] || 'Treatment'}</h5>
                        <p>${result.treatment}</p>
                    </div>
                    ${result.recommendations ? `
                        <div class="recommendations-section">
                            <h5>💡 ${translations[currentLang]['recommendations'] || 'Recommendations'}</h5>
                            <ul>
                                ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        } else {
            if (resultsContent) {
                resultsContent.innerHTML = `
                    <div class="error-message">
                        <p>❌ ${data.error || 'Analysis failed. Please try again.'}</p>
                    </div>
                `;
            }
        }
    })
    .catch(error => {
        console.error('Diagnosis error:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div class="error-message">
                    <p>❌ ${translations[currentLang]['error'] || 'Error'}: Failed to analyze image</p>
                </div>
            `;
        }
    });
}
// Sensor Data Management
function updateSensorDisplay(data) {
    const sensors = ['soilPH', 'soilMoisture', 'soilTemp', 'nitrogen', 'phosphorus', 'potassium'];
    
    sensors.forEach(sensor => {
        const element = document.getElementById(sensor);
        if (element && data[sensor.replace('soil', 'soil_')]) {
            let value = data[sensor.replace('soil', 'soil_')];
            let unit = '';
            
            switch(sensor) {
                case 'soilPH':
                    unit = '';
                    value = value.toFixed(1);
                    break;
                case 'soilMoisture':
                    unit = '%';
                    value = Math.round(value);
                    break;
                case 'soilTemp':
                    unit = '°C';
                    value = value.toFixed(1);
                    break;
                default:
                    unit = ' ppm';
                    value = Math.round(value);
            }
            
            element.textContent = value + unit;
            
            // Add visual indicators
            element.className = 'sensor-value ' + getSensorStatusClass(sensor, parseFloat(value));
        }
    });
    
    // Update last updated time
    const lastUpdated = document.getElementById('lastUpdated');
    if (lastUpdated) {
        lastUpdated.textContent = new Date(data.last_updated).toLocaleTimeString();
    }
}

function getSensorStatusClass(sensor, value) {
    switch(sensor) {
        case 'soilPH':
            if (value >= 6.0 && value <= 7.5) return 'good';
            if (value >= 5.5 && value <= 8.0) return 'fair';
            return 'poor';
        case 'soilMoisture':
            if (value >= 50 && value <= 80) return 'good';
            if (value >= 30 && value <= 90) return 'fair';
            return 'poor';
        case 'soilTemp':
            if (value >= 20 && value <= 30) return 'good';
            if (value >= 15 && value <= 35) return 'fair';
            return 'poor';
        default:
            return 'neutral';
    }
}

function refreshSensorData() {
    fetch('/api/sensor-data')
    .then(response => response.json())
    .then(data => {
        if (data.sensor_data) {
            updateSensorDisplay(data.sensor_data);
            showMessage('success', 'Sensor data updated successfully');
            
            // Update soil analysis if available
            if (data.analysis) {
                updateSoilAnalysis(data.analysis);
            }
        }
    })
    .catch(error => {
        console.error('Sensor refresh error:', error);
        showMessage('error', 'Failed to refresh sensor data');
    });
}

function updateSoilAnalysis(analysis) {
    const analysisContainer = document.getElementById('soilAnalysis');
    if (analysisContainer && analysis) {
        analysisContainer.innerHTML = `
            <div class="soil-analysis">
                <h4>🌱 ${translations[currentLang]['soil-analysis'] || 'Soil Analysis'}</h4>
                <div class="health-status ${analysis.overall_health.toLowerCase()}">
                    <strong>Overall Health:</strong> ${analysis.overall_health}
                </div>
                ${analysis.warnings.length > 0 ? `
                    <div class="warnings">
                        <h5>⚠️ Warnings:</h5>
                        <ul>${analysis.warnings.map(w => `<li>${w}</li>`).join('')}</ul>
                    </div>
                ` : ''}
                ${analysis.recommendations.length > 0 ? `
                    <div class="recommendations">
                        <h5>💡 Recommendations:</h5>
                        <ul>${analysis.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                    </div>
                ` : ''}
                ${analysis.suitable_crops.length > 0 ? `
                    <div class="suitable-crops">
                        <h5>🌾 Suitable Crops:</h5>
                        <div class="crop-tags">
                            ${analysis.suitable_crops.map(crop => `<span class="crop-tag">${crop}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }
}

// Weather Management
const apiKey = '246d2bc3ae694631a9e193617250509';

// Function to translate a given text string
function translateText(key, defaultText) {
    return translations[currentLang][key] || defaultText;
}

// Function to clear all weather display elements
function clearWeatherDisplay() {
    const idsToClear = ['temperature', 'humidity', 'weatherDesc', 'locationName', 'wind', 'precip', 'airQuality'];
    idsToClear.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = '';
        }
    });
    // Also clear weather alerts
    const alertContainer = document.getElementById('weatherAlerts');
    if (alertContainer) {
        alertContainer.style.display = 'none';
        alertContainer.innerHTML = '';
    }
}

function updateWeatherDisplay(data) {
    const temperature = data.current.temp_c;
    const humidity = data.current.humidity;
    const weatherDescText = data.current.condition.text; // Store the text in a new variable
    const locationName = data.location.name;

    // New data points
    const windSpeed = data.current.wind_kph;
    const windDir = data.current.wind_dir;
    
    // Conditional check for precipitation data. Defaults to 0 if not present.
    const precipitation = data.current.precip_mm || 0; 
    
    // Air quality data
    const airQualityData = data.current.air_quality;
    const aqi_us = airQualityData['us-epa-index']; 
    let aqiStatus = '';
    
    // Assign a text status based on the AQI index
    if (aqi_us === 1) aqiStatus = 'Good';
    else if (aqi_us === 2) aqiStatus = 'Moderate';
    else if (aqi_us === 3) aqiStatus = 'Unhealthy for Sensitive Groups';
    else if (aqi_us === 4) aqiStatus = 'Unhealthy';
    else if (aqi_us === 5) aqiStatus = 'Very Unhealthy';
    else if (aqi_us === 6) aqiStatus = 'Hazardous';

    const elements = {
        temperature: `${temperature}°C`,
        humidity: `${humidity}%`,
        weatherDesc: weatherDescText, // Use the new variable that contains the text
        locationName: locationName,
        wind: `${windSpeed} km/h ${windDir}`,
        precip: `${precipitation} mm`,
        airQuality: `${aqiStatus}`
    };
    
    Object.keys(elements).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = elements[id];
        }
    });
    
    // Update weather alerts if available
    if (data.alerts && data.alerts.alert && data.alerts.alert.length > 0) {
        showWeatherAlerts(data.alerts.alert);
    }
}

function updateWeather() {
    clearWeatherDisplay(); // Clear the display immediately

    // Get user's location or try to use cached data
    const cached = localStorage.getItem('lastWeatherUpdate');
    if (cached) {
        const cachedData = JSON.parse(cached);
        if (Date.now() - cachedData.timestamp < 3600000) { // 1 hour
            updateWeatherDisplay(cachedData.data);
            showMessage('info', 'Using cached weather data');
            return;
        }
    }

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            position => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                fetchWeatherData(`${lat},${lon}`);
            },
            error => {
                console.log('Location access denied or timed out. Please check location settings.');
                showMessage('error', 'Location not available. Please allow location access or try again later.');
            }
        );
    } else {
        console.log('Geolocation not supported by this browser.');
        showMessage('error', 'Geolocation not supported.');
    }
}

function fetchWeatherData(location) {
    const url = `http://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${location}&aqi=yes&alerts=yes`;
    
    fetch(url)
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        updateWeatherDisplay(data);
        showMessage('success', 'Weather data updated');
        
        // Store weather data locally for offline access
        localStorage.setItem('lastWeatherUpdate', JSON.stringify({
            data: data,
            timestamp: Date.now()
        }));
    })
    .catch(error => {
        console.error('Weather update error:', error);
        showMessage('error', 'Failed to fetch weather data. Please check your internet connection.');
    });
}

function showWeatherAlerts(alerts) {
    const alertContainer = document.getElementById('weatherAlerts') || createWeatherAlertsContainer();
    
    if (alertContainer) {
        alertContainer.innerHTML = `
            <div class="weather-alerts">
                <h5>🌤️ Weather Alerts</h5>
                ${alerts.map(alert => `<div class="alert-item">${alert.headline}: ${alert.desc}</div>`).join('')}
            </div>
        `;
        alertContainer.style.display = 'block';
    }
}

function createWeatherAlertsContainer() {
    const container = document.createElement('div');
    container.id = 'weatherAlerts';
    container.className = 'weather-alerts-container';
    
    const weatherWidget = document.getElementById('weatherWidget');
    if (weatherWidget) {
        weatherWidget.appendChild(container);
    }
    
    return container;
}

// Market Price Management
// Your corrected getMarketPrice() function
function getMarketPrice() {
    const outputDiv = document.getElementById('priceOutput');
    const cropSelect = document.getElementById('priceCropType');
    const cropType = cropSelect.value;
    const cropNameTranslated = translations[currentLang][`crop-${cropType}`] || cropType; // Get the translated crop name

    // Display a loading state
    outputDiv.innerHTML = `
        <div class="loading-placeholder">
            <i class="fas fa-spinner fa-spin"></i>
            <span>${translations[currentLang]['loading'] || 'Loading...'}</span>
        </div>
    `;

    // This is the correct way to structure your price data
    const prices = {
        'rice': { value: '1,500', unitKey: 'quintal' },
        'wheat': { value: '2,100', unitKey: 'quintal' },
        'tomato': { value: '35', unitKey: 'kg' },
        'potato': { value: '25', unitKey: 'kg' },
        'onion': { value: '20', unitKey: 'kg' },
        'corn': { value: '1,800', unitKey: 'quintal' },
        'soybean': { value: '4,500', unitKey: 'quintal' }
    };
    // Simulate API call
    setTimeout(() => {
        // Use the new, structured prices object
        const priceData = prices[cropType];

        let priceText;
        if (priceData) {
            // Translate the unit based on the unitKey
            const translatedUnit = translations[currentLang][`unit-${priceData.unitKey}`] || priceData.unitKey;
            priceText = `₹${priceData.value}/${translatedUnit}`;
        } else {
            priceText = translations[currentLang]['price-not-available'] || 'Price Not Available';
        }

        // Dynamically create the final HTML with the translated content
        outputDiv.innerHTML = `
            <h4>${translations[currentLang]['price-for'] || 'Market Price for'} ${cropNameTranslated}</h4>
            <div class="price-value">${priceText}</div>
            <div class="price-date">${translations[currentLang]['data-as-of-today'] || 'Data as of today.'}</div>
        `;
    }, 1500);
}


// Crop Advisory
function getCropAdvice(event) {
    if (event) {
        event.preventDefault();
    }
    
    const cropType = document.getElementById('cropType')?.value;
    const farmSize = document.getElementById('farmSize')?.value;
    const soilType = document.getElementById('soilType')?.value;
    const irrigation = document.getElementById('irrigation')?.value;
    const region = document.getElementById('region')?.value;
    const specificIssue = document.getElementById('specificIssue')?.value;
    
    if (!cropType || !farmSize) {
        showMessage('error', 'Please select a crop and enter farm size');
        return;
    }
    
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    if (resultsContent) {
        resultsContent.innerHTML = `
            <div class="loading-container">
                <div class="spinner"></div>
                <p>Analyzing conditions for ${farmSize} acres of ${translations[currentLang][`crop-${cropType}`] || cropType}...</p>
            </div>
        `;
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    fetch('/api/crop-advisory', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            crop_type: cropType,
            farm_size: parseFloat(farmSize),
            soil_type: soilType,
            irrigation: irrigation,
            region: region,
            specific_issue: specificIssue,
            language: currentLang
        })
    })
    .then(response => response.json())
    .then(data => {
        if (resultsContent) {
            displayCropAdvice(data, resultsContent);
        }
    })
    .catch(error => {
        console.error('Crop advice error:', error);
        if (resultsContent) {
            resultsContent.innerHTML = `
                <div class="error-message">
                    <p> <li>🌱 Farmers are advised to monitor soil moisture regularly and provide irrigation as per crop stage.</li>
                        <li>🌾 Timely weeding is essential to reduce competition for nutrients.</li>
                        <li>🛡️ Apply recommended doses of fertilizers for balanced crop growth.</li>
                        <li>🍂 Keep a close watch for pest and disease symptoms, and use biopesticides or safe sprays when required.</li>
                        <li>☀️ Avoid water stagnation during heavy rains to protect roots.</li>
                        <li>🚜 Harvest crops at proper maturity for best yield and quality.</li></p>
                </div>
            `;
        }
    });
}

function displayCropAdvice(data, container) {
    // This is a mock function, in a real app this data would come from the API response
    const advisoryData = {
        'rice': {
            title: translations[currentLang]['crop-rice'] + " " + (translations[currentLang]['crop-recommendations'] || 'Cultivation Advisory'),
            recommendations: [
                translations[currentLang]['crop-rice-rec-1'] || 'Use 20-25 kg seeds per acre for transplanting',
                translations[currentLang]['crop-rice-rec-2'] || 'Apply 80:40:40 kg NPK per acre in split doses',
                translations[currentLang]['crop-rice-rec-3'] || 'Maintain 2-3 cm water depth during vegetative stage',
                translations[currentLang]['crop-rice-rec-4'] || 'Control weeds with pre-emergence herbicide',
                translations[currentLang]['crop-rice-rec-5'] || 'Monitor for stem borer and leaf folder'
            ],
            schedule: {
                'Land Preparation': 'June',
                'Transplanting': 'July',
                'First Irrigation': '3 days after transplanting',
                'First Fertilizer': '10 days after transplanting',
                'Harvest': 'November'
            }
        },
        'wheat': {
            title: translations[currentLang]['crop-wheat'] + " " + (translations[currentLang]['crop-recommendations'] || 'Cultivation Advisory'),
            recommendations: [
                translations[currentLang]['crop-wheat-rec-1'] || 'Use 40-50 kg seeds per acre',
                translations[currentLang]['crop-wheat-rec-2'] || 'Apply 120:60:40 kg NPK per acre',
                translations[currentLang]['crop-wheat-rec-3'] || 'First irrigation at crown root initiation (20-25 DAS)',
                translations[currentLang]['crop-wheat-rec-4'] || 'Second irrigation at late jointing stage',
                translations[currentLang]['crop-wheat-rec-5'] || 'Control rust disease with propiconazole'
            ],
            schedule: {
                'Sowing': 'November 15-30',
                'First Irrigation': '21 days after sowing',
                'First Fertilizer': 'At sowing (basal)',
                'Second Fertilizer': 'First irrigation',
                'Harvest': 'March-April'
            }
        },
        'default': {
            title: `${data.crop_type.charAt(0).toUpperCase() + data.crop_type.slice(1)} ${translations[currentLang]['crop-recommendations'] || 'Cultivation Advisory'}`,
            recommendations: [
                translations[currentLang]['default-rec-1'] || 'Test soil pH and nutrient levels before planting',
                translations[currentLang]['default-rec-2'] || 'Use certified seeds from reliable sources',
                translations[currentLang]['default-rec-3'] || 'Follow recommended spacing for optimal growth',
                translations[currentLang]['default-rec-4'] || 'Implement integrated pest management practices',
                translations[currentLang]['default-rec-5'] || 'Maintain proper irrigation schedule based on crop stage'
            ],
            schedule: {
                'Soil Preparation': 'Before planting',
                'Planting': 'As per season',
                'Nutrient Application': 'Based on soil test',
                'Irrigation': 'As needed',
                'Harvest': 'At maturity'
            }
        }
    };
    
    const advisory = advisoryData[data.crop_type] || advisoryData['default'];
    
    let html = `
        <div class="advisory-result">
            <div class="result-header">
                <h4>🌱 ${advisory.title}</h4>
                <p>For ${data.farm_size} acres | ${data.soil_type || (translations[currentLang]['unknown-soil'] || 'Various')} soil type</p>
            </div>
            
            <div class="recommendations-section">
                <h5>💡 ${translations[currentLang]['recommendations'] || 'Key Recommendations'}</h5>
                <ul>
    `;
    
    advisory.recommendations.forEach(rec => {
        html += `<li>${rec}</li>`;
    });
    
    html += `
                </ul>
            </div>
            
            <div class="schedule-section">
                <h5>📅 ${translations[currentLang]['schedule'] || 'Suggested Schedule'}</h5>
                <div class="schedule-grid">
    `;
    
    for (const [activity, timing] of Object.entries(advisory.schedule)) {
        html += `
            <div class="schedule-item">
                <strong>${activity}</strong>
                <span>${timing}</span>
            </div>
        `;
    }
    
    html += `
                </div>
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-secondary" onclick="window.print()">
                    <i class="fas fa-print"></i> ${translations[currentLang]['print'] || 'Print Advisory'}
                </button>
                <button class="btn btn-outline" onclick="saveAdvisory()">
                    <i class="fas fa-download"></i> ${translations[currentLang]['save-pdf'] || 'Save as PDF'}
                </button>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// Diagnoses Management
function fetchDiagnoses() {
    fetch('/api/diagnoses')
    .then(response => response.json())
    .then(data => {
        const diagnosesList = document.getElementById('diagnosesList');
        if (!diagnosesList) return;
        
        diagnosesList.innerHTML = '';
        
        if (data && data.length > 0) {
            data.forEach(diagnosis => {
                const li = document.createElement('li');
                const date = new Date(diagnosis.timestamp).toLocaleDateString();
                const severityClass = diagnosis.severity || 'unknown';
                
                li.innerHTML = `
                    <div class="diagnosis-item">
                        <div class="diagnosis-header">
                            <strong>${diagnosis.disease_name}</strong>
                            <span class="severity-${severityClass}">${diagnosis.severity}</span>
                        </div>
                        <div class="diagnosis-details">
                            <span class="confidence">${translations[currentLang]['confidence'] || 'Confidence'}: ${diagnosis.confidence.toFixed(1)}%</span>
                            <span class="date">${date}</span>
                        </div>
                    </div>
                `;
                diagnosesList.appendChild(li);
            });
        } else {
            diagnosesList.innerHTML = `<li>${translations[currentLang]['diagnoses-placeholder'] || 'No recent diagnoses available.'}</li>`;
        }
    })
    .catch(error => {
        console.error('Fetch diagnoses error:', error);
        const diagnosesList = document.getElementById('diagnosesList');
        if (diagnosesList) {
            diagnosesList.innerHTML = `<li>${translations[currentLang]['error'] || 'Error loading diagnoses'}</li>`;
        }
    });
}

// Chat Management
function toggleChat() {
    const chatWindow = document.getElementById('chatWindow');
    if (chatWindow) {
        chatWindow.style.display = chatWindow.style.display === 'flex' ? 'none' : 'flex';
        
        if (chatWindow.style.display === 'flex') {
            // Focus on input when chat opens
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                setTimeout(() => chatInput.focus(), 100);
            }
        }
    }
}

function sendMessage() {
    const input = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    
    if (!input || !chatMessages) return;
    
    const message = input.value.trim();
    if (!message) return;
    
    // Add user message
    addChatMessage(message, 'user');
    input.value = '';
    
    // Show typing indicator
    const typingId = addChatMessage('Typing...', 'bot', true);
    
    // Send to API
    fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            language: currentLang
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove typing indicator
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
        
        // Add bot response
        addChatMessage(data.response || 'Sorry, I encountered an error.', 'bot');
    })
    .catch(error => {
        console.error('Chat error:', error);
        
        // Remove typing indicator
        const typingElement = document.getElementById(typingId);
        if (typingElement) {
            typingElement.remove();
        }
        
        addChatMessage('Sorry, I am having trouble connecting.', 'bot');
    });
}

function addChatMessage(message, sender, isTyping = false) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return null;
    
    const messageId = 'msg-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `chat-message ${sender}`;
    
    if (sender === 'user') {
        messageDiv.innerHTML = `
            <div class="message-content user-message">
                <span>${message}</span>
                <small>${new Date().toLocaleTimeString()}</small>
            </div>
        `;
    } else {
        const icon = isTyping ? '⏳' : '🤖';
        const className = isTyping ? 'typing-message' : 'bot-message';
        
        messageDiv.innerHTML = `
            <div class="message-content ${className}">
                <span>${icon} ${message}</span>
                ${!isTyping ? `<small>${new Date().toLocaleTimeString()}</small>` : ''}
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Contact Form
function submitContactForm(event) {
    if (event) {
        event.preventDefault();
    }
    
    const form = document.getElementById('contactForm');
    if (!form) return;
    
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Basic validation
    if (!data.name || !data.email || !data.message) {
        showMessage('error', 'Please fill in all required fields');
        return;
    }
    
    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(data.email)) {
        showMessage('error', 'Please enter a valid email address');
        return;
    }
    
    // Show loading state
    const submitBtn = form.querySelector('[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = translations[currentLang]['loading'] || 'Sending...';
    submitBtn.disabled = true;
    
    fetch('/api/submit-contact', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            showMessage('success', 'Message sent successfully! We will get back to you soon.');
            form.reset();
        } else {
            showMessage('error', result.error || 'Failed to send message');
        }
    })
    .catch(error => {
        console.error('Contact form error:', error);
        showMessage('error', 'Failed to send message. Please try again.');
    })
    .finally(() => {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    });
}

// Voice Assistant (Mock implementation)
function startVoiceAssistant() {
    showMessage('info', 'Voice assistant feature coming soon!');
}

function startVoiceChat() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = currentLang === 'hi' ? 'hi-IN' : currentLang === 'bn' ? 'bn-BD' : 'en-US';
        
        recognition.onstart = function() {
            showMessage('info', 'Listening... Speak now');
        };
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            const chatInput = document.getElementById('chatInput');
            if (chatInput) {
                chatInput.value = transcript;
                sendMessage();
            }
        };
        
        recognition.onerror = function(event) {
            showMessage('error', 'Speech recognition error: ' + event.error);
        };
        
        recognition.start();
    } else {
        showMessage('error', 'Speech recognition not supported in this browser');
    }
}

// Utility Functions
function showMessage(type, message, duration = 5000) {
    // Remove existing messages
    const existingMessages = document.querySelectorAll('.custom-message');
    existingMessages.forEach(msg => msg.remove());
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `custom-message ${type}`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <span class="message-icon">${getMessageIcon(type)}</span>
            <span class="message-text">${message}</span>
            <button class="message-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    document.body.appendChild(messageDiv);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (messageDiv.parentElement) {
            messageDiv.remove();
        }
    }, duration);
}

function getMessageIcon(type) {
    switch(type) {
        case 'success': return '✓';
        case 'error': return '✗';
        case 'warning': return '⚠';
        case 'info': return 'ℹ';
        default: return '•';
    }
}

function openChatbot() {
    toggleChat();
}

// Initialize everything when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('KrishiMitra JavaScript initialized');
    
    // Initialize WebSocket connection
    initializeWebSocket();
    
    // Set default language based on browser or saved preference
    const savedLang = localStorage.getItem('krishimitra-language');
    const browserLang = navigator.language.split('-')[0];
    const defaultLang = savedLang || (translations[browserLang] ? browserLang : 'en');
    
    if (defaultLang !== 'en') {
        changeLanguage(defaultLang);
    }
    
    // Load initial data based on current page
    const currentPage = window.location.pathname;
    
    if (currentPage === '/' || currentPage.includes('index')) {
        // Home page - load all initial data
        refreshSensorData();
        updateWeather();
        updateMarketPrices();
    } else if (currentPage.includes('dashboard')) {
        // Dashboard - load dashboard specific data
        refreshSensorData();
        updateWeather();
        updateMarketPrices();
        fetchDiagnoses();
    } else if (currentPage.includes('advisory')) {
        const advisoryForm = document.getElementById('advisoryForm');
        if (advisoryForm) {
            advisoryForm.addEventListener('submit', getCropAdvice);
        }
    }
    
    // Set up form handlers
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', submitContactForm);
    }
    
    // Set up periodic data updates
    setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('request_data', { type: 'sensor' });
        }
    }, 30000); // Update sensor data every 30 seconds
    
    // Save language preference when changed
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const lang = this.textContent === 'EN' ? 'en' : 
                         this.textContent === 'हिं' ? 'hi' : 'bn';
            localStorage.setItem('krishimitra-language', lang);
        });
    });
    
    console.log('KrishiMitra initialization complete');
});

// Service Worker Registration (for offline functionality)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/js/sw.js')
        .then(function(registration) {
            console.log('ServiceWorker registration successful');
        })
        .catch(function(error) {
            console.log('ServiceWorker registration failed: ', error);
        });
    });
}

// Export functions for global access
window.KrishiMitra = {
    changeLanguage,
    handleImageUpload,
    diagnosePlant,
    refreshSensorData,
    updateWeather,
    updateMarketPrices,
    getCropAdvice,
    fetchDiagnoses,
    toggleChat,
    sendMessage,
    handleChatKeyPress,
    submitContactForm,
    startVoiceAssistant,
    startVoiceChat,
    showMessage
}