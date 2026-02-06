"""
WhatsApp Medical AI Bot avec 5 Modèles
Orchestration: GPT-4, Claude, DeepSeek, Qwen, Llama
"""

from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv
import asyncio
import time
from openai import OpenAI
from anthropic import Anthropic
from together import Together
from langdetect import detect, LangDetectException
import base64
from io import BytesIO

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)

# Track processed message IDs to prevent duplicates
processed_messages = set()

# Middleware pour logger toutes les requêtes
@app.before_request
def log_request_info():
    """Log toutes les requêtes entrantes"""
    if request.path != '/favicon.ico':  # Ignorer les requêtes favicon
        print(f"\n🌐 Requête reçue: {request.method} {request.path} depuis {request.remote_addr}")

# ============================================
# CONFIGURATION
# ============================================

# WhatsApp
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
VERIFY_TOKEN = WHATSAPP_TOKEN

# Clients IA
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ============================================
# DÉTECTION DE LANGUE
# ============================================

def detect_language(text):
    """Détecte la langue du texte"""
    try:
        # Clean the text
        text_clean = text.lower().strip()
        
        # Check for obvious English keywords (short messages often fail auto-detection)
        english_indicators = ['hello', 'hi', 'hey', 'what', 'how', 'can', 'help', 'please', 'thanks', 'thank you']
        french_indicators = ['bonjour', 'salut', 'merci', 'aide', 'comment', 'quoi', 'pourquoi']
        
        # Check if text contains clear language indicators
        for word in english_indicators:
            if word in text_clean:
                return "en"
        
        for word in french_indicators:
            if word in text_clean:
                return "fr"
        
        # Use langdetect for longer/unclear messages
        lang = detect(text)
        return "fr" if lang == "fr" else "en"  # Support French and English
        
    except LangDetectException:
        # Default to English if detection fails (more universal)
        return "en"


def get_system_prompt(language):
    """Retourne le prompt système dans la langue appropriée"""
    if language == "fr":
        return "Tu es un assistant médical expert. Fournis des informations médicales précises et empathiques. Rappelle toujours que tu ne remplaces pas un médecin. Réponds UNIQUEMENT en français."
    else:
        return "You are an expert medical assistant. Provide accurate and empathetic medical information. Always remind that you do not replace a doctor. Respond ONLY in English."


# ============================================
# FONCTIONS IA
# ============================================

def call_gpt4(question, language="fr", image_data=None):
    """Appelle GPT-4o-mini avec support vision"""
    try:
        print("  🤖 Appel GPT-4o-mini...")
        start = time.time()
        
        system_prompt = get_system_prompt(language)
        
        messages = [
            {
                "role": "system", 
                "content": system_prompt
            }
        ]
        
        # Build user message with image if provided
        if image_data:
            user_content = []
            # GPT-4o-mini can use base64 directly
            if image_data.get("base64"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data['base64']}"}
                })
            elif image_data.get("url"):
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data["url"]}
                })
            
            if question:
                user_content.append({"type": "text", "text": question})
            else:
                # If no text, add a default prompt for image analysis
                prompt = "Analyse cette image médicale et fournis des informations pertinentes." if language == "fr" else "Analyze this medical image and provide relevant information."
                user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": question})
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        elapsed = time.time() - start
        result = response.choices[0].message.content
        
        print(f"  ✅ GPT-4o-mini répondu en {elapsed:.2f}s")
        return {
            "model": "GPT-4o-mini",
            "response": result,
            "time": elapsed,
            "success": True
        }
    except Exception as e:
        error_msg = str(e).lower()
        print(f"  ❌ GPT-4o-mini erreur: {e}")
        
        if "429" in error_msg or "quota" in error_msg or "exceeded" in error_msg:
            return {"model": "GPT-4o-mini", "success": False, "error": "Quota épuisé - Ajoutez des crédits sur https://platform.openai.com/account/billing"}
        else:
            return {"model": "GPT-4o-mini", "success": False, "error": str(e)}


def call_claude(question, language="fr", image_data=None):
    """Appelle Claude avec support vision"""
    try:
        print("  🤖 Appel Claude...")
        start = time.time()
        
        system_prompt = get_system_prompt(language)
        
        # Build content for Claude
        content = []
        if image_data and image_data.get("base64"):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data["base64"]
                }
            })
        
        if question:
            content.append({"type": "text", "text": question})
        else:
            prompt = "Analyse cette image médicale et fournis des informations pertinentes." if language == "fr" else "Analyze this medical image and provide relevant information."
            content.append({"type": "text", "text": prompt})
        
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ]
        )
        
        elapsed = time.time() - start
        result = response.content[0].text
        
        print(f"  ✅ Claude répondu en {elapsed:.2f}s")
        return {
            "model": "Claude",
            "response": result,
            "time": elapsed,
            "success": True
        }
    except Exception as e:
        print(f"  ❌ Claude erreur: {e}")
        return {"model": "Claude", "success": False, "error": str(e)}


def call_deepseek(question, language="fr"):
    """Appelle DeepSeek"""
    try:
        print("  🤖 Appel DeepSeek...")
        start = time.time()
        
        system_prompt = get_system_prompt(language)
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        
        elapsed = time.time() - start
        data = response.json()
        result = data['choices'][0]['message']['content']
        
        print(f"  ✅ DeepSeek répondu en {elapsed:.2f}s")
        return {
            "model": "DeepSeek",
            "response": result,
            "time": elapsed,
            "success": True
        }
    except Exception as e:
        print(f"  ❌ DeepSeek erreur: {e}")
        return {"model": "DeepSeek", "success": False, "error": str(e)}


def call_qwen(question, language="fr"):
    """Appelle Qwen via Together AI"""
    try:
        print("  🤖 Appel Qwen...")
        start = time.time()
        
        system_prompt = get_system_prompt(language)
        
        response = together_client.chat.completions.create(
            model="Qwen/Qwen3-Next-80B-A3B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        elapsed = time.time() - start
        result = response.choices[0].message.content
        
        print(f"  ✅ Qwen répondu en {elapsed:.2f}s")
        return {
            "model": "Qwen",
            "response": result,
            "time": elapsed,
            "success": True
        }
    except Exception as e:
        print(f"  ❌ Qwen erreur: {e}")
        return {"model": "Qwen", "success": False, "error": str(e)}


def call_llama(question, language="fr"):
    """Appelle Llama via Together AI"""
    try:
        print("  🤖 Appel Llama...")
        start = time.time()
        
        system_prompt = get_system_prompt(language)
        
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        elapsed = time.time() - start
        result = response.choices[0].message.content
        
        print(f"  ✅ Llama répondu en {elapsed:.2f}s")
        return {
            "model": "Llama",
            "response": result,
            "time": elapsed,
            "success": True
        }
    except Exception as e:
        print(f"  ❌ Llama erreur: {e}")
        return {"model": "Llama", "success": False, "error": str(e)}


# ============================================
# GÉNÉRATION D'IMAGES
# ============================================

def generate_image_together(prompt, language="en"):
    """Génère une image avec Together AI (Stable Diffusion)"""
    try:
        print("  🎨 Génération d'image avec Together AI...")
        start = time.time()
        
        # Together AI image generation endpoint - use /images/generations
        response = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "n": 1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            # Together AI /images/generations returns image data in 'data' array
            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                # Check if it's a URL or base64
                if 'url' in image_data:
                    image_url = image_data['url']
                    elapsed = time.time() - start
                    print(f"  ✅ Image générée en {elapsed:.2f}s")
                    return {
                        "success": True,
                        "image_url": image_url,
                        "time": elapsed
                    }
                elif 'b64_json' in image_data:
                    image_base64 = image_data['b64_json']
                    elapsed = time.time() - start
                    print(f"  ✅ Image générée en {elapsed:.2f}s")
                    return {
                        "success": True,
                        "image_base64": image_base64,
                        "time": elapsed
                    }
                else:
                    raise Exception("Unexpected response format from Together AI")
            else:
                raise Exception("No image data in response")
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text) if error_data else response.text
            raise Exception(f"Together AI API error: {response.status_code} - {error_msg}")
            
    except Exception as e:
        print(f"  ❌ Erreur génération image: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def is_image_generation_request(text):
    """Détecte si l'utilisateur demande une génération d'image"""
    text_lower = text.lower().strip()
    
    # Keywords for image generation
    generation_keywords = [
        "generate", "create", "draw", "make", "show me",
        "génère", "crée", "dessine", "fais", "montre-moi",
        "image of", "picture of", "photo of",
        "image de", "photo de"
    ]
    
    return any(keyword in text_lower for keyword in generation_keywords)


# ============================================
# GÉNÉRATION D'IMAGES
# ============================================

def generate_image_together(prompt, language="en"):
    """Génère une image avec Together AI (Stable Diffusion)"""
    try:
        print("  🎨 Génération d'image avec Together AI...")
        start = time.time()
        
        # Together AI image generation endpoint - use /images/generations
        response = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "prompt": prompt,
                "width": 1024,
                "height": 1024,
                "steps": 20,
                "n": 1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            # Together AI /images/generations returns image data in 'data' array
            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                # Check if it's a URL or base64
                if 'url' in image_data:
                    image_url = image_data['url']
                    elapsed = time.time() - start
                    print(f"  ✅ Image générée en {elapsed:.2f}s")
                    return {
                        "success": True,
                        "image_url": image_url,
                        "time": elapsed
                    }
                elif 'b64_json' in image_data:
                    image_base64 = image_data['b64_json']
                    elapsed = time.time() - start
                    print(f"  ✅ Image générée en {elapsed:.2f}s")
                    return {
                        "success": True,
                        "image_base64": image_base64,
                        "time": elapsed
                    }
                else:
                    raise Exception("Unexpected response format from Together AI")
            else:
                raise Exception("No image data in response")
        else:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get('error', {}).get('message', response.text) if error_data else response.text
            raise Exception(f"Together AI API error: {response.status_code} - {error_msg}")
            
    except Exception as e:
        print(f"  ❌ Erreur génération image: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def is_image_generation_request(text):
    """Détecte si l'utilisateur demande une génération d'image"""
    text_lower = text.lower().strip()
    
    # Keywords for image generation
    generation_keywords = [
        "generate", "create", "draw", "make", "show me",
        "génère", "crée", "dessine", "fais", "montre-moi",
        "image of", "picture of", "photo of",
        "image de", "photo de"
    ]
    
    return any(keyword in text_lower for keyword in generation_keywords)


# ============================================
# ORCHESTRATEUR ET JUGE
# ============================================

def orchestrate_ai_response(question, language="fr", image_data=None):
    """
    Appelle les modèles IA en parallèle et sélectionne la meilleure réponse
    Supporte les images pour les modèles avec vision (GPT-4o-mini, Claude)
    """
    print("\n" + "="*60)
    if image_data:
        print(f"🧠 ORCHESTRATION IA - Analyse d'image")
    else:
        print(f"🧠 ORCHESTRATION IA - Question: {question[:50] if question else 'No question'}...")
    print(f"🌐 Langue détectée: {language}")
    print("="*60)
    
    # Appeler tous les modèles
    responses = []
    
    # GPT-4 (supporte vision)
    gpt4_response = call_gpt4(question, language, image_data)
    if gpt4_response["success"]:
        responses.append(gpt4_response)
    
    # Claude (supporte vision)
    claude_response = call_claude(question, language, image_data)
    if claude_response["success"]:
        responses.append(claude_response)
    
    # DeepSeek (pas de vision pour l'instant, seulement texte)
    if not image_data:
        deepseek_response = call_deepseek(question, language)
        if deepseek_response["success"]:
            responses.append(deepseek_response)
    
    # Qwen (pas de vision pour l'instant, seulement texte)
    if not image_data:
        qwen_response = call_qwen(question, language)
        if qwen_response["success"]:
            responses.append(qwen_response)
    
    # Llama (pas de vision pour l'instant, seulement texte)
    if not image_data:
        llama_response = call_llama(question, language)
        if llama_response["success"]:
            responses.append(llama_response)
    
    # Si aucune réponse
    if not responses:
        error_msg = "Désolé, tous les modèles IA sont indisponibles pour le moment. Veuillez réessayer plus tard." if language == "fr" else "Sorry, all AI models are currently unavailable. Please try again later."
        return {
            "selected_model": "Aucun",
            "response": error_msg,
            "all_responses": []
        }
    
    # Pour l'instant: sélection simple (le plus rapide qui a réussi)
    # Plus tard on ajoutera le judge GPT-4
    best_response = min(responses, key=lambda x: x["time"])
    
    print(f"\n🏆 Meilleure réponse: {best_response['model']} ({best_response['time']:.2f}s)")
    print(f"📊 Total modèles répondus: {len(responses)}/5")
    print("="*60 + "\n")
    
    return {
        "selected_model": best_response["model"],
        "response": best_response["response"],
        "all_responses": responses,
        "total_time": sum(r["time"] for r in responses)
    }


# ============================================
# DÉTECTION D'URGENCE
# ============================================

EMERGENCY_KEYWORDS = [
    "chest pain", "douleur poitrine", "can't breathe", "ne peux pas respirer",
    "unconscious", "inconscient", "severe bleeding", "saignement sévère",
    "stroke", "avc", "heart attack", "crise cardiaque",
    "suicide", "overdose", "surdose"
]

def is_emergency(message):
    """Détecte si c'est une urgence médicale"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in EMERGENCY_KEYWORDS)


# ============================================
# WHATSAPP WEBHOOK
# ============================================

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """Vérification du webhook par Meta"""
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION WEBHOOK")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Token reçu: {token[:10] + '...' if token and len(token) > 10 else token}")
    print(f"Token attendu: {VERIFY_TOKEN[:10] + '...' if VERIFY_TOKEN and len(VERIFY_TOKEN) > 10 else VERIFY_TOKEN}")
    print(f"Challenge: {challenge}")
    print("="*60)
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        print("✅ VÉRIFICATION RÉUSSIE")
        print("="*60 + "\n")
        return str(challenge), 200
    else:
        print("❌ VÉRIFICATION ÉCHOUÉE")
        if mode != 'subscribe':
            print(f"   Raison: Mode incorrect (attendu: 'subscribe', reçu: '{mode}')")
        if token != VERIFY_TOKEN:
            print(f"   Raison: Token incorrect")
        print("="*60 + "\n")
        return 'Verification failed', 403


@app.route('/webhook', methods=['POST'])
def receive_message():
    """Réception des messages WhatsApp"""
    print("\n" + "="*60)
    print("📥 WEBHOOK POST REÇU")
    print(f"📥 Headers: {dict(request.headers)}")
    print(f"📥 Method: {request.method}")
    print(f"📥 Content-Type: {request.content_type}")
    print(f"📥 Remote Address: {request.remote_addr}")
    print("="*60)
    
    try:
        # Log raw data first
        raw_data = request.data
        print(f"📥 Raw data length: {len(raw_data) if raw_data else 0} bytes")
        
        data = request.json
        if not data:
            print("⚠️ Aucune donnée JSON reçue")
            print(f"Raw data: {request.data[:200] if request.data else 'None'}")
            return jsonify({"status": "error", "message": "No data"}), 400
        
        print(f"📋 Données reçues: {str(data)[:500]}...")
        
        # Check if data has the expected structure
        if 'entry' not in data or len(data['entry']) == 0:
            print("⚠️ Pas de 'entry' dans les données")
            return jsonify({"status": "success"}), 200
        
        entry = data['entry'][0]
        
        if 'changes' not in entry or len(entry['changes']) == 0:
            print("⚠️ Pas de 'changes' dans entry")
            return jsonify({"status": "success"}), 200
        
        changes = entry['changes'][0]
        
        if 'value' not in changes:
            print("⚠️ Pas de 'value' dans changes")
            return jsonify({"status": "success"}), 200
        
        value = changes['value']
        print(f"🔍 Value keys: {list(value.keys()) if value else 'None'}")
        
        # Log the full value structure for debugging
        print(f"🔍 Full value structure: {str(value)[:1000]}")
        
        # Check if this is a message event
        if 'messages' not in value:
            # This might be a status update, read receipt, or other event
            # Vérifier s'il y a des statuts (delivery, read, etc.)
            if 'statuses' in value:
                print(f"ℹ️ Événement de statut reçu (livraison/lecture), ignoré")
                print(f"   Statuts: {value.get('statuses')}")
                return jsonify({"status": "success"}), 200
            
            print(f"⚠️ Pas de 'messages' dans value. Type d'événement: {value.get('statuses', 'unknown')}")
            print(f"⚠️ Toutes les clés dans value: {list(value.keys())}")
            return jsonify({"status": "success"}), 200
        
        # Check if messages array is empty
        if not value.get('messages') or len(value['messages']) == 0:
            print("⚠️ Tableau 'messages' vide ou inexistant")
            return jsonify({"status": "success"}), 200
        
        message_data = value['messages'][0]
        sender = message_data.get('from')
        message_id = message_data.get('id')
        message_type = message_data.get('type', 'unknown')
        timestamp = message_data.get('timestamp')
        
        # Vérifier si c'est un message sortant (de nous vers l'utilisateur)
        # Les messages sortants ont un champ 'from' qui est notre PHONE_NUMBER_ID
        if sender == PHONE_NUMBER_ID:
            print(f"ℹ️ Message sortant ignoré (c'est un message que nous avons envoyé)")
            return jsonify({"status": "success"}), 200
        
        # Vérifier si le message est trop ancien (plus de 5 minutes)
        # Cela peut arriver si Meta renvoie des messages en retard
        if timestamp:
            current_time = int(time.time())
            message_age = current_time - int(timestamp)
            if message_age > 300:  # 5 minutes
                print(f"⚠️ Message ancien ignoré (âge: {message_age}s, ID: {message_id})")
                return jsonify({"status": "success"}), 200
        
        print(f"📨 Message détecté - Type: {message_type}, Sender: {sender}, ID: {message_id}, Timestamp: {timestamp}")
        print(f"📋 Contenu du message: {str(message_data)[:300]}...")
        
        # Skip status messages (system messages, not from users)
        if sender == 'status' or not sender:
            print(f"ℹ️ Message système ignoré (type: {message_type}, sender: {sender})")
            return jsonify({"status": "success"}), 200
        
        # Vérifier si c'est un événement de statut (delivery, read, etc.)
        # Ces événements ont souvent un champ 'status' dans le message
        if 'status' in message_data:
            print(f"ℹ️ Événement de statut ignoré (type: {message_type}, status: {message_data.get('status')})")
            return jsonify({"status": "success"}), 200
        
        # Skip if no message ID (shouldn't happen, but safety check)
        if not message_id:
            print(f"⚠️ Message sans ID (type: {message_type}, sender: {sender}), ignoré")
            return jsonify({"status": "success"}), 200
        
        # Vérifier que c'est bien un message entrant avec du contenu
        # Les messages texte doivent avoir un champ 'text' avec 'body'
        if message_type == 'text' and not message_data.get('text', {}).get('body'):
            print(f"⚠️ Message texte vide ignoré (ID: {message_id})")
            return jsonify({"status": "success"}), 200
        
        # Vérifier si le message a déjà été traité
        if message_id in processed_messages:
            print(f"⚠️ Message {message_id} déjà traité, ignoré")
            print(f"📊 Taille de processed_messages: {len(processed_messages)}")
            return jsonify({"status": "success"}), 200
        
        # Ajouter le message ID à la liste des messages traités AVANT le traitement
        # Cela évite de traiter le même message deux fois si une erreur se produit
        processed_messages.add(message_id)
        print(f"📊 Message ajouté à processed_messages. Taille actuelle: {len(processed_messages)}")
        
        # Limiter la taille du set pour éviter la consommation excessive de mémoire
        # Garder les 2000 derniers messages (augmenté pour éviter les pertes)
        if len(processed_messages) > 2000:
            # Convertir en liste, garder les 1500 derniers
            print("⚠️ Limite de processed_messages atteinte, nettoyage...")
            # Créer un nouveau set avec seulement les IDs les plus récents
            # Note: Les sets ne gardent pas l'ordre, donc on garde juste une taille raisonnable
            # En pratique, on peut simplement vider et recommencer
            processed_messages.clear()
            print("📊 processed_messages nettoyé")
        
        print("\n" + "="*60)
        print(f"📨 NOUVEAU MESSAGE REÇU de {sender} (ID: {message_id})")
        print("="*60)
        
        try:
            # Handle text messages
            if message_type == 'text':
                message_text = message_data.get('text', {}).get('body', '')
                if not message_text:
                    print("⚠️ Message texte vide, ignoré")
                    return jsonify({"status": "success"}), 200
                message_text = message_data['text']['body']
                print(f"💬 Message: {message_text}")
                
                # Détecter la langue
                language = detect_language(message_text)
                print(f"🌐 Langue détectée: {language}")
                
                # Détection d'urgence
                if is_emergency(message_text):
                    print("🚨 URGENCE DÉTECTÉE!")
                    if language == "fr":
                        emergency_response = """🚨 URGENCE MÉDICALE DÉTECTÉE

Appelez immédiatement le 911 ou votre numéro d'urgence local.

En attendant les secours:
- Restez calme
- Ne bougez pas si possible
- Si seul, appelez quelqu'un

Ceci est une urgence. Contactez les services d'urgence MAINTENANT."""
                    else:
                        emergency_response = """🚨 MEDICAL EMERGENCY DETECTED

Call 911 or your local emergency number immediately.

While waiting for help:
- Stay calm
- Don't move if possible
- If alone, call someone

This is an emergency. Contact emergency services NOW."""
                    send_whatsapp_message(sender, emergency_response)
                
                # Check if user wants to generate an image
                elif is_image_generation_request(message_text):
                    print("🎨 Demande de génération d'image détectée!")
                    
                    # Send sticker first if configured
                    sticker_id = os.getenv("CRAYHEALTH_STICKER_ID")
                    if sticker_id:
                        send_whatsapp_sticker(sender, sticker_id)
                    
                    # Send loading message
                    loading_msg = "Crayhealth AI Engine generating image..." if language == "en" else "Crayhealth AI Engine génère l'image..."
                    send_whatsapp_message(sender, loading_msg)
                    
                    # Generate image
                    image_result = generate_image_together(message_text, language)
                    
                    if image_result["success"]:
                        # Send the generated image
                        if "image_url" in image_result:
                            # Send image via URL
                            send_whatsapp_image_with_caption(
                                sender,
                                image_result["image_url"],
                                "Crayhealth AI Beta - Generated Image" if language == "en" else "Crayhealth AI Beta - Image générée"
                            )
                        elif "image_base64" in image_result:
                            # Convert base64 to temporary file and upload
                            # For now, inform user (full implementation would require media upload API)
                            send_whatsapp_message(
                                sender,
                                "✅ Image générée avec succès! (Upload d'image en cours...)" if language == "fr" 
                                else "✅ Image generated successfully! (Uploading image...)"
                            )
                        else:
                            send_whatsapp_message(
                                sender,
                                "✅ Image générée!" if language == "fr" else "✅ Image generated!"
                            )
                    else:
                        error_msg = f"Désolé, je n'ai pas pu générer l'image. {image_result.get('error', 'Erreur inconnue')}" if language == "fr" else f"Sorry, I couldn't generate the image. {image_result.get('error', 'Unknown error')}"
                        send_whatsapp_message(sender, error_msg)
                
                else:
                    # Réponse IA normale - Send sticker + message
                    sticker_id = os.getenv("CRAYHEALTH_STICKER_ID")
                    
                    # Send sticker first if configured
                    if sticker_id:
                        send_whatsapp_sticker(sender, sticker_id)
                    
                    # Then send loading message
                    loading_msg = "Crayhealth AI Engine analyzing..." if language == "en" else "Crayhealth AI Engine en cours d'analyse..."
                    send_whatsapp_message(sender, loading_msg)
                    
                    result = orchestrate_ai_response(message_text, language)
                    
                    if language == "fr":
                        final_response = f"""Crayhealth AI Beta - Intelligence Médicale IA sur WhatsApp

{result['response']}

⚠️ *Disclaimer:* Ceci est une information médicale générée par IA à titre informatif seulement. Consultez toujours un professionnel de santé qualifié pour un diagnostic et un traitement appropriés. En cas d'urgence, appelez le 911."""
                    else:
                        final_response = f"""Crayhealth AI Beta - Medical Intelligence AI on WhatsApp

{result['response']}

⚠️ *Disclaimer:* This is AI-generated medical information for informational purposes only. Always consult a qualified healthcare professional for proper diagnosis and treatment. In case of emergency, call 911."""
                    
                    send_whatsapp_message(sender, final_response)
                    
            # Handle image messages
            elif message_data['type'] == 'image':
                print("📷 Image reçue!")
                
                # Get image ID and download URL
                image_id = message_data['image']['id']
                image_caption = message_data.get('image', {}).get('caption', '')
                
                print(f"🖼️ Image ID: {image_id}")
                if image_caption:
                    print(f"💬 Caption: {image_caption}")
                
                # Détecter la langue depuis le caption ou default
                language = detect_language(image_caption) if image_caption else "en"
                print(f"🌐 Langue détectée: {language}")
                
                # Download image from WhatsApp
                image_data = download_whatsapp_media(image_id)
                
                if image_data:
                    # Send sticker first if configured
                    sticker_id = os.getenv("CRAYHEALTH_STICKER_ID")
                    if sticker_id:
                        send_whatsapp_sticker(sender, sticker_id)
                    
                    # Send loading message
                    loading_msg = "Crayhealth AI Engine analyzing image..." if language == "en" else "Crayhealth AI Engine analyse l'image..."
                    send_whatsapp_message(sender, loading_msg)
                    
                    # Analyze image with AI
                    result = orchestrate_ai_response(image_caption, language, image_data=image_data)
                    
                    if language == "fr":
                        final_response = f"""Crayhealth AI Beta - Intelligence Médicale IA sur WhatsApp

{result['response']}

⚠️ *Disclaimer:* Ceci est une information médicale générée par IA à titre informatif seulement. Consultez toujours un professionnel de santé qualifié pour un diagnostic et un traitement appropriés. En cas d'urgence, appelez le 911."""
                    else:
                        final_response = f"""Crayhealth AI Beta - Medical Intelligence AI on WhatsApp

{result['response']}

⚠️ *Disclaimer:* This is AI-generated medical information for informational purposes only. Always consult a qualified healthcare professional for proper diagnosis and treatment. In case of emergency, call 911."""
                    
                    send_whatsapp_message(sender, final_response)
                else:
                    error_msg = "Désolé, je n'ai pas pu télécharger l'image. Veuillez réessayer." if language == "fr" else "Sorry, I couldn't download the image. Please try again."
                    send_whatsapp_message(sender, error_msg)
            
            # Handle other message types (audio, video, document, etc.)
            else:
                print(f"⚠️ Type de message non supporté: {message_type}")
                # Default to English for unsupported types
                unsupported_msg = f"Désolé, le type de message '{message_type}' n'est pas encore supporté. Veuillez envoyer un message texte ou une image."
                send_whatsapp_message(sender, unsupported_msg)
        
        except Exception as msg_error:
            import traceback
            print(f"\n❌ ERREUR LORS DU TRAITEMENT DU MESSAGE: {msg_error}")
            print(f"📋 Traceback:")
            traceback.print_exc()
            # Try to send error message to user
            try:
                # Default to French if language not defined
                error_lang = locals().get('language', 'fr')
                error_response = "Désolé, une erreur s'est produite lors du traitement de votre message. Veuillez réessayer." if error_lang == "fr" else "Sorry, an error occurred while processing your message. Please try again."
                send_whatsapp_message(sender, error_response)
            except:
                pass
        
        return jsonify({"status": "success"}), 200
    
    except KeyError as e:
        # Événement non-message (status, etc.)
        print(f"ℹ️ KeyError (événement non-message): {e}")
        print(f"📋 Données reçues: {str(request.json)[:300] if request.json else 'None'}")
        return jsonify({"status": "success"}), 200
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERREUR DANS receive_message: {e}")
        print(f"📋 Traceback complet:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def send_whatsapp_message(to_number, message_text):
    """Envoie un message WhatsApp"""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text}
    }
    
    try:
        print(f"📤 Envoi message à {to_number}: {message_text[:50]}...")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✅ Message WhatsApp envoyé avec succès!")
            return True
        else:
            error_data = response.json() if response.text else {}
            print(f"❌ Erreur envoi WhatsApp (status {response.status_code}): {error_data}")
            return False
    except Exception as e:
        print(f"❌ Erreur d'envoi WhatsApp: {e}")
        import traceback
        traceback.print_exc()
        return False


def send_whatsapp_sticker(to_number, sticker_id):
    """Envoie un sticker WhatsApp"""
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "sticker",
        "sticker": {
            "id": sticker_id
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✅ Sticker WhatsApp envoyé!")
            return True
        else:
            print(f"❌ Erreur envoi sticker: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Erreur d'envoi sticker: {e}")
        return False


def download_whatsapp_media(media_id):
    """Télécharge un média (image) depuis WhatsApp et retourne l'URL ou base64"""
    try:
        # Step 1: Get media URL from WhatsApp API
        url = f"https://graph.facebook.com/v18.0/{media_id}"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            media_data = response.json()
            media_url = media_data.get('url')
            
            if media_url:
                # Step 2: Download the actual image using the URL with access token
                download_headers = {
                    "Authorization": f"Bearer {WHATSAPP_TOKEN}"
                }
                
                img_response = requests.get(media_url, headers=download_headers)
                if img_response.status_code == 200:
                    # Convert to base64 for Claude (requires base64)
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    print(f"✅ Image téléchargée et convertie: {media_id}")
                    
                    # Return both URL (for GPT-4o) and base64 (for Claude)
                    # We'll store base64 in a way that both can use
                    return {
                        "url": media_url,
                        "base64": image_base64,
                        "content": img_response.content
                    }
        
        print(f"❌ Erreur téléchargement média: {response.json() if response.status_code != 200 else 'Unknown error'}")
        return None
    except Exception as e:
        print(f"❌ Exception téléchargement média: {e}")
        return None


def send_whatsapp_image_with_caption(to_number, image_url, caption_text):
    """Envoie une image WhatsApp avec une légende"""
    
    # Check if image URL is configured
    if not image_url or image_url == "None" or image_url == "":
        print("⚠️ CRAYHEALTH_LOGO_URL non configuré, envoi texte uniquement")
        send_whatsapp_message(to_number, f"🤖 {caption_text}")
        return
    
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "image",
        "image": {
            "link": image_url,
            "caption": caption_text
        }
    }
    
    try:
        print(f"📤 Tentative d'envoi image: {image_url}")
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("✅ Image WhatsApp envoyée avec succès!")
        else:
            error_data = response.json()
            print(f"❌ Erreur envoi image WhatsApp: {error_data}")
            # Fallback to text message with emoji if image fails
            send_whatsapp_message(to_number, f"🤖 {caption_text}")
    except Exception as e:
        print(f"❌ Exception lors de l'envoi image: {e}")
        # Fallback to text message with emoji if image fails
        send_whatsapp_message(to_number, f"🤖 {caption_text}")


@app.route('/test', methods=['GET', 'POST'])
def test():
    """Endpoint de test pour vérifier que le serveur fonctionne"""
    return jsonify({
        "status": "success",
        "message": "Webhook server is running",
        "timestamp": time.time(),
        "processed_messages_count": len(processed_messages),
        "config": {
            "phone_number_id": PHONE_NUMBER_ID[:10] + "..." if PHONE_NUMBER_ID else "NOT SET",
            "whatsapp_token": WHATSAPP_TOKEN[:10] + "..." if WHATSAPP_TOKEN else "NOT SET",
            "has_openai": bool(os.getenv("OPENAI_API_KEY")),
            "has_anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "has_together": bool(os.getenv("TOGETHER_API_KEY")),
            "has_deepseek": bool(DEEPSEEK_API_KEY)
        },
        "instructions": {
            "webhook_url": "http://VOTRE_IP_PUBLIQUE:5000/webhook",
            "verify_token": "Utilisez votre WHATSAPP_TOKEN",
            "note": "Assurez-vous que votre serveur est accessible depuis Internet (utilisez ngrok si en local)"
        }
    }), 200


@app.route('/')
def home():
    """Page d'accueil"""
    return """
    <html>
    <head>
        <title>Medical AI Bot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .card {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 { color: #667eea; }
            .status { 
                background: #10b981;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                display: inline-block;
                margin: 10px 0;
            }
            .models {
                background: #f3f4f6;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤖 Medical AI Bot</h1>
            <div class="status">✅ Serveur Actif</div>
            
            <h2>Configuration:</h2>
            <div class="models">
                <strong>Modèles IA Actifs:</strong><br>
                🤖 GPT-4 (OpenAI)<br>
                🤖 Claude Sonnet 4 (Anthropic)<br>
                🤖 DeepSeek<br>
                🤖 Qwen (Alibaba)<br>
                🤖 Llama 3.1 (Meta)
            </div>
            
            <h2>Features:</h2>
            <ul>
                <li>✅ Orchestration de 5 modèles IA</li>
                <li>✅ Détection d'urgence</li>
                <li>✅ Réponses en temps réel</li>
                <li>⏳ RAG médical (à venir)</li>
                <li>⏳ Judge IA avancé (à venir)</li>
            </ul>
            
            <p><strong>Numéro WhatsApp:</strong> +1 555 150 3964</p>
        </div>
    </body>
    </html>
    """


# ============================================
# DÉMARRAGE
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MEDICAL AI BOT - DÉMARRAGE")
    print("="*60)
    print(f"\n📱 WhatsApp: +1 555 150 3964")
    print(f"🤖 Modèles IA: GPT-4, Claude, DeepSeek, Qwen, Llama")
    print(f"🌐 Port: 5000")
    
    # Vérification des variables d'environnement
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION DE LA CONFIGURATION")
    print("="*60)
    print(f"PHONE_NUMBER_ID: {'✅ Configuré' if PHONE_NUMBER_ID else '❌ MANQUANT'}")
    print(f"WHATSAPP_TOKEN: {'✅ Configuré' if WHATSAPP_TOKEN else '❌ MANQUANT'}")
    print(f"OPENAI_API_KEY: {'✅ Configuré' if os.getenv('OPENAI_API_KEY') else '❌ MANQUANT'}")
    print(f"ANTHROPIC_API_KEY: {'✅ Configuré' if os.getenv('ANTHROPIC_API_KEY') else '❌ MANQUANT'}")
    print(f"TOGETHER_API_KEY: {'✅ Configuré' if os.getenv('TOGETHER_API_KEY') else '❌ MANQUANT'}")
    print(f"DEEPSEEK_API_KEY: {'✅ Configuré' if DEEPSEEK_API_KEY else '❌ MANQUANT'}")
    print("="*60)
    
    if not PHONE_NUMBER_ID or not WHATSAPP_TOKEN:
        print("\n⚠️ ATTENTION: PHONE_NUMBER_ID ou WHATSAPP_TOKEN manquants!")
        print("⚠️ Le webhook ne pourra pas envoyer de messages WhatsApp.")
        print("⚠️ Vérifiez votre fichier .env\n")
    
    print("\n💡 Pour tester le serveur, visitez: http://localhost:5000/test")
    print("💡 Webhook URL: http://VOTRE_IP:5000/webhook")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)