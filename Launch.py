import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import time
import colorama
from colorama import Fore, Style, Back
import sys
import threading
import re
import json
import numpy as np
from collections import deque

# Initialisation des couleurs
colorama.init()

# Configuration - MODIFIÉ POUR UTILISER LE MODÈLE FINE-TUNÉ
# Utilisons le chemin absolu pour éviter les erreurs
MODEL_PATH = os.path.abspath("FineTune/mistral-finetuned")

print(f"Chemin du modèle: {MODEL_PATH}")

# Token non nécessaire pour un modèle local
os.environ["HF_TOKEN"] = ""

# Système de raisonnement avancé - catégories et mots-clés
CATEGORIES = {
    "SALUTATION": ["bonjour", "salut", "bonsoir", "hello", "coucou", "hey", "yo", "wesh"],
    "RECHERCHE_PRODUIT": ["cherche", "trouver", "recherche", "où", "produit", "article", "acheter", "vendre", "achat"],
    "COMPTE_UTILISATEUR": ["compte", "inscription", "connecter", "mot de passe", "profil", "identifiant", "connexion"],
    "COMMANDE": ["commande", "commander", "achat", "acheter", "panier", "caddie", "achats"],
    "PAIEMENT": ["payer", "paiement", "carte", "bancaire", "paypal", "prix", "coût", "euros", "€", "tarif"],
    "LIVRAISON": ["livrer", "livraison", "expédition", "délai", "quand", "colis", "recevoir", "envoi"],
    "RETOUR": ["retour", "renvoyer", "rembourser", "remboursement", "annuler", "annulation", "renvoie"],
    "SERVICE_CLIENT": ["problème", "aide", "assister", "contacter", "contact", "service client", "assistance"],
    "PRODUIT_SPECIFIQUE": ["smartphone", "ordinateur", "vêtement", "meuble", "télévision", "chaussure", "électronique"],
    "CONVERSATION": ["comment ça va", "qui es-tu", "merci", "au revoir", "bonne journée", "t'es qui", "tu es qui"],
    "INFORMATION_SITE": ["site", "entreprise", "magasin", "boutique", "marketplace", "shop", "web"],
    "FIN_CONVERSATION": ["au revoir", "bye", "à bientôt", "adieu", "terminé", "fin", "salut", "ciao"],
    "OPINION": ["penses", "crois", "avis", "opinion", "préfères", "meilleur", "idée", "suggestion"]
}

# Dictionnaire inverse pour le lookup rapide des mots-clés vers catégories
KEYWORD_TO_CATEGORY = {}
for category, keywords in CATEGORIES.items():
    for keyword in keywords:
        if keyword not in KEYWORD_TO_CATEGORY:
            KEYWORD_TO_CATEGORY[keyword] = []
        KEYWORD_TO_CATEGORY[keyword].append(category)

# Modèles de réponse pour assurer la cohérence
RESPONSE_TEMPLATES = {
    "SALUTATION": [
        "Bonjour ! Comment puis-je vous aider avec vos achats aujourd'hui ?",
        "Salut ! Ravi de vous parler. Que puis-je faire pour vous ?",
        "Bonjour ! Je suis là pour répondre à vos questions sur notre site e-commerce."
    ],
    "FIN_CONVERSATION": [
        "Au revoir ! N'hésitez pas à revenir si vous avez d'autres questions.",
        "À bientôt ! Votre panier sera sauvegardé pour votre prochaine visite.",
        "Merci de votre visite ! J'espère avoir pu vous aider aujourd'hui."
    ]
}

# Empreintes vectorielles simplifiées pour certains thèmes
THEME_VECTORS = {
    "produit": np.array([1.0, 0.2, 0.1, 0.3, 0.0]),
    "paiement": np.array([0.2, 1.0, 0.3, 0.1, 0.0]),
    "livraison": np.array([0.1, 0.3, 1.0, 0.2, 0.0]),
    "compte": np.array([0.3, 0.1, 0.2, 1.0, 0.0]),
    "conversation": np.array([0.0, 0.0, 0.0, 0.0, 1.0])
}


class AdvancedReasoningSystem:
    def __init__(self):
        # Historique complet de la conversation
        self.conversation_history = []

        # File d'attente pour le contexte immédiat (derniers échanges)
        self.recent_context = deque(maxlen=5)

        # Données utilisateur persistantes
        self.user_data = {
            "interests": set(),  # Centre d'intérêts
            "products_viewed": set(),  # Produits consultés
            "issues": set(),  # Problèmes rencontrés
            "sentiment": "neutral",  # Sentiment général
            "name": None,  # Nom si mentionné
            "conversation_topics": {}  # Fréquence des sujets abordés
        }

        # État actuel de la conversation
        self.current_state = {
            "topic": None,
            "subtopic": None,
            "needs_clarification": False,
            "expected_response_type": None,
            "topic_vector": np.zeros(5)  # Représentation vectorielle du thème actuel
        }

        # Mécanisme d'auto-correction
        self.correction_history = []

        # Log des raisonnements
        self.reasoning_log = []

    def add_to_history(self, user_input, response):
        """Ajoute un échange à l'historique et met à jour le contexte récent"""
        exchange = {"user": user_input, "assistant": response, "timestamp": time.time()}
        self.conversation_history.append(exchange)
        self.recent_context.append(exchange)

        # Mettre à jour les données utilisateur
        self._update_user_data(user_input)

        # Mettre à jour le vecteur de thème actuel
        self._update_topic_vector(user_input)

    def _update_user_data(self, text):
        """Mise à jour des données utilisateur basées sur le texte"""
        text_lower = text.lower()

        # Détection d'intérêts
        for category, keywords in CATEGORIES.items():
            if category in ["PRODUIT_SPECIFIQUE", "RECHERCHE_PRODUIT"]:
                for keyword in keywords:
                    if keyword in text_lower:
                        self.user_data["interests"].add(keyword)

        # Comptage des sujets de conversation
        category = self.categorize_query(text)
        if category in self.user_data["conversation_topics"]:
            self.user_data["conversation_topics"][category] += 1
        else:
            self.user_data["conversation_topics"][category] = 1

        # Analyse de sentiment basique
        positive_words = ["merci", "super", "génial", "content", "satisfait", "excellent"]
        negative_words = ["problème", "erreur", "insatisfait", "déçu", "mauvais", "difficile"]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            self.user_data["sentiment"] = "positive"
        elif neg_count > pos_count:
            self.user_data["sentiment"] = "negative"

    def _update_topic_vector(self, text):
        """Met à jour le vecteur représentant le thème actuel de la conversation"""
        text_lower = text.lower()

        # Réinitialiser légèrement le vecteur pour permettre l'évolution du thème
        self.current_state["topic_vector"] *= 0.8

        # Ajouter la contribution du message actuel
        for theme, vector in THEME_VECTORS.items():
            if any(keyword in text_lower for keyword in CATEGORIES.get(theme.upper(), [])):
                self.current_state["topic_vector"] += 0.5 * vector

        # Normaliser le vecteur
        norm = np.linalg.norm(self.current_state["topic_vector"])
        if norm > 0:
            self.current_state["topic_vector"] /= norm

    def categorize_query(self, query):
        """Version améliorée de la catégorisation des requêtes"""
        query_lower = query.lower()

        # Comptage des mots-clés par catégorie
        category_scores = {}

        # Analyse basée sur les mots-clés
        for word in query_lower.split():
            if word in KEYWORD_TO_CATEGORY:
                for category in KEYWORD_TO_CATEGORY[word]:
                    category_scores[category] = category_scores.get(category, 0) + 1

        # Analyse basée sur des expressions plus longues
        for category, keywords in CATEGORIES.items():
            for keyword in keywords:
                if len(keyword.split()) > 1 and keyword in query_lower:
                    category_scores[category] = category_scores.get(category,
                                                                    0) + 2  # Poids plus élevé pour les expressions

        # Si c'est une question explicite
        if "?" in query and not category_scores:
            # Détecter le type de question basé sur les mots interrogatifs
            question_words = ["comment", "pourquoi", "quand", "où", "qui", "quoi", "quel", "quelle", "quels", "quelles"]
            for word in question_words:
                if word in query_lower:
                    if any(k in query_lower for k in CATEGORIES["RECHERCHE_PRODUIT"]):
                        category_scores["RECHERCHE_PRODUIT"] = category_scores.get("RECHERCHE_PRODUIT", 0) + 1
                    elif any(k in query_lower for k in CATEGORIES["LIVRAISON"]):
                        category_scores["LIVRAISON"] = category_scores.get("LIVRAISON", 0) + 1
                    elif any(k in query_lower for k in CATEGORIES["PAIEMENT"]):
                        category_scores["PAIEMENT"] = category_scores.get("PAIEMENT", 0) + 1
                    break

        # Vérifier la proximité vectorielle avec le thème précédent pour la continuité
        prev_category = self.current_state.get("topic")
        if prev_category and prev_category in category_scores:
            # Bonus pour maintenir la cohérence thématique
            category_scores[prev_category] += 0.5

        # Si aucune catégorie n'est identifiée ou que le score est faible
        if not category_scores or max(category_scores.values(), default=0) < 0.7:
            # Vérifier s'il s'agit d'une salutation ou d'une fin de conversation
            if any(word in query_lower for word in CATEGORIES["SALUTATION"]) and len(query_lower) < 20:
                return "SALUTATION"
            elif any(word in query_lower for word in CATEGORIES["FIN_CONVERSATION"]) and len(query_lower) < 20:
                return "FIN_CONVERSATION"
            else:
                # Continuité par défaut ou conversation générale
                return "CONVERSATION"

        # Retourner la catégorie avec le score le plus élevé
        return max(category_scores, key=category_scores.get)

    def extract_key_entities(self, query):
        """Extrait les entités clés de la requête"""
        entities = {
            "products": [],
            "actions": [],
            "concerns": [],
            "locations": [],
            "time_references": []
        }

        query_lower = query.lower()

        # Extraction de produits
        for product in CATEGORIES["PRODUIT_SPECIFIQUE"]:
            if product in query_lower:
                entities["products"].append(product)

        # Extraction d'actions
        action_words = ["acheter", "commander", "trouver", "chercher", "payer", "annuler", "retourner", "échanger"]
        for action in action_words:
            if action in query_lower:
                entities["actions"].append(action)

        # Extraction de préoccupations
        concern_words = ["problème", "défaut", "retard", "erreur", "question", "inquiétude", "livraison", "délai"]
        for concern in concern_words:
            if concern in query_lower:
                entities["concerns"].append(concern)

        # Extraction simple de références temporelles
        time_refs = ["aujourd'hui", "demain", "bientôt", "maintenant", "après", "avant", "délai", "attendre"]
        for time_ref in time_refs:
            if time_ref in query_lower:
                entities["time_references"].append(time_ref)

        return entities

    def analyze_context_relevance(self, query, category):
        """Analyse la pertinence du contexte récent pour la requête actuelle"""
        if not self.recent_context:
            return {
                "relevance": "no_context",
                "continuity": 0.0,
                "reference_to_previous": False
            }

        query_lower = query.lower()

        # Vérifier les références explicites aux messages précédents
        reference_words = ["tu as dit", "tu mentionnais", "comme tu disais", "précédemment", "avant", "tu parlais"]
        explicit_reference = any(ref in query_lower for ref in reference_words)

        # Vérifier la continuité thématique
        last_exchange = self.recent_context[-1]
        last_category = self.categorize_query(last_exchange["user"])

        # Calcul de similarité basé sur les catégories
        category_continuity = 1.0 if category == last_category else 0.0

        # Calcul de similarité basé sur les mots partagés (simple)
        last_words = set(last_exchange["user"].lower().split())
        current_words = set(query_lower.split())
        word_overlap = len(last_words.intersection(current_words)) / max(1, len(current_words))

        # Combinaison des scores
        continuity_score = 0.7 * category_continuity + 0.3 * word_overlap

        if explicit_reference:
            return {
                "relevance": "explicit_reference",
                "continuity": continuity_score,
                "reference_to_previous": True
            }
        elif continuity_score > 0.3:
            return {
                "relevance": "thematic_continuity",
                "continuity": continuity_score,
                "reference_to_previous": False
            }
        else:
            return {
                "relevance": "new_topic",
                "continuity": continuity_score,
                "reference_to_previous": False
            }

    def determine_response_approach(self, query, category, entities, context_analysis):
        """Détermine l'approche de réponse en fonction de l'analyse"""
        approach = {
            "tone": "neutral",
            "format": "standard",
            "focus": category.lower(),
            "needs_clarification": False,
            "should_verify_understanding": False
        }

        # Ajustement du ton en fonction du sentiment utilisateur
        if self.user_data["sentiment"] == "positive":
            approach["tone"] = "friendly"
        elif self.user_data["sentiment"] == "negative":
            approach["tone"] = "helpful"

        # Ajustement du format en fonction de la requête
        if category == "RECHERCHE_PRODUIT" and len(entities["products"]) == 0:
            approach["needs_clarification"] = True
            approach["format"] = "question"
        elif category == "OPINION":
            approach["format"] = "balanced"
        elif category in ["LIVRAISON", "PAIEMENT", "RETOUR"]:
            approach["format"] = "informative"

        # Ajustement si référence au contexte précédent
        if context_analysis["reference_to_previous"]:
            approach["should_verify_understanding"] = True

        # Ajout de structure pour les questions complexes
        query_words = query.lower().split()
        if len(query_words) > 10 and "?" in query:
            approach["format"] = "structured"

        return approach

    def detect_offensive_content(self, text):
        """
        Détecte le contenu offensant ou inapproprié dans le texte utilisateur.
        Retourne un dictionnaire avec les résultats de l'analyse.
        """
        text_lower = text.lower()

        # Liste de mots et expressions offensants
        # Cette liste peut être étendue selon les besoins
        offensive_words = [
            # Insultes directes
            "connard", "connasse", "con", "salope", "pute", "putain", "enculé", "encule",
            "fils de pute", "ta gueule", "ta mère", "va te faire", "ntm", "fdp", "pd", "tg",
            "nique", "niquer", "bite", "couilles", "merde", "crétin", "débile", "abruti",

            # Expressions désobligeantes
            "ferme la", "ta race", "sale", "dégueulasse", "idiot", "imbécile", "bâtard",

            # Discriminations
            "negro", "négro", "nègre", "bougnoule", "pédé", "pd", "gouine", "travelo",
            "tantouze", "tapette", "transsexuel", "sale juif", "sale arabe", "sale noir",

            # Menaces
            "je vais te", "je te tue", "je te retrouve", "je vais te retrouver",

            # Expressions grossières
            "va chier", "va te faire foutre", "mange tes morts", "suce ma"
        ]

        # Détection précise pour éviter les faux positifs
        # Par exemple, "constitution" contient "con" mais n'est pas offensant
        detected_words = []
        for word in offensive_words:
            # Si le mot offensant est entouré d'espaces ou de ponctuation
            pattern = r'(\s|^|[,.!?;:\'\"]){0}(\s|$|[,.!?;:\'\"])'.format(word)
            if re.search(pattern, text_lower, re.IGNORECASE) or word in text_lower.split():
                detected_words.append(word)

        # Analyse du niveau d'hostilité
        hostility_level = 0
        if detected_words:
            # Plus il y a de mots offensants, plus le niveau d'hostilité est élevé
            hostility_level = min(1.0, len(detected_words) * 0.2)  # Plafonné à 1.0

            # Certains mots indiquent une hostilité plus élevée
            high_hostility_words = ["fils de pute", "je te tue", "je vais te retrouver", "nique"]
            if any(word in detected_words for word in high_hostility_words):
                hostility_level = 1.0

        # Analyse d'autres signes d'hostilité (majuscules excessives, ponctuation excessive)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        if caps_ratio > 0.5 and len(text) > 5:  # Si plus de 50% en majuscules
            hostility_level += 0.2

        exclamation_count = text.count('!')
        if exclamation_count > 3:  # Beaucoup de points d'exclamation indiquent de l'agressivité
            hostility_level += 0.1

        # Plafonner à 1.0
        hostility_level = min(1.0, hostility_level)

        # Classification du contenu offensant
        offensive_type = None
        if detected_words:
            if any(word in ["connard", "con", "abruti", "débile", "idiot"] for word in detected_words):
                offensive_type = "insulte_intelligence"
            elif any(word in ["salope", "pute", "putain", "enculé"] for word in detected_words):
                offensive_type = "insulte_sexuelle"
            elif any(word in ["negro", "négro", "nègre", "bougnoule", "sale juif", "sale arabe"] for word in
                     detected_words):
                offensive_type = "discrimination_raciale"
            elif any(word in ["je vais te", "je te tue", "je te retrouve"] for word in detected_words):
                offensive_type = "menace"
            else:
                offensive_type = "insulte_générale"

        result = {
            "is_offensive": len(detected_words) > 0,
            "offensive_words": detected_words,
            "hostility_level": hostility_level,
            "offensive_type": offensive_type,
            "caps_ratio": caps_ratio,
            "exclamation_count": exclamation_count
        }

        return result

    def determine_response_to_offensive_content(self, offensive_analysis):
        """
        Détermine la réponse appropriée au contenu offensant.
        """
        if not offensive_analysis["is_offensive"]:
            return None

        hostility = offensive_analysis["hostility_level"]
        offensive_type = offensive_analysis["offensive_type"]

        if hostility > 0.8:
            # Contenu très offensant - réponse ferme mais professionnelle
            return {
                "response_type": "firm_boundary",
                "message": "Je ne peux pas continuer cette conversation avec ce type de langage. Je suis là pour vous aider de manière respectueuse. Si vous souhaitez poursuivre, merci d'utiliser un langage approprié."
            }
        elif hostility > 0.5:
            # Contenu modérément offensant - rappel des règles
            return {
                "response_type": "polite_reminder",
                "message": "Je vous invite à utiliser un langage plus respectueux pour que je puisse vous aider efficacement. Comment puis-je vous assister aujourd'hui ?"
            }
        else:
            # Contenu légèrement offensant - réorientation positive
            return {
                "response_type": "redirection",
                "message": "Je suis là pour vous aider. Comment puis-je vous être utile concernant notre site e-commerce ?"
            }

    def verify_response_consistency(self, query, proposed_response, category):
        """Vérifie que la réponse est cohérente avec la requête et la catégorie"""
        consistency = {
            "is_consistent": True,
            "issues": [],
            "suggestions": []
        }

        # Vérification de cohérence basique
        if category == "SALUTATION" and len(proposed_response) > 200:
            consistency["is_consistent"] = False
            consistency["issues"].append("La réponse à une salutation est trop longue")
            consistency["suggestions"].append("Réduire la réponse à une salutation simple")

        # Vérification de présence de contenu non demandé
        if category == "SALUTATION" and any(
                keyword in proposed_response.lower() for keyword in ["cadeau", "promotion", "offre"]):
            consistency["is_consistent"] = False
            consistency["issues"].append("La réponse contient des informations non demandées")
            consistency["suggestions"].append("Se limiter à une salutation cordiale")

        # Vérification de correspondance à la question
        if "?" in query:
            # Vérifier si la réponse contient des informations pertinentes
            question_topic = "inconnu"
            for cat, keywords in CATEGORIES.items():
                if any(keyword in query.lower() for keyword in keywords):
                    question_topic = cat.lower()
                    break

            # Vérifie si la réponse aborde bien le sujet de la question
            if question_topic != "inconnu" and not any(
                    keyword in proposed_response.lower() for keyword in CATEGORIES.get(question_topic.upper(), [])):
                consistency["is_consistent"] = False
                consistency["issues"].append(
                    f"La réponse ne semble pas aborder le sujet de la question ({question_topic})")
                consistency["suggestions"].append(f"Inclure des informations sur {question_topic}")

        return consistency

    def format_reasoning_for_prompt(self, query):
        """Génère un raisonnement structuré pour guider la génération de réponse"""
        # 1. Catégorisation
        category = self.categorize_query(query)

        # 2. Extraction d'entités
        entities = self.extract_key_entities(query)

        # 3. Analyse du contexte
        context_analysis = self.analyze_context_relevance(query, category)

        # 4. Détermination de l'approche
        approach = self.determine_response_approach(query, category, entities, context_analysis)

        # Construction du raisonnement structuré
        reasoning_steps = [
            f"1. CATÉGORISATION: Cette question appartient à la catégorie {category}.",
            f"2. CONTEXTE: {context_analysis['relevance'].replace('_', ' ').title()}. "
            f"Niveau de continuité avec l'échange précédent: {context_analysis['continuity']:.1f}/1.0.",
            f"3. ENTITÉS DÉTECTÉES:"
        ]

        # Ajouter les entités détectées
        for entity_type, entity_values in entities.items():
            if entity_values:
                reasoning_steps.append(f"   - {entity_type.capitalize()}: {', '.join(entity_values)}")

        # Ajouter l'approche de réponse
        reasoning_steps.append(f"4. APPROCHE DE RÉPONSE:")
        reasoning_steps.append(f"   - Ton: {approach['tone']}")
        reasoning_steps.append(f"   - Format: {approach['format']}")
        reasoning_steps.append(f"   - Focus: {approach['focus']}")

        if approach['needs_clarification']:
            reasoning_steps.append(f"   - BESOIN DE CLARIFICATION: Oui")

        # Ajouter le contexte utilisateur pertinent
        reasoning_steps.append(f"5. CONTEXTE UTILISATEUR:")
        if self.user_data["interests"]:
            reasoning_steps.append(f"   - Intérêts détectés: {', '.join(self.user_data['interests'])}")
        reasoning_steps.append(f"   - Sentiment utilisateur: {self.user_data['sentiment']}")

        # Ajouter des recommandations spécifiques
        reasoning_steps.append(f"6. RECOMMANDATIONS:")

        if category == "SALUTATION":
            reasoning_steps.append(f"   - Garder la réponse brève et accueillante")
            reasoning_steps.append(f"   - Ne pas introduire de sujets non sollicités")
        elif category == "RECHERCHE_PRODUIT":
            if not entities["products"]:
                reasoning_steps.append(f"   - Demander plus de précisions sur le type de produit recherché")
            else:
                reasoning_steps.append(f"   - Fournir des informations sur {', '.join(entities['products'])}")

        # Ajouter des contraintes anti-hallucination pour les cas spécifiques
        if category == "SALUTATION":
            reasoning_steps.append(f"7. CONTRAINTES:")
            reasoning_steps.append(f"   - Ne pas inventer de besoins ou de questions de l'utilisateur")
            reasoning_steps.append(f"   - Se limiter à une salutation et une offre d'aide générale")
            reasoning_steps.append(f"   - Ne pas supposer ce que l'utilisateur veut sans qu'il l'ait mentionné")

        # Stocker le raisonnement dans le log
        reasoning_dict = {
            "timestamp": time.time(),
            "query": query,
            "category": category,
            "context_analysis": context_analysis,
            "entities": entities,
            "approach": approach,
            "reasoning_steps": reasoning_steps
        }
        self.reasoning_log.append(reasoning_dict)

        return "\n".join(reasoning_steps), category, approach


def print_slowly(text, delay=0.001):
    """Affiche le texte caractère par caractère pour simuler une réponse en temps réel"""
    for char in text:
        print(char, end='', flush=True)
        # Pause plus longue après les sauts de ligne pour un effet plus naturel
        if char == '\n':
            time.sleep(delay * 5)  # Pause plus longue pour les sauts de ligne
        else:
            time.sleep(delay)
    print()  # Saut de ligne final


def format_response_text(text):
    """Améliore le formatage du texte pour qu'il soit plus lisible"""

    # Si le texte est court ou contient déjà des sauts de ligne, on le retourne tel quel
    if len(text) < 100 or '\n' in text:
        return text

    # Sinon, on ajoute des sauts de ligne pour améliorer la lisibilité
    formatted_text = ""
    buffer = ""

    # Ajoute un saut de ligne après chaque phrase
    for i, char in enumerate(text):
        buffer += char

        # Détecte la fin d'une phrase (point, point d'exclamation ou d'interrogation
        # suivi d'un espace ou en fin de texte)
        if char in ['.', '!', '?'] and (i + 1 < len(text) and text[i + 1] == ' ' or i + 1 == len(text)):
            formatted_text += buffer
            buffer = ""
            if i + 1 < len(text):  # S'assurer qu'on n'est pas à la fin du texte
                formatted_text += "\n\n"  # Ajouter un double saut de ligne entre les phrases

    # Ajouter ce qui reste dans le buffer
    formatted_text += buffer

    return formatted_text


def print_header():
    """Affiche l'en-tête de l'interface de chat"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Efface l'écran
    print(f"{Back.BLUE}{Fore.WHITE} MON POTE MISTRAL - SYSTÈME DE RAISONNEMENT AVANCÉ {Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Modèle: {MODEL_PATH}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


# Fonction d'animation pendant la génération
def thinking_animation(stop_event):
    """Animation à afficher pendant que le modèle génère une réponse"""
    animations = [
        "🧠 ⚙️ ... ",
        "🧠  ⚙️... ",
        "🧠   ⚙️.. ",
        "🧠    ⚙️. ",
        "🧠     ⚙️ ",
        "🧠    ⚙️. ",
        "🧠   ⚙️.. ",
        "🧠  ⚙️... ",
    ]
    i = 0
    while not stop_event.is_set():
        print(f"\r{Fore.YELLOW}Ton pote réfléchit {animations[i]}{Style.RESET_ALL}", end="", flush=True)
        i = (i + 1) % len(animations)
        time.sleep(0.1)
    # Nettoyer la ligne une fois terminé
    print("\r" + " " * 50, end="\r")


def correct_response_if_needed(reasoning_system, query, response, category, approach):
    """Corrige la réponse si elle n'est pas cohérente avec la demande"""
    # Pour les salutations, vérifier que la réponse est simple et appropriée
    if category == "SALUTATION" and len(response) > 200:
        # La réponse à une salutation est trop longue, on la remplace
        corrected = "Bonjour ! Je suis là pour vous aider avec vos questions sur notre site e-commerce. Comment puis-je vous être utile aujourd'hui ?"
        print(f"{Fore.RED}[Correction auto: Réponse simplifiée pour salutation]{Style.RESET_ALL}")
        return corrected

    # Vérifier si la réponse contient des informations non sollicitées pour une salutation
    if category == "SALUTATION" and ("cadeau" in response.lower() or "noël" in response.lower()):
        # La réponse contient des informations non sollicitées
        corrected = "Bonjour ! Je suis ravi de vous accueillir. Comment puis-je vous aider aujourd'hui ?"
        print(f"{Fore.RED}[Correction auto: Suppression de contenu non sollicité]{Style.RESET_ALL}")
        return corrected

    return response


def main():
    # Initialisation du système de raisonnement avancé
    reasoning_system = AdvancedReasoningSystem()

    # Mode debug pour afficher le raisonnement
    debug_mode = False

    # Vérification de la disponibilité du GPU
    if torch.cuda.is_available():
        print(f"{Fore.GREEN}GPU disponible: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Pas de GPU disponible, utilisation du CPU{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}Chargement du modèle personnalisé depuis {MODEL_PATH}...{Style.RESET_ALL}")

    # Configuration pour la quantification 8-bit
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Chargement du tokenizer - Modèle local, pas besoin de token
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True  # Indique explicitement d'utiliser uniquement les fichiers locaux
    )

    # Options pour optimiser pour une 3080 (environ 10GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,  # Utilisation de la précision mixte
        quantization_config=quantization_config,
        local_files_only=True  # Indique explicitement d'utiliser uniquement les fichiers locaux
    )

    # Créer le pipeline de génération de texte - SANS spécifier device
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id  # Évite le message de warning
    )

    # Interface de chat améliorée
    chat_history = []

    # Affichage de l'interface
    print_header()
    print(
        f"{Fore.WHITE}Tapez {Fore.GREEN}'aide'{Fore.WHITE} pour les commandes ou {Fore.RED}'quitter'{Fore.WHITE} pour terminer.{Style.RESET_ALL}\n")
    print(f"{Fore.CYAN}Essayez de lui parler de façon amicale pour voir sa nouvelle personnalité!{Style.RESET_ALL}\n")

    while True:
        # Interface utilisateur avec indicateur visuel
        try:
            print(f"{Fore.CYAN}Vous:{Style.RESET_ALL} ", end='', flush=True)
            user_input = input()

            offensive_analysis = reasoning_system.detect_offensive_content(user_input)
            if offensive_analysis["is_offensive"]:
                if debug_mode:
                    print(f"\n{Fore.RED}Contenu offensant détecté:{Style.RESET_ALL}")
                    print(
                        f"{Fore.RED}Mots détectés: {', '.join(offensive_analysis['offensive_words'])}{Style.RESET_ALL}")
                    print(
                        f"{Fore.RED}Niveau d'hostilité: {offensive_analysis['hostility_level']:.2f}/1.0{Style.RESET_ALL}")
                    print(f"{Fore.RED}Type d'offense: {offensive_analysis['offensive_type']}{Style.RESET_ALL}\n")

                # Déterminer la réponse appropriée
                offensive_response = reasoning_system.determine_response_to_offensive_content(offensive_analysis)

                # Afficher la réponse préprogrammée au lieu de générer
                print(f"\n{Fore.GREEN}Ton pote:{Style.RESET_ALL}")
                print_slowly(offensive_response["message"])
                print(f"\n{Fore.BLUE}[Réponse automatique à un contenu inapproprié]{Style.RESET_ALL}\n")

                # Mettre à jour l'historique avec cette interaction
                chat_history.append((user_input, offensive_response["message"]))
                reasoning_system.add_to_history(user_input, offensive_response["message"])

                # Passer au message suivant
                continue

            # Validation d'entrée - s'assurer que l'entrée n'est pas vide
            if not user_input.strip():
                print(f"{Fore.YELLOW}Veuillez entrer un message non vide.{Style.RESET_ALL}")
                continue

            # Commandes spéciales
            if user_input.lower() in ["quitter", "exit", "bye"]:
                print(f"\n{Fore.YELLOW}Au revoir mon pote!{Style.RESET_ALL}")
                break
            elif user_input.lower() in ["aide", "help"]:
                print(f"\n{Fore.GREEN}Commandes disponibles:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}- quitter:{Style.RESET_ALL} Termine la conversation")
                print(f"{Fore.YELLOW}- aide:{Style.RESET_ALL} Affiche cette aide")
                print(f"{Fore.YELLOW}- effacer:{Style.RESET_ALL} Efface l'historique de conversation")
                print(f"{Fore.YELLOW}- reload:{Style.RESET_ALL} Recharge l'interface")
                print(
                    f"{Fore.YELLOW}- debug:{Style.RESET_ALL} Active/désactive le mode debug (affiche le raisonnement)")
                print(f"{Fore.YELLOW}- debug+:{Style.RESET_ALL} Mode debug avec détails avancés")
                continue
            elif user_input.lower() == "effacer":
                chat_history = []
                reasoning_system = AdvancedReasoningSystem()  # Réinitialise le système de raisonnement
                print(f"\n{Fore.GREEN}Historique de conversation effacé.{Style.RESET_ALL}")
                continue
            elif user_input.lower() == "reload":
                print_header()
                continue
            elif user_input.lower() == "debug":
                debug_mode = not debug_mode
                print(f"\n{Fore.GREEN}Mode debug {'activé' if debug_mode else 'désactivé'}.{Style.RESET_ALL}")
                continue
            elif user_input.lower() == "debug+":
                debug_mode = True
                print(f"\n{Fore.GREEN}Mode debug avancé activé.{Style.RESET_ALL}")
                continue

            # Appliquer le système de raisonnement avancé
            reasoning, category, approach = reasoning_system.format_reasoning_for_prompt(user_input)

            # Afficher le raisonnement en mode debug
            if debug_mode:
                print(f"\n{Fore.MAGENTA}{'=' * 40} ANALYSE {'=' * 40}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}{reasoning}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}{'=' * 87}{Style.RESET_ALL}\n")

            # Format d'instruction Mistral pour le modèle fine-tuné
            # On inclut le raisonnement dans le prompt pour guider la génération
            prompt = f"<s>[INST] "

            if debug_mode:
                prompt += f"Analyse:\n{reasoning}\n\n"
                # Ajouter des instructions spécifiques basées sur la catégorie
                if category == "SALUTATION":
                    prompt += "IMPORTANT: Reste simple et concis pour cette salutation. Ne suppose pas ce que l'utilisateur veut sans qu'il l'ait mentionné. Ne parle pas de cadeaux ou d'achats si l'utilisateur n'en a pas parlé.\n\n"

            prompt += f"{user_input} [/INST]"

            # Animation de chargement avec un thread séparé
            stop_animation = threading.Event()
            animation_thread = threading.Thread(target=thinking_animation, args=(stop_animation,))
            animation_thread.daemon = True
            animation_thread.start()

            try:
                # Générer la réponse
                start_time = time.time()
                response = pipe(prompt, return_full_text=False)[0]["generated_text"]
                gen_time = time.time() - start_time

                # Vérifier et corriger la réponse si nécessaire
                response = correct_response_if_needed(reasoning_system, user_input, response, category, approach)

                # Formater la réponse pour une meilleure lisibilité
                response = format_response_text(response)

                # Arrêter l'animation
                stop_animation.set()
                animation_thread.join()

                # Afficher la réponse avec mise en forme
                print(f"\n{Fore.GREEN}Ton pote:{Style.RESET_ALL}")

                # Simuler une réponse en temps réel (comme un vrai chat)
                # Si la réponse est longue, afficher plus rapidement
                delay = 0.0005 if len(response) > 500 else 0.001  # Délais très courts
                print_slowly(response)

                # Petit indicateur de temps de génération
                print(f"\n{Fore.BLUE}[Généré en {gen_time:.2f}s]{Style.RESET_ALL}\n")

                # Mettre à jour l'historique de conversation
                chat_history.append((user_input, response))
                reasoning_system.add_to_history(user_input, response)

            except Exception as e:
                # Arrêter l'animation en cas d'erreur
                stop_animation.set()
                animation_thread.join()
                print(f"\n{Fore.RED}Erreur: {str(e)}{Style.RESET_ALL}\n")

        except KeyboardInterrupt:
            print(
                f"\n\n{Fore.YELLOW}Session interrompue. Tapez 'quitter' pour sortir ou continuez votre message.{Style.RESET_ALL}\n")
            continue

    # Nettoyage à la sortie
    colorama.deinit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Session terminée par l'utilisateur.{Style.RESET_ALL}")
        colorama.deinit()
        sys.exit(0)
