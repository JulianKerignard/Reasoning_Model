import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from datasets import Dataset
import csv

# Configuration pour éviter les warnings CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Utilisez la première GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration du modèle et du token
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = "ADD YOUR HF TOKEN HERE"
OUTPUT_DIR = "./mistral-finetuned" 

# Paramètres de l'entraînement - ajustez selon votre mémoire GPU
MICRO_BATCH_SIZE = 1  # Taille de batch par GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulation des gradients
LEARNING_RATE = 2e-4  # Taux d'apprentissage
EPOCHS = 3  # Nombre d'époques
MAX_SEQ_LENGTH = 512  # Longueur maximale des séquences
LORA_R = 8  # Rang de l'adaptation LoRA
LORA_ALPHA = 16  # Scaling alpha pour LoRA
LORA_DROPOUT = 0.05  # Dropout pour LoRA
SAVE_STEPS = 100  # Fréquence de sauvegarde


def prepare_model_and_tokenizer():
    """Prépare le modèle et le tokenizer avec QLoRA pour le fine-tuning."""
    print(f"Préparation du modèle {MODEL_NAME} pour fine-tuning avec QLoRA...")

    # Configuration pour la quantification 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Chargement du modèle quantifié
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Préparer le modèle pour la formation QLoRA
    model = prepare_model_for_kbit_training(model)

    # Configuration LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Appliquer LoRA au modèle
    model = get_peft_model(model, lora_config)

    # Chargement du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_instruction(row):
    """Formate une ligne de données au format d'instruction pour Mistral."""
    # Format Mistral: <s>[INST] instruction [/INST] answer </s>
    return f"<s>[INST] {row['instruction']} [/INST] {row['response']}</s>"


def prepare_dataset(csv_path):
    """Prépare le dataset à partir d'un fichier CSV de manière robuste."""
    print(f"Préparation des données depuis {csv_path}...")

    # Méthode robuste pour lire le CSV
    try:
        # Première tentative avec pandas
        df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Fichier CSV lu avec succès. {len(df)} exemples chargés.")
    except Exception as e:
        print(f"Erreur avec pandas: {e}")
        print("Tentative de lecture avec le module csv...")

        # Méthode alternative avec le module csv
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Lire l'en-tête
            header = next(csv.reader([f.readline()]))

            # Lire le reste ligne par ligne
            reader = csv.reader(f, quoting=csv.QUOTE_ALL)
            for row in reader:
                if len(row) >= 2:
                    data.append({
                        'instruction': row[0],
                        'response': row[1]
                    })
                else:
                    print(f"Attention: ligne ignorée car format incorrect: {row}")

        # Convertir en DataFrame
        df = pd.DataFrame(data)
        print(f"Lecture alternative réussie. {len(df)} exemples chargés.")

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['instruction', 'response']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' est requise dans le CSV")

    # Afficher quelques statistiques
    print(f"Nombre d'exemples chargés: {len(df)}")
    print(f"Longueur moyenne des instructions: {df['instruction'].str.len().mean():.1f} caractères")
    print(f"Longueur moyenne des réponses: {df['response'].str.len().mean():.1f} caractères")

    # Vérifier pour les valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Attention: valeurs manquantes détectées: {missing}")
        # Supprimer les lignes avec des valeurs manquantes
        df = df.dropna()
        print(f"Après suppression des valeurs manquantes: {len(df)} exemples")

    # Créer la colonne text avec le format d'instruction
    df['text'] = df.apply(format_instruction, axis=1)

    # Convertir en dataset Hugging Face
    dataset = Dataset.from_pandas(df[['text']])

    return dataset


def tokenize_dataset(dataset, tokenizer):
    """Tokenize le dataset."""
    print("Tokenisation des données...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None
        )

    # Mapper la fonction de tokenisation sur tout le dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Afficher des statistiques sur la tokenisation
    token_lengths = [len(x['input_ids']) for x in tokenized_dataset]
    avg_tokens = sum(token_lengths) / len(token_lengths)
    max_tokens = max(token_lengths)

    print(f"Longueur moyenne après tokenisation: {avg_tokens:.1f} tokens")
    print(f"Longueur maximale après tokenisation: {max_tokens} tokens")

    # Vérifier si certains exemples sont tronqués
    if max_tokens >= MAX_SEQ_LENGTH:
        print(f"Attention: {sum(l >= MAX_SEQ_LENGTH for l in token_lengths)} exemples sont tronqués")

    return tokenized_dataset


def train(model, tokenizer, dataset):
    """Entraîne le modèle sur le dataset."""
    print("Début de l'entraînement...")

    # Collator pour le language modeling causal
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        fp16=True,
        logging_steps=10,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        save_safetensors=True
    )

    # Initialiser le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Entraîner le modèle
    trainer.train()

    # Sauvegarder le modèle fine-tuné
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Modèle fine-tuné sauvegardé dans {OUTPUT_DIR}")
    return model, tokenizer


def fix_csv_file(input_path, output_path="dataset_amical_fixed.csv"):
    """Répare un fichier CSV problématique avec des virgules non échappées."""
    print(f"Tentative de réparation du fichier CSV {input_path}...")

    try:
        # Lire le contenu du fichier
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Essayer de détecter s'il s'agit d'un CSV standard ou d'un problème d'échappement
        lines = content.strip().split('\n')
        header = lines[0]

        # Si le header contient juste "instruction,response", c'est probablement un problème d'échappement
        if header.strip() == "instruction,response":
            print("Format CSV détecté, tentative de correction des lignes...")

            # Écrire un nouveau fichier avec échappement correct
            with open(output_path, 'w', encoding='utf-8', newline='') as out_file:
                writer = csv.writer(out_file, quoting=csv.QUOTE_ALL)

                # Écrire l'en-tête
                writer.writerow(["instruction", "response"])

                # Traiter chaque ligne après l'en-tête
                for i, line in enumerate(lines[1:], 1):
                    try:
                        # Diviser la ligne en conservant les citations
                        row = next(csv.reader([line]))

                        # Si on a exactement deux colonnes, c'est parfait
                        if len(row) == 2:
                            writer.writerow(row)
                        # Si on a plus de deux colonnes, c'est un problème de virgules
                        elif len(row) > 2:
                            # Supposons que la première colonne est l'instruction
                            # et tout le reste appartient à la réponse
                            instruction = row[0]
                            response = ','.join(row[1:])
                            writer.writerow([instruction, response])
                        else:
                            print(f"Ligne {i + 1} ignorée: pas assez de colonnes")
                    except Exception as e:
                        print(f"Erreur lors du traitement de la ligne {i + 1}: {e}")

            print(f"Fichier corrigé créé à {output_path}")
            return output_path
        else:
            print("Le format du fichier ne semble pas être un CSV standard")
            return input_path

    except Exception as e:
        print(f"Erreur lors de la réparation du fichier: {e}")
        return input_path


def main():
    # Vérification GPU
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("ATTENTION: Pas de GPU détecté, l'entraînement sera très lent.")

    # Chemin vers votre fichier de données
    csv_path = input("Entrez le chemin vers votre fichier CSV de données: ")

    # Tenter de corriger le fichier CSV si nécessaire
    try:
        fixed_csv_path = fix_csv_file(csv_path)
        if fixed_csv_path != csv_path:
            print(f"Utilisation du fichier corrigé: {fixed_csv_path}")
            csv_path = fixed_csv_path
    except Exception as e:
        print(f"Erreur lors de la tentative de correction du CSV: {e}")
        print("Tentative de poursuite avec le fichier original...")

    # Préparation du modèle et tokenizer
    model, tokenizer = prepare_model_and_tokenizer()

    # Préparation et tokenisation du dataset
    try:
        dataset = prepare_dataset(csv_path)
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)

        # Entraînement
        train(model, tokenizer, tokenized_dataset)

        print("Fine-tuning terminé avec succès!")

    except Exception as e:
        print(f"Erreur pendant le traitement: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
