# 🏛️ Table-Ronde

> **Orchestrateur multi-agents pour le débat technique et la génération de plans d'implémentation v2.0 / v3.0.**

`table-ronde` est un outil en ligne de commande (CLI) propulsé par **LangChain** et **Rich**. Il simule une table-ronde entre trois agents IA aux rôles et personnalités complémentaires afin d'analyser un sujet, un concept d'architecture ou un codebase existant et de produire un plan d'action structuré et sans concession.

---

## 🎭 Les Agents

| Emoji | Rôle | Nom | Description & Rôle dans le débat | Température |
| :---: | :--- | :--- | :--- | :---: |
| 😈 | **Agent 1** | **Le Sceptique** | L'Avocat du Diable. Recherche activement les failles, la dette technique, les risques de sécurité et la complexité inutile. | `0.6` |
| 🚀 | **Agent 2** | **L'Enthousiaste** | Le Visionnaire. Répond aux critiques, propose des solutions modernes, cherche le chemin le plus rapide pour délivrer de la valeur. | `0.8` |
| 🏛️ | **Agent 3** | **L'Architecte** | Le Modérateur. Oriente les échanges, tranche les débats de manière pragmatique et génère le **Plan d'Implémentation Final** en Markdown. | `0.3` |

---

## ⚡ Caractéristiques

- **Multi-fournisseurs LLM** : Support natif de **Google Gemini** (`gemini-2.5-flash`) et **GitHub Copilot / GitHub Models / OpenAI** (`gpt-4o`).
- **Streaming en temps réel** : Affichage fluide chunk par chunk des réflexions de chaque agent avec `rich.Live`.
- **Mode Interactif (Human-in-the-Loop)** : Possibilité d'intervenir et d'orienter l'Architecte avec une note personnelle avant la résolution finale (`--interactive`).
- **Analyse de codebase** : Scanner de projet intelligent intégrant le filtrage `.gitignore` via `pathspec`.
- **Débat en 3 phases** :
  1. **Phase 1 : Audit** (Ouverture de l'Architecte, audit incisif du Sceptique, vision de l'Enthousiaste).
  2. **Phase 2 : Choc des idées** (Réfutation et contre-propositions).
  3. **Phase 3 : Résolution** (Synthèse et génération du Plan d'Implémentation).
- **Interface Console Rich** : Rendu dynamique et coloré en temps réel dans le terminal.
- **Export des livrables** : Exportation du plan final au format Markdown et export optionnel du transcript complet des échanges.

---

## 🛠️ Installation

Le projet utilise [`uv`](https://github.com/astral-sh/uv) pour la gestion des dépendances et de l'environnement virtuel (Python ≥ 3.12).

```bash
# Cloner le dépôt
git clone https://github.com/art0-v1r0s/table-ronde.git
cd table-ronde

# Synchroniser l'environnement et installer les dépendances
uv sync
```

---

## 🔑 Configuration des clés d'API

Définissez la variable d'environnement correspondant au fournisseur que vous souhaitez utiliser :

### Pour Google Gemini (par défaut)
```bash
export GEMINI_API_KEY="votre_cle_api_gemini"
```

### Pour GitHub Copilot / GitHub Models / OpenAI
```bash
export GITHUB_TOKEN="votre_token_github"
# ou
export COPILOT_API_KEY="votre_cle_copilot"
# ou
export OPENAI_API_KEY="votre_cle_openai"
```

---

## 🚀 Utilisation

### 1. Analyse basée sur un prompt / sujet libre

```bash
uv run table-ronde "Concevoir une architecture microservices temps réel pour une plateforme d'enchères"
```

### 2. Audit et refactorisation d'un projet existant

```bash
uv run table-ronde --path /chemin/vers/mon-projet "Optimiser la sécurité et la scalabilité du projet"
```

### 3. Utilisation de GitHub Copilot / GPT-4o

```bash
uv run table-ronde --provider copilot --model gpt-4o "Évaluer la transition vers une architecture Serverless"
```

### 4. Mode interactif (Human-in-the-loop)

```bash
uv run table-ronde --interactive "Moderniser notre architecture de pipeline de données"
```
*Le débat s'interrompt avant la Phase 3 pour vous permettre d'insérer vos directives à l'Architecte.*

### 5. Options CLI complètes

```bash
uv run table-ronde [OPTIONS] [PROMPT]
```

| Option | Raccourci | Description | Valeur par défaut |
| :--- | :--- | :--- | :--- |
| `--path` | `-p` | Chemin vers un répertoire de projet existant à scanner | `None` |
| `--output` | `-o` | Fichier de sortie pour le plan d'implémentation final | `plan_v2.md` |
| `--provider` | `-pr` | Fournisseur LLM (`gemini`, `copilot`, `github`, `openai`) | `gemini` |
| `--model` | `-m` | Modèle spécifique à utiliser (ex: `gemini-2.5-flash`, `gpt-4o`) | Auto selon provider |
| `--export-transcript` | `-t` | Chemin du fichier pour exporter l'intégralité du débat | `None` |
| `--interactive` | `-i` | Pause avant la résolution pour injecter votre note | `False` |

---

## 🧪 Tests & Qualité

Exécuter la suite de tests unitaires avec `pytest` :

```bash
uv run pytest
```

Vérifier la qualité du code avec `ruff` :

```bash
uv run ruff check .
```

---

## 📄 Licence

Ce projet est sous licence MIT.
