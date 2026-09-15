# 🏛️ Table-Ronde

> **Orchestrateur multi-agents dynamique pour le débat technique et la génération de plans d'implémentation v2.0 / v3.0.**

`table-ronde` est un outil en ligne de commande (CLI) propulsé par **LangChain** et **Rich**. Il simule une table-ronde entre plusieurs agents IA aux rôles et personnalités complémentaires (par défaut : l'Architecte, le Sceptique et l'Enthousiaste) afin d'analyser un sujet, un concept d'architecture ou un codebase existant et de produire un plan d'action structuré.

Depuis sa version 2.0, **Table-Ronde est entièrement dynamique** : vous pouvez configurer vos propres experts, utiliser différents LLMs simultanément, et orchestrer des débats sur plusieurs rounds !

---

## ⚡ Caractéristiques (v2.0)

- **Configuration Dynamique (`config.yml`)** : Définissez vos propres personas (ex: Expert Sécurité, Développeur Junior), leurs emojis, leurs prompts et leurs températures individuelles.
- **Modèles hybrides** : Possibilité d'assigner un modèle / fournisseur LLM différent à chaque agent (ex: `gpt-4o` pour l'Architecte, `gemini-3.6-flash` pour le Sceptique).
- **Rounds multiples** : Lancez des débats approfondis sur plusieurs boucles (`--rounds N`).
- **Mode Interactif Avancé** : Intervenez avant la synthèse finale pour donner vos directives, ou tapez `/tour` pour obliger les agents à refaire un nouveau round de débat en tenant compte de votre remarque (`--interactive`).
- **Analyse de codebase** : Scanner de projet intelligent intégrant le filtrage `.gitignore` via `pathspec`.
- **Réseau ultra-optimisé** : Implémentation réseau (IPv4 forcée) pour des temps de connexion instantanés, évitant les blackholes IPv6 sur l'API Google.
- **Streaming en temps réel** : Affichage fluide chunk par chunk des réflexions de chaque agent avec `rich.Live`.
- **Export des livrables** : Exportation du plan final au format Markdown et export optionnel du transcript complet des échanges.

---

## 🎭 Les Agents par défaut

Si vous n'utilisez pas de configuration personnalisée, le système lance ces 3 agents par défaut :

| Emoji | Rôle | Nom | Description & Rôle dans le débat | Température |
| :---: | :--- | :--- | :--- | :---: |
| 😈 | **Agent 1** | **Le Sceptique** | L'Avocat du Diable. Recherche activement les failles, la dette technique, les risques de sécurité et la complexité inutile. | `0.6` |
| 🚀 | **Agent 2** | **L'Enthousiaste** | Le Visionnaire. Répond aux critiques, propose des solutions modernes, cherche le chemin le plus rapide pour délivrer de la valeur. | `0.8` |
| 🏛️ | **Agent 3** | **L'Architecte** | Le Modérateur. Oriente les échanges, tranche les débats de manière pragmatique et génère le **Plan d'Implémentation Final** en Markdown. | `0.3` |

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

### 1. Analyse simple (Prompts et Agents par défaut)

```bash
uv run table-ronde "Concevoir une architecture microservices temps réel pour une plateforme d'enchères"
```

### 2. Audit et refactorisation d'un projet existant

```bash
uv run table-ronde --path /chemin/vers/mon-projet "Optimiser la sécurité et la scalabilité du projet"
```

### 3. Utiliser votre propre configuration de Personas (NOUVEAU)

Créez un fichier `my_config.yml` pour définir vos experts sur mesure :
```yaml
orchestrator:
  rounds: 2
  default_provider: gemini
  default_model: gemini-3.6-flash

architect:
  role: architect
  title: "L'Architecte"
  emoji: "🏛️"
  temperature: 0.2
  prompt: "Tu es l'Architecte. Modère les experts et propose un plan ultra détaillé."
  # model: gpt-4o (vous pouvez surcharger le modèle par défaut ici)

personas:
  - role: security
    title: "L'Expert Sécurité"
    emoji: "🛡️"
    temperature: 0.3
    prompt: "Tu cherches systématiquement les failles OWASP dans l'idée proposée."
  - role: dev
    title: "Lead Developer"
    emoji: "💻"
    temperature: 0.6
    prompt: "Tu parles architecture logicielle, design patterns et scalabilité."
```
Puis lancez l'outil :
```bash
uv run table-ronde --config my_config.yml "Mettre en place un SSO OAuth2"
```

### 4. Mode interactif (Human-in-the-loop) et Multi-tours

```bash
uv run table-ronde --interactive --rounds 2 "Moderniser notre architecture de pipeline de données"
```
*Le débat s'interrompra à la fin de chaque round pour vous permettre d'insérer vos directives à l'Architecte. Si vous répondez avec la commande magique `/tour`, les agents referont un cycle complet de débat autour de votre remarque !*

---

## 🛠️ Options CLI complètes

```bash
uv run table-ronde [OPTIONS] [PROMPT]
```

| Option | Raccourci | Description | Valeur par défaut |
| :--- | :--- | :--- | :--- |
| `--config` | `-c` | Fichier YAML de configuration personnalisée des personas | `None` |
| `--rounds` | `-r` | Nombre de tours de débat (surcharge la config YAML) | `1` |
| `--interactive` | `-i` | Pause avant la résolution pour injecter votre note ou `/tour` | `False` |
| `--path` | `-p` | Chemin vers un répertoire de projet existant à scanner | `None` |
| `--output` | `-o` | Fichier de sortie pour le plan d'implémentation final | `plan_v2.md` |
| `--provider` | `-pr` | Fournisseur LLM (`gemini`, `copilot`, `github`, `openai`) | `gemini` |
| `--model` | `-m` | Modèle spécifique à utiliser (ex: `gemini-3.6-flash`, `gpt-4o`) | Auto selon provider |
| `--export-transcript` | `-t` | Chemin du fichier pour exporter l'intégralité du débat | `None` |

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
