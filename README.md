# music-generation
## Introduction
### Objectif
L’objectif de ce projet de Génération de musique est de pouvoir générer des grilles musicales en format MIDI grâce à un VAE récurrent.

### Contrainte supplémentaire
Une contrainte personnelle supplémentaire : utiliser uniquement des ressources gratuites pour mener à bien le projet. Etant donnée l’indisponibilité du serveur avec GPU du Cnam, le choix s’est naturellement tourné vers l’utilisation de ressources cloud gratuites (Google colab+google Drive).

## Architecture globale du modèle
Ce projet implémente un générateur de musique basé sur un Variational Autoencoder (VAE), en deux parties.
### Architecture du modèle VAE basique

Le modèle MIDI_VAE est structuré comme suit :
* Encodeur : Deux couches LSTM pour traiter les séquences MIDI et des couches denses pour générer la moyenne (μ) et la log-variance (log σ2) de la distribution latente
* Espace latent : Reparamétrisation pour permettre la rétropropagation. Échantillonnage à partir de la distribution gaussienne définie par μ et σ2
* Décodeur : Une couche dense initiale qui transforme le vecteur latent.Une couche LSTM qui reconstruit la séquence temporelle. Une couche dense finale avec activation sigmoïde pour produire les probabilités de notes.
Fonction de perte :Perte de reconstruction BCE (Binary Cross Entropy) et divergence KL entre la distribution latente et une distribution normale standard
### Architecture du modèle Style conditioned VAE
Cette extension ajoute une couche d’embedding de style pour conditionner la génération musicale. Le style influence à la fois l'encodage vers l'espace latent et le décodage à partir de l'espace latent. Cette approche est inspirée de l'article de référence, où un classifier de style est utilisé pour forcer l'encodeur à apprendre une représentation compacte du style dans l'espace latent.
On prévoit une méthode pour générer des séquences dans un style spécifique et une méthode pour interpoler entre deux styles différents.

## Contenu du repository
* Rapport du projet (PDF)
* Notebook pour le modèle basique
* Notebook contenant l'extension
