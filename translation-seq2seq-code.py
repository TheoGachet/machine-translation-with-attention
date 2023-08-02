# Importer les bibliothèques nécessaires
import numpy as np

import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text

# Cette classe permet de vérifier les formes des tenseurs lors de la manipulation des données
class ShapeChecker():
  def __init__(self):
    # Conserver un cache de chaque nom d'axe vu
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    # Si TensorFlow n'est pas en mode eager, ne fait rien
    if not tf.executing_eagerly():
      return

    # Analyser la forme du tenseur et la comparer avec les noms donnés
    parsed = einops.parse_shape(tensor, names)

    for name, new_dim in parsed.items():
      old_dim = self.shapes.get(name, None)

      # Si la nouvelle dimension est 1 et qu'elle doit être diffusée, continue
      if (broadcast and new_dim == 1):
        continue

      # Si le nom de l'axe est nouveau, ajouter sa longueur au cache
      if old_dim is None:
        self.shapes[name] = new_dim
        continue

      # Si la nouvelle dimension ne correspond pas à l'ancienne, lever une erreur
      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")

# Télécharger le fichier contenant les données
import pathlib

path_to_zip = tf.keras.utils.get_file(
    'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
    extract=True)

path_to_file = '/root/fra.txt'

# Charger les données depuis le fichier spécifié
def load_data(path):
    # Lire le fichier texte
    text = pathlib.Path(path).read_text(encoding='utf-8')

    # Diviser le texte en lignes
    lines = text.splitlines()
    # Diviser chaque ligne en paires de phrases (anglais, français)
    pairs = [line.split('\t') for line in lines]

    # Séparer les phrases anglaises et françaises
    target_raw = [pair[0] for pair in pairs]
    context_raw = [pair[1] for pair in pairs]

    return target_raw, context_raw

# Charger les données et afficher la dernière phrase française et anglaise
target_raw, context_raw = load_data(str(path_to_file))
print(context_raw[-1])
print(target_raw[-1])

# Paramètres pour la préparation des données
BUFFER_SIZE = len(context_raw)  # Taille du buffer pour le mélange des données.
                               # Ici, elle est définie comme étant la longueur totale des données pour assurer un mélange complet.
BATCH_SIZE = 64                 # Nombre d'exemples à traiter en une fois lors de l'entraînement.

# Créer un tableau de booléens pour déterminer si un exemple doit être utilisé pour l'entraînement.
# Une valeur True signifie que l'exemple est destiné à l'entraînement, tandis qu'une valeur False
# signifie qu'il est destiné à la validation. 80 % des exemples sont destinés à l'entraînement.
is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

# Extraire les indices des exemples d'entraînement et de validation.
train_indices = np.where(is_train)[0]
val_indices = np.where(~is_train)[0]

# Créer des datasets pour l'entraînement et la validation en utilisant les indices précédemment déterminés.
train_raw = (
    tf.data.Dataset
    .from_tensor_slices((np.array(context_raw)[train_indices], np.array(target_raw)[train_indices]))  # Créer un dataset à partir des tableaux
    .shuffle(BUFFER_SIZE)  # Mélanger les exemples pour garantir la variabilité lors de l'entraînement
    .batch(BATCH_SIZE))    # Grouper les exemples en lots (batches) pour l'entraînement
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((np.array(context_raw)[val_indices], np.array(target_raw)[val_indices]))  # De même pour les données de validation
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

# Afficher quelques exemples de l'ensemble d'entraînement pour vérifier le format
for example_context_strings, example_target_strings in train_raw.take(1):  # Prendre un lot du dataset d'entraînement
  print(example_context_strings[:5])  # Afficher les 5 premiers contextes du lot
  print()
  print(example_target_strings[:5])  # Afficher les 5 premières cibles correspondantes
  break

# Exemple de texte à transformer
example_text = tf.constant('Êtes-vous un chercheur en Intelligence Artificielle ?')

# Afficher le texte initial
print(example_text.numpy())
# Normaliser le texte pour éliminer les variations dues aux caractères spéciaux ou à la casse, par exemple.
# Ici, 'NFKD' est un type de normalisation Unicode qui décompose les caractères en leurs formes compatibles.
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())

# Fonction pour transformer le texte : le mettre en minuscule et séparer la ponctuation
def tf_lower_and_split_punct(text):
  # Séparer les caractères accentués
  text = tf_text.normalize_utf8(text, 'NFKD')
  # Convertir le texte en minuscule
  text = tf.strings.lower(text)
  # Garder l'espace, les lettres de a à z, et certains signes de ponctuation
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Ajouter des espaces autour de la ponctuation
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Supprimer les espaces superflus
  text = tf.strings.strip(text)

  # Ajouter des tokens de début et de fin autour du texte
  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

# Afficher l'exemple transformé
print(example_text.numpy().decode())
print(tf_lower_and_split_punct(example_text).numpy().decode())

# On définie une taille maximale pour le vocabulaire.
max_vocab_size = 5000

# On crée un processeur de texte pour le contexte (anglais). Ce processeur est une couche
# de vectorisation de texte qui permet de convertir les textes en séquences de tokens.
context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct, # Fonction pour normaliser le texte
    max_tokens=max_vocab_size,            # Taille maximale du vocabulaire
    ragged=True)                          # Renvoie un tensor de forme variable

# Le processeur de texte est "adapté" aux données d'entraînement. C'est comme ajuster un
# tokenizer sur des données: il apprend le vocabulaire.
context_text_processor.adapt(train_raw.map(lambda context, target: context))

# Affiche les 10 premiers mots du vocabulaire pour vérifier ce qu'il a appris.
context_text_processor.get_vocabulary()[:10]

# De la même manière, on crée un processeur de texte pour la cible (français).
target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True)

# Adapter le processeur de texte aux cibles du dataset d'entraînement.
target_text_processor.adapt(train_raw.map(lambda context, target: target))
target_text_processor.get_vocabulary()[:10]

# Exemple de tokens générés par le processeur de texte. On prend quelques exemples
# de chaînes contextuelles et on les tokenise.
example_tokens = context_text_processor(example_context_strings)
example_tokens[:3, :]

# Convertit les tokens en mots en utilisant le vocabulaire. Cela nous permet de
# vérifier si la tokenisation fonctionne correctement.
context_vocab = np.array(context_text_processor.get_vocabulary())
tokens = context_vocab[example_tokens[0].numpy()]
' '.join(tokens)

# Affiche une représentation visuelle des IDs de tokens et de leur masque.
# Le masque indique où les tokens sont présents (valeur 1) et où ils ne le sont pas (valeur 0).
plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens.to_tensor())
plt.title('Token IDs')

plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens.to_tensor() != 0)
plt.title('Mask')

# Fonction pour traiter le texte avant de le fournir au modèle. Elle convertit
# le contexte et la cible en tokens, et crée également les entrées et les sorties
# pour la cible (en décalant d'un token).
def process_text(context, target):
  context = context_text_processor(context).to_tensor()
  target = target_text_processor(target)
  targ_in = target[:,:-1].to_tensor()
  targ_out = target[:,1:].to_tensor()
  return (context, targ_in), targ_out

# Applique cette fonction de traitement aux datasets d'entraînement et de validation.
train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

# Affiche un exemple de données traitées pour vérifier la structure des tokens.
for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
  print(ex_context_tok[0, :10].numpy())
  print()
  print(ex_tar_in[0, :10].numpy())
  print(ex_tar_out[0, :10].numpy())


# On définit le nombre d'unités pour les couches d'encodage et d'attention.
UNITS = 256

# L'encodeur est une couche personnalisée qui convertit une séquence de texte en une séquence de vecteurs.
class Encoder(tf.keras.layers.Layer):
  # Initialisation de l'encodeur
  def __init__(self, text_processor, units):
    # Initialisation de la superclasse
    super(Encoder, self).__init__()
    # Le traitement du texte est nécessaire pour tokeniser le texte
    self.text_processor = text_processor
    # La taille du vocabulaire détermine le nombre de mots différents qui peuvent être traités
    self.vocab_size = text_processor.vocabulary_size()
    # Le nombre d'unités dans les couches RNN et d'embedding
    self.units = units

    # La couche d'embedding convertit les tokens (mots) en vecteurs
    # Cela permet d'avoir une représentation dense du texte
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

    # La couche RNN (GRU ici) traite ces vecteurs séquentiellement
    # Bidirectionnel signifie que le RNN traite le texte dans les deux directions (avant et arrière)
    self.rnn = tf.keras.layers.Bidirectional(
        merge_mode='sum',
        layer=tf.keras.layers.GRU(units, return_sequences=True, recurrent_initializer='glorot_uniform'))

  # Cette méthode est appelée pour traiter une séquence d'entrée x
  def call(self, x):
    # Une instance pour vérifier la forme des tenseurs
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch s')

    # Conversion des tokens en vecteurs
    x = self.embedding(x)
    shape_checker(x, 'batch s units')

    # Traitement des vecteurs avec RNN
    x = self.rnn(x)
    shape_checker(x, 'batch s units')

    # Renvoyer la séquence traitée
    return x

  # Cette méthode convertit un texte brut en sa représentation encodée
  def convert_input(self, texts):
    texts = tf.convert_to_tensor(texts)
    if len(texts.shape) == 0:
      texts = tf.convert_to_tensor(texts)[tf.newaxis]
    context = self.text_processor(texts).to_tensor()
    context = self(context)
    return context

# Encoder la séquence d'entrée

# Instanciation de l'encodeur
encoder = Encoder(context_text_processor, UNITS)
ex_context = encoder(ex_context_tok)

print(f'Tokens de contexte, forme (batch, s): {ex_context_tok.shape}')
print(f'Sortie de l\'encodeur, forme (batch, s, units): {ex_context.shape}')

# Cette couche fournit un mécanisme d'attention pour focaliser sur certaines parties du contexte lors de la traduction
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    # MultiHeadAttention est une méthode d'attention qui traite l'information de plusieurs manières à la fois
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    shape_checker = ShapeChecker()

    # Obtention des poids d'attention et sortie
    attn_output, attn_scores = self.mha(query=x, value=context, return_attention_scores=True)
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    self.last_attention_weights = attn_scores

    # Combinaison des sorties
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

attention_layer = CrossAttention(UNITS)

# Assister aux tokens encodés pour vérifier le fonctionnement de la couche d'attention
embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(), output_dim=UNITS, mask_zero=True)
ex_tar_embed = embed(ex_tar_in)
result = attention_layer(ex_tar_embed, ex_context)

# Affichage des formes pour comprendre la transformation des données
print(f'Séquence de contexte, forme (batch, s, units): {ex_context.shape}')
print(f'Séquence cible, forme (batch, t, units): {ex_tar_embed.shape}')
print(f'Résultat de l\'attention, forme (batch, t, units): {result.shape}')
print(f'Poids de l\'attention, forme (batch, t, s):    {attention_layer.last_attention_weights.shape}')

# Calculer la somme des poids d'attention pour vérifier qu'ils somment bien à 1
attention_layer.last_attention_weights[0].numpy().sum(axis=-1)

# Visualisation des poids d'attention

# Récupération des poids d'attention à partir de la dernière couche d'attention
attention_weights = attention_layer.last_attention_weights
# Création d'un masque pour exclure les tokens qui sont des zéros (c'est-à-dire des paddings)
mask=(ex_context_tok != 0).numpy()

# Création d'une figure avec 2 subplots côte à côte
plt.subplot(1, 2, 1)
# Affichage des poids d'attention multipliés par le masque (pour supprimer l'affichage des paddings)
plt.pcolormesh(mask*attention_weights[:, 0, :])
plt.title('Poids de l\'attention')

plt.subplot(1, 2, 2)
# Affichage du masque lui-même pour visualiser les régions effectivement masquées
plt.pcolormesh(mask)
plt.title('Masque');

# Définition de la classe du décodeur
class Decoder(tf.keras.layers.Layer):
  # Définir une méthode de classe pour ajouter dynamiquement des méthodes à la classe
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  # Initialisation de la classe décodeur
  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()  # Initialiser la superclasse Layer
    # Text processor pour traiter les séquences cibles
    self.text_processor = text_processor
    # Taille du vocabulaire des séquences cibles
    self.vocab_size = text_processor.vocabulary_size()
    # Conversion de mots en identifiants uniques
    self.word_to_id = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
    # Conversion inverse d'identifiants uniques en mots
    self.id_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
    # Identifier le token de début et de fin
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')
    # Définir le nombre d'unités pour le RNN et la couche d'embedding
    self.units = units
    # Convertir les identifiants de tokens en vecteurs d'embedding
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
    # RNN pour traiter les séquences cibles
    self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    # Couche d'attention pour se concentrer sur le contexte pertinent
    self.attention = CrossAttention(units)
    # Couche de sortie pour prédire le prochain token
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)

# La méthode "call" pour traiter les séquences avec le décodeur
@Decoder.add_method
def call(self, context, x, state=None, return_state=False):
  shape_checker = ShapeChecker()  # Utilitaire pour vérifier les dimensions des tensors
  # Vérifier les dimensions du tensor d'entrée
  shape_checker(x, 'batch t')
  shape_checker(context, 'batch s units')
  # Convertir les identifiants de tokens en vecteurs d'embedding
  x = self.embedding(x)
  shape_checker(x, 'batch t units')
  # Traiter la séquence avec le RNN
  x, state = self.rnn(x, initial_state=state)
  shape_checker(x, 'batch t units')
  # Utiliser la couche d'attention pour se concentrer sur le contexte pertinent
  x = self.attention(x, context)
  # Sauvegarder les poids d'attention pour une éventuelle visualisation
  self.last_attention_weights = self.attention.last_attention_weights
  shape_checker(x, 'batch t units')
  shape_checker(self.last_attention_weights, 'batch t s')
  # Prédire le prochain token avec la couche de sortie
  logits = self.output_layer(x)
  shape_checker(logits, 'batch t target_vocab_size')
  # Renvoyer soit les logits avec l'état, soit juste les logits en fonction de "return_state"
  if return_state:
    return logits, state
  else:
    return logits

# Instancier le décodeur avec les paramètres appropriés
decoder = Decoder(target_text_processor, UNITS)

# Tester le décodeur avec un exemple de contexte et une séquence d'entrée
logits = decoder(ex_context, ex_tar_in)

# Afficher les formes des tensors pour s'assurer qu'ils sont corrects
print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')

# Définition des méthodes supplémentaires pour le décodeur

# Cette méthode initialise l'état du décodeur avant la traduction
@Decoder.add_method
def get_initial_state(self, context):
  # Obtenir la taille du batch à partir du contexte
  batch_size = tf.shape(context)[0]
  # Créer le token de départ pour chaque séquence du batch
  start_tokens = tf.fill([batch_size, 1], self.start_token)
  # Initialiser la variable "done" à faux pour toutes les séquences
  done = tf.zeros([batch_size, 1], dtype=tf.bool)
  # Convertir les tokens de départ en embeddings
  embedded = self.embedding(start_tokens)
  # Retourner le token de départ, la variable "done", et l'état initial du RNN
  return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

# Convertir les tokens en texte
@Decoder.add_method
def tokens_to_text(self, tokens):
  # Convertir les identifiants de tokens en mots
  words = self.id_to_word(tokens)
  # Joindre les mots pour former une phrase
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  # Retirer les tokens de départ et de fin
  result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
  result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
  return result

# Prédire le prochain token
@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):
  # Obtenir les logits et l'état du décodeur
  logits, state = self(
    context, next_token,
    state = state,
    return_state=True)

  # Si la température est égale à 0, choisir le token avec la probabilité la plus élevée
  if temperature == 0.0:
    next_token = tf.argmax(logits, axis=-1)
  else:
    # Sinon, choisir un token de manière aléatoire en fonction des logits
    logits = logits[:, -1, :]/temperature
    next_token = tf.random.categorical(logits, num_samples=1)

  # Si un token de fin est généré, mettre à jour la variable "done"
  done = done | (next_token == self.end_token)
  # Si une séquence est terminée, produire uniquement du padding
  next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

  return next_token, done, state

# Initialiser les variables pour la boucle de traduction
next_token, done, state = decoder.get_initial_state(ex_context)
tokens = []

# Boucle pour générer une traduction
for n in range(10):
  # Obtenir le prochain token
  next_token, done, state = decoder.get_next_token(
      ex_context, next_token, done, state, temperature=1.0)
  # Ajouter le token à la liste des tokens
  tokens.append(next_token)

# Concaténer tous les tokens pour obtenir la traduction complète
tokens = tf.concat(tokens, axis=-1) # (batch, t)

# Convertir les tokens en texte
result = decoder.tokens_to_text(tokens)
result[:3].numpy()

# Classe Translator pour combiner l'encodeur et le décodeur
class Translator(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units,
               context_text_processor,
               target_text_processor):
    super().__init__()
    # Construire l'encodeur et le décodeur
    encoder = Encoder(context_text_processor, units)
    decoder = Decoder(target_text_processor, units)

    self.encoder = encoder
    self.decoder = decoder

  # Définition de la méthode "call" pour exécuter la traduction
  def call(self, inputs):
    context, x = inputs
    # Exécuter l'encodeur pour obtenir le contexte encodé
    context = self.encoder(context)
    # Exécuter le décodeur pour obtenir les logits
    logits = self.decoder(context, x)

    # Suppression du masque Keras (note spécifique au code, probablement liée à une contrainte technique)
    try:
      del logits._keras_mask
    except AttributeError:
      pass

    return logits

# Création d'une instance du modèle Translator et test sur un exemple
model = Translator(UNITS, context_text_processor, target_text_processor)

logits = model((ex_context_tok, ex_tar_in))

print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')

def masked_loss(y_true, y_pred):
    # Calcul de la perte pour chaque élément du batch
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # On retourne le résultat
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

# La fonction "masked_acc" est conçue pour calculer la précision en ignorant également les tokens de padding.
# Tout comme la perte masquée, la précision masquée s'assure que la métrique n'est calculée que sur
# les tokens pertinents et ignore les tokens de padding.

def masked_acc(y_true, y_pred):
    # Calcul de la perte pour chaque élément du batch
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

# Compiler le modèle avec un optimiseur, une fonction de perte et des métriques

# La méthode compile configure le processus d'apprentissage du modèle
model.compile(optimizer='adam', # Utilise l'optimiseur "adam" pour les mises à jour des poids
              loss=masked_loss, # Fonction de perte personnalisée pour évaluer les erreurs du modèle
              metrics=[masked_acc, masked_loss]) # Métriques pour surveiller la performance pendant l'entraînement

# Calculer la taille du vocabulaire cible

# Obtenir la taille du vocabulaire à partir du processeur de texte
vocab_size = 1.0 * target_text_processor.vocabulary_size()

# Calculer la perte et la précision attendues pour un modèle qui prédit les sorties au hasard
{"expected_loss": tf.math.log(vocab_size).numpy(),
 "expected_acc": 1/vocab_size}

# Évaluer la performance du modèle sur un jeu de données de validation

# Évalue la performance actuelle du modèle sur les données de validation
model.evaluate(val_ds, steps=20, return_dict=True)

# Entraîner le modèle

# Utiliser la méthode "fit" pour entraîner le modèle sur les données d'entraînement
history = model.fit(
    train_ds.repeat(), # Répète les données d'entraînement pour plusieurs passages (époques)
    epochs=100, # Nombre d'époques d'entraînement
    steps_per_epoch = 100, # Nombre de lots traités avant d'aller à l'époque suivante
    validation_data=val_ds, # Données utilisées pour la validation
    validation_steps = 20, # Nombre de lots de validation à utiliser à chaque époque
    callbacks=[ # Mécanismes pour intervenir pendant l'entraînement
        tf.keras.callbacks.EarlyStopping(patience=3)]) # Arrête l'entraînement si la perte ne s'améliore pas pendant 3 époques consécutives

# Afficher la courbe de la perte pendant l'entraînement
plt.plot(history.history['loss'], label='loss') # Courbe de la perte d'entraînement
plt.plot(history.history['val_loss'], label='val_loss') # Courbe de la perte de validation
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #') # Axe des abscisses
plt.ylabel('CE/token') # Axe des ordonnées
plt.legend()

# Afficher la courbe de la précision pendant l'entraînement
plt.plot(history.history['masked_acc'], label='accuracy') # Courbe de la précision d'entraînement
plt.plot(history.history['val_masked_acc'], label='val_accuracy') # Courbe de la précision de validation
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()

# Ajouter une méthode de traduction à la classe Translator

@Translator.add_method
def translate(self,
              texts, *,
              max_length=50, # Longueur maximale de la sortie
              temperature=0.0): # Paramètre pour contrôler la diversité de la sortie
  # Convertit le texte brut en représentation encodée
  context = self.encoder.convert_input(texts)
  batch_size = tf.shape(texts)[0] # Obtenir la taille du lot

  # Initialiser les variables pour la boucle de génération de tokens
  tokens = []
  attention_weights = []
  next_token, done, state = self.decoder.get_initial_state(context)

  for _ in range(max_length):
    # Générer le token suivant
    next_token, done, state = self.decoder.get_next_token(
        context, next_token, done, state, temperature)

    # Ajouter le token et les poids d'attention à leurs listes respectives
    tokens.append(next_token)
    attention_weights.append(self.decoder.last_attention_weights)

    # Arrêter la génération si tous les textes sont finis
    if tf.executing_eagerly() and tf.reduce_all(done):
      break

  # Concaténer les listes de tokens et de poids d'attention
  tokens = tf.concat(tokens, axis=-1)
  self.last_attention_weights = tf.concat(attention_weights, axis=1)

  # Convertir les tokens en texte
  result = self.decoder.tokens_to_text(tokens)
  return result

# Tester la méthode de traduction

# Traduire un exemple de phrase
result = model.translate(['J\'aime les pommes.'])
# Afficher la traduction
result[0].numpy().decode()

# Ajouter une méthode pour visualiser l'attention lors de la traduction

@Translator.add_method
def plot_attention(self, text, **kwargs):
  # Assurez-vous que le texte est une chaîne
  assert isinstance(text, str)
  # Obtenir la traduction du texte
  output = self.translate([text], **kwargs)
  output = output[0].numpy().decode()

  attention = self.last_attention_weights[0] # Récupérer les poids d'attention

  # Prétraitement des textes pour l'affichage
  context = tf_lower_and_split_punct(text)
  context = context.numpy().decode().split()
  output = tf_lower_and_split_punct(output)
  output = output.numpy().decode().split()[1:]

  # Créer un graphique pour visualiser les poids d'attention
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis', vmin=0.0)
  fontdict = {'fontsize': 14}
  ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + output, fontdict=fontdict)
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.set_xlabel('Texte en entrée')
  ax.set_ylabel('Texte en sortie')

# Tester la visualisation de l'attention
model.plot_attention('J\'aime les pommes rouges.')

# Récupérer un long texte pour le test
long_text = context_raw[-1]

# Afficher la traduction attendue
import textwrap
print('Sortie attendue:\n', '\n'.join(textwrap.wrap(target_raw[-1])))

# Tester la visualisation de l'attention sur le long texte
model.plot_attention(long_text)

# Préparer quelques entrées pour des tests supplémentaires
inputs = [
    'J\'aime les pommes rouges.',
    'Qui es tu ?',
    'Tu es très jolie aujourd\'hui.',
    'Quand viendras tu ?'
]

for t in inputs:
  print(model.translate([t])[0].numpy().decode())

print()

def plot_gradient_histogram(model):
    with tf.GradientTape() as tape:
        logits = model((ex_context_tok, ex_tar_in))
        loss_value = masked_loss(ex_tar_out, logits)

    grads = tape.gradient(loss_value, model.trainable_variables)
    gradients = [tf.norm(grad).numpy() for grad in grads if grad is not None]

    plt.hist(gradients, bins=50)
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.title('Histogram of Gradients')
    plt.show()

# Exemple d'utilisation après chaque époque ou itération :
plot_gradient_histogram(model)

def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Perte (entraînement)')
    plt.plot(epochs, val_loss, 'b', label='Perte (validation)')
    plt.title('Perte (entraînement et validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()

plot_loss(history)

import time

def plot_loss_vs_time(history):
    loss = history.history['loss']
    timestamps = [time.time() - start_time for start_time in history.epoch]
    plt.plot(timestamps, loss, 'bo-')
    plt.title('Perte en fonction du temps')
    plt.xlabel('Temps (secondes)')
    plt.ylabel('Perte')
    plt.show()

plot_loss_vs_time(history)

def plot_loss_distribution(history):
    loss = history.history['loss']
    plt.hist(loss, bins=10)
    plt.title('Distribution de la perte')
    plt.xlabel('Perte')
    plt.ylabel('Fréquence')
    plt.show()

plot_loss_distribution(history)

import pickle

# Enregistrer le modèle
model.save('nom_du_modele')

# Enregistrer l'historique
with open('historique.pkl', 'wb') as f:
    pickle.dump(history.history, f)