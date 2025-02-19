# xai-long-text-classification

[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()


SIDU method for long-text classification using transformer-based models


## **Installation**

To install the project, simply clone the repository and get the necessary dependencies. Then, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted.
```sh
# clone repo
git clone https://github.com/MarcoParola/xai-long-text-classification.git
cd xai-long-text-classification
mkdir models data

# Create virtual environment and install dependencies 
python -m venv env
. env/bin/activate
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt 

# Weights&Biases login 
wandb login 
```

Edit your wand user in the [config file](./config/config.yaml#L34)

# Per Giovanni

Carissimo! Ho implementato una prima bozza di workflow per gli esperimenti, in particolare la parte di training.
Che per ora ha relativamente senso perchè è già fatta e continuerei ad usare BERT finetunato su movie review

Per ora c'è solo la parte di training che va migliorata e affinata (vedi istruzioni). 
Ti ho creato un branch develop tutto per te, usa pure quello e quando ci sono delle versioni stabili del progetto facciamo la merge su main, ok?

Dai pure un'occhiata, però domani facciamo una chiamata e ti spiego meglio cosa fare/come usare la repo. Nel dubbio rileggi più volte quello che trovi qui, dove sono spiegate un po' di cose.

Ci sono una serie di TODO che sono abbastanza intuitivi. Altri invece meno. Fissiamo una chiamata nei prossimi giorni, cerco di farti sapere quando.