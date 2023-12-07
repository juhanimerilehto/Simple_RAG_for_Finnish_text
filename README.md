
# RAG Solution for Document Processing
## Version: 1.0 - 2023-12-04
### Creator: Juhani Merilehto - @juhanimerilehto

## Description
This Python script provides a simple solution for processing and analyzing documents. It utilizes the Finnish BERT model for text extraction and processing, integrates with OpenAI's GPT models for advanced queries, and employs vector similarity indexing (FAISS) for data handling. This is just a personal weekend project, so please be gentle with comments ^^

### Description in Finnish
Tämä Python-skripti tarjoaa ratkaisun asiakirjojen käsittelyyn ja analysointiin.
Se käyttää suomalaista BERT-mallia tekstin poimintaan ja käsittelyyn, integroituu OpenAI:n GPT-mallien kanssa edistyneitä kyselyitä varten ja käyttää vektorisamankaltaisuusindeksointia (FAISS) datan käsittelyyn. Tämä on vain oma viikonloppuprojektini, joten olethan ystävällinen kommenttien kanssa ^^

## Requirements
- Python 3.x
- Libraries: `os`, `re`, `dotenv`, `numpy`, `textract`, `faiss`, `transformers`
- OpenAI API key for GPT integration
- Finnish BERT model `TurkuNLP/bert-base-finnish-uncased-v1`


## Installation
1. Clone this repository to your local machine.
2. Install the required Python packages using `pip install -r requirements.txt`.

### Installation in Finnish
1. Kloonaa tämä repositorio koneellesi.
2. Asenna tarvittavat Python-paketit käyttäen `pip install -r requirements.txt`.

## Usage
1. Set your OpenAI API key in an `.env` file.
2. Place your PDF documents in the `./data/` directory.
3. Run the script using `python rag_solution.py`.
4. Follow the interactive prompts for document processing and queries.

The script prints out extracted_paragraphs.txt and processed_paragraphs.txt in order to compare the PDF-processing iterations.
Vector information is also printed out to see the formed vectorspace.

### Usage in Finnish
1. Aseta OpenAI API-avaimesi `.env`-tiedostoon.
2. Sijoita PDF-asiakirjasi `./data/` hakemistoon.
3. Suorita skripti käyttäen `python rag_solution.py`.
4. Noudata interaktiivisia kehotteita asiakirjojen käsittelyyn ja kyselyihin.

Skripti tulostaa extracted_paragraphs.txt ja processed_paragraphs.txt tiedostot, jotta voidaan verrata PDF-tiedostojen käsittelyn iteraatioita.
Myös vektoritiedot tulostetaan jotta nähdään tietoja vektorien muodostuksesta.

## GUI
A simple GUI can be found in the "GUI_mvp.py" file. It works simply:

1. Upload a PDF-file, preview of raw text is shown
2. Process data, for cleaning paragraphs into vectorbase
3. Query the data by writing a prompt
4. Repeat

### GUI
Yksinkertainen graafinen käyttöliittymä löytyy tiedostosta "GUI_mvp.py". Sen toiminta on yksinkertaista:

1. Lataa PDF-tiedosto, näytetään raakatekstin esikatselu
2. Käsittele data, puhdistaaksesi kappaleet vektoritietokantaan
3. Tee kyselyjä kirjoittamalla kehotteita
4. Toista


## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

### Contributing in Finnish
Osallistuminen on tervetullutta. Ole hyvä ja tee haara repositoriosta ja lähetä vetopyyntö muutoksillasi.

## License
[MIT License](LICENSE)

### License in Finnish
[MIT-lisenssi](LICENSE)
