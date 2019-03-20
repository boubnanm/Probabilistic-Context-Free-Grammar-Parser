
## Probabilistic Parser for French

This system provides a Probabilistic Parser for French based on the CYK algorithm, and the PCFG model and that is robust to unknown words thanks an OOV (out-of-vocabulary) module.

### System arguments
Input corpus parameters : 
* `--corpus_path`: Path to the corpus treebank to take as input
    * By default : "sequoia-corpus+fct.mrg_strict" (SEQUOIA treebank)
* `--train_size`: Size of the training corpus to extract
    * By default : 0.9 (90%)
* `--valid_size`: Size of the validation corpus to extract
    * By default : 0 (0%)
* `--test_size`: Size of the testing corpus to extract
    * By default : 0.1 (10%)
* `--shuffle_split`: Shuffle the corpus before splitting the three corpus.
    * By default : False

OOV parameters :
* `--embed_path`: Path to word embeddings pickle file
    * By default : 'polyglot-fr.pkl' (Polyglot French embeddings)
* `--lev_candidates`: Number of formal similar words candidates for the OOV module proposed.
    * By default : 2
* `--emb_candidates`: Number of embedding similar words candidates for the OOV module proposed.
    * By default : 20
* `--use_damereau`: Use Damereau-Levenstein distance for the formal similarities
    * By default : False
* `--linear_interpolation`: Linear interpolation coefficient for the bigram language model
    * By default : 0.2

CYK parameters :
* `--written_sentence`: Parse a sentence provided through the argument `--sent_to_parse`. If not, parse a file of sentences provided through the argument `--file_to_parse`
    * By default : False
* `--sent_to_parse`: Written sentence to parse
    * By default : "Le maire de Paris ."
* `--file_to_parse`: Path to the file containing sentences to parse (one sentence per line, exactly one whitespace between each token)
    * By default : "test_sentences.txt" (Last 10% of the SEQUOIA treebank)
* `--output_path`: Output path for the sentences parses outputs.
    * By default : "parse_results.txt"

Multiprocessing parameters:
* `--multiprocess`: Use multiprocessing (This may not work on windows)
    * By default : False
* `--n_cpus`: Number of CPUs to use. If -1, use all the CPUs available.
    * By default : -1


### How to use
Here are some examples to use the system proposed :

1. To train the parser on the first 90% of the SEQUOIA treebank and parse the last 10% **without** multiprocessing and output results to "parse_results.txt":
```bash
bash run.sh
```

2. To train the parser on the first 90% of the SEQUOIA treebank and parse the last 10% **with** multiprocessing and output results to "parse_results.txt":
```bash
bash run.sh --multiprocess
```

3. To train the parser on all the SEQUOIA treebank and parse the sentences in the file "sentences_to_parse.txt" and write the result to the file "parsed_sentences.txt":
```bash
bash run.sh --train_size 1 --test_size 0 --file_to_parse "sentences_to_parse.txt" --output_path "parsed_sentences.txt"
```

4. To train the parser on all the SEQUOIA treebank and parse a written sentence "Le maire de Paris .":
```bash
bash run.sh --train_size 1 --test_size 0 --written_sentence --sent_to_parse "Le maire de Paris ."
```
