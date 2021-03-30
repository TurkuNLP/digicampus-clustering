from tnparser.pipeline import read_pipelines, Pipeline
import json
import tqdm
import argparse
import sys

ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# GPU
import types
extra_args=types.SimpleNamespace()
extra_args.__dict__["udify_mod.device"]="0" #simulates someone giving a --device 0 parameter to Udify
extra_args.__dict__["lemmatizer_mod.device"]="0"

available_pipelines=read_pipelines("models_fi_tdt_v2.7/pipelines.yaml")               # {pipeline_name -> its steps}
p=Pipeline(available_pipelines["parse_plaintext"], extra_args)                                      # launch the pipeline from the steps



def read_conllu(txt):
    sent=[]
    comment=[]
    for line in txt.split("\n"):
        line=line.strip()
        if not line: # new sentence
            if sent:
                yield comment,sent
            comment=[]
            sent=[]
        elif line.startswith("#"):
            comment.append(line)
        else: #normal line
            sent.append(line.split("\t"))
    else:
        if sent:
            yield comment, sent


def parse(txt):

    txt_parsed = p.parse(txt)
    conllu = [(comm, sent) for comm, sent in read_conllu(txt_parsed)]
    return conllu

def parse_all(data):
    examples = []
    for example in tqdm.tqdm(data):
        parsed = parse(example["essay"])
        example["essay_conllu"] = parsed
        examples.append(example)
    return examples



def main(args):

    with open(args.json, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    examples = parse_all(data)
    
    print(json.dumps(examples, indent=4, ensure_ascii=False, sort_keys=True))
        
        
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Essay json file")
    args=parser.parse_args()
    
    main(args)
