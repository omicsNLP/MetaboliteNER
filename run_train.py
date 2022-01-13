import tabolistem_model as tabm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PROG')

    parser.add_argument('-t', '--text_path', type=str)
    parser.add_argument('-a', '--annot_path', type=str)
    parser.add_argument('-o', '--output_name', type=str)

    args = parser.parse_args()

    tm = tabm.TaboListem()
    tm.train(args.text_path, args.annot_path,
             args.output_name)

# e.g. python run_app.py -t "TrainingSet.txt" -a "TrainingSetAnnot.tsv" -o "TaboListemModel"