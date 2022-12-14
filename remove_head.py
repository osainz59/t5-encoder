from argparse import ArgumentParser
from transformers import T5EncoderModel, AutoTokenizer

parser = ArgumentParser('Head remover')

parser.add_argument("-i", '--input_model', dest='input_model', type=str, help="Input model.")
parser.add_argument("-o", '--output_model', dest='output_model', type=str, help="Output model.")

if __name__ == "__main__":
    args = parser.parse_args()
    # Save model
    model = T5EncoderModel.from_pretrained(args.input_model)
    model.save_pretrained(args.output_model)
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.input_model)
    tokenizer.save_pretrained(args.output_model)