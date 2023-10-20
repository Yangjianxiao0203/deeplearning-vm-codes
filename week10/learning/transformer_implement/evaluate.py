from loader import load_data, load_vocab
from collections import defaultdict, OrderedDict

from transformer.Translator import Translator


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.vocab = load_vocab(config["vocab_path"])
        self.reverse_vocab = dict([(y, x) for x, y in self.vocab.items()])

        # valid set
        self.valid_data = load_data(config["valid_data_path"], config, logger, shuffle=False)
        self.translator = Translator(model=self.model, beam_size=config["beam_size"],
                                     max_seq_len=config["output_max_length"], src_pad_idx=self.vocab["[PAD]"],
                                     trg_pad_idx=self.vocab["[PAD]"],
                                     trg_bos_idx=self.vocab["[CLS]"], trg_eos_idx=self.vocab["[SEP]"]
                                     )
    def eval(self,epoch):
        self.logger.info("epoch: %s, evaluation begins：" % epoch)
        self.model.eval()
        self.model.cpu()

        # just like generate model
        for index,batch_data in enumerate(self.valid_data):
            inputs, targets, golds = batch_data
            for input in inputs:
                output = self.translator.translate_sentence(input.unsqueeze(0))
                print("encode input: ", self.decode_seq(input))
                print("translate from transformer output: ", self.decode_seq(output))
                break

    def decode_seq(self, seq):
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])

# if __name__ == "__main__":
#     label = [2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
#
#     print([(i, l) for i, l in enumerate(label)])
#     print(Evaluator.get_chucks(label))