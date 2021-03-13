from fairseq.data import Dictionary
import sentencepiece as spm
from os.path import join as pjoin
from transformers import PreTrainedTokenizer
import sentencepiece as spm


class XLMRobertaTokenizer(PreTrainedTokenizer):
    """Custom tokenizer for our custom pretrained model. 
    You can ignore this file if you use another pretrained model. For example, if you use PhoBert, you should tokenize by using VnCoreNLP.
    """
    def __init__(
        self,
        pretrained_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # load bpe model and vocab file
        sentencepiece_model = pjoin(pretrained_file, 'sentencepiece.bpe.model')
        vocab_file = pjoin(pretrained_file, 'dict.txt')
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sentencepiece_model)   # please dont use anything from sp_model bcz it makes everything goes wrong

        self.bpe_dict = Dictionary().load(vocab_file)

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 0

        self.fairseq_tokens_to_ids["<mask>"] = len(self.bpe_dict) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}
     
     
    def _tokenize(self, text): 
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.bpe_dict.index(token)
        return spm_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.bpe_dict[index]
        
    @property
    def vocab_size(self):
        return len(self.bpe_dict) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
