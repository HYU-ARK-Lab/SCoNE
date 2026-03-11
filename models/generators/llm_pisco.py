import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.generators.generator import Generator


class LLMPISCO(Generator):
    def __init__(
        self,
        model_name: str,  # in practice this is checkpoint path
        compressor_max_length: int = 128,
        decoder_max_length: int = 1280,  # maximum number of supported tokens in prompts
        batch_size: int = 32,
        prompt: str = None,
        max_new_tokens: int = 128,
        query_dependent: bool = False,
        **kwargs,
    ):
        """
        Class to use cocom with compression
        max_new_tokens: maximum number of tokens for generation
        model_max_length: maximum length used in the final query (should be large enough)
        """
        # Lazy import to prevent dependency
        # Should point to a compatible branch of cocom repo (preferably oscar_pisco release)
        from pisco.model import PISCO
        from pisco.collator import FineTuningCollator

        Generator.__init__(
            self,
            model_name=model_name,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_length=2048,
        )

        self.query_dependent = query_dependent

        self.pisco = PISCO.from_pretrained(model_name).cuda()
        self.pisco.prepare()

        self.pisco.config.compressor_max_length = compressor_max_length
        self.pisco.config.decoder_max_length = decoder_max_length

        self.collator = FineTuningCollator(
            self.pisco.compressor_tokenizer,
            self.pisco.decoder_tokenizer,
            self.pisco.compr_rate,
            query_dependent=query_dependent,
            compressor_max_length=compressor_max_length,
            decoder_max_length=decoder_max_length,
        )

        self.pisco.eval()

        self.prompt = prompt

        self.max_new_tokens = max_new_tokens

    def generate(self, instr_tokenized):
        """
        Nothing to do here, just convey to cocom since instr_tokenized went throught the collate_fn
        """
        device = next(self.pisco.parameters()).device
        instr_tokenized = {
            k: v.to(device)
            for k, v in instr_tokenized.items()
            if isinstance(v, torch.Tensor)
        }
        return self.pisco.generate(instr_tokenized, max_new_tokens=self.max_new_tokens)

    def eval(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda l: self.collate_fn(l, eval=True),
            num_workers=4,
        )

        responses, instructions, query_ids, queries, labels, ranking_labels = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        with torch.no_grad():
            for data_dict in tqdm(dataloader, desc="Generating"):
                id_ = data_dict["q_id"]
                instruction = data_dict["instruction"]
                query_ids += id_
                label = data_dict["label"]
                labels += label
                queries += data_dict["query"]
                ranking_labels += data_dict["ranking_label"]
                instructions += instruction
                generated_response = self.generate(data_dict["model_input"])
                responses += generated_response

        return query_ids, queries, instructions, responses, labels, ranking_labels

    def collate_fn(self, examples, eval=False):
        """
        Collates a batch of examples.

        Args:
            examples (list): batch from dataset
            eval (bool): Whether the function is being called for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Collated batch of data.
        """
        assert eval
        # example contains 'doc' (list), 'query', 'q_id', 'd_idx', 'label'
        [elt["doc"] for elt in examples]
        [elt["query"] for elt in examples]
        batch = self.collator.torch_call(
            [
                {
                    "docs": elt["doc"],
                    "query": elt["query"],
                    "mistral_label": "",  # so that we generate from there :)
                }
                for elt in examples
            ]
        )

        return {
            "model_input": batch,
            "q_id": [elt["q_id"] for elt in examples],
            "query": [elt["query"] for elt in examples],
            # "instruction": instr,
            "label": [elt["label"] for elt in examples],
            "instruction": self.collator.decoder_tokenizer.batch_decode(
                batch["decoder_input_ids"]
            ),
            "ranking_label": [[] for _ in examples],
        }

    def prsediction_step(self, model, model_input, label_ids=None):
        raise NotImplementedError
