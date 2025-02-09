import json
from random import shuffle

import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An example of dataset for ChatGLM."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "典型问题_人群购买.json"


class ExampleDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "input": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Dict[int, Dict[str, Any]]:
        with open(filepath, "r") as f:
            for key, data in enumerate(f.readlines()):
                data = json.loads(data)
                yield key, {"instruction": data["input"], "input": "", "output": data["output"], "history": []}
