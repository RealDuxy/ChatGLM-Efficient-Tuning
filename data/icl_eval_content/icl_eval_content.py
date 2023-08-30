import json
from random import shuffle

import datasets
from typing import Any, Dict, List


_DESCRIPTION = "An example of dataset for ChatGLM."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL = "fc-bench-content.json"

prompt_template = """
你是平安集团的保险专家，请你根据下面的参考信息回复用户或者保险代理人提出的问题。
参考信息里也可能有部分文档的信息与问题无关，请忽略这些无关信息，也不要编造其他你不确定的信息。

【参考信息】
{context}

【用户问题】
{input}
"""

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
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))["instance"]
        for key, example in enumerate(example_dataset):
            correct_context = example["reference"]
            correct_context_body = "\n【文档】\n" + correct_context.replace("\n\n", "\n") + "\n"

            sim_ctx1 = example["similar_context"][0]["title"] + "\n\n" + example["similar_context"][0]["content"]
            sim_ctx2 = example["similar_context"][1]["title"] + "\n\n" + example["similar_context"][1]["content"]
            similar_context_bodies = ["\n【文档】\n" + (doc.replace("\n\n", "\n") + "\n") for doc in [sim_ctx1, sim_ctx2]]

            # doc_list = [ref, sim_ctx1, sim_ctx2]
            # shuffle(doc_list)
            doc_str = "\n\n".join([correct_context_body] + similar_context_bodies)

            query = example["input"]
            output = example["output"]
            prompt = prompt_template.replace("{context}", doc_str).replace("{input}", query).strip()
            yield key, {"instruction": prompt, "input": "", "output": output, "history": []}
