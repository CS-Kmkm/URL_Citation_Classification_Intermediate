import torch
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput


class CLS_bert(torch.nn.Module):
    """
    text classification model
    output has N labels
    """

    def __init__(
        self, vocab_size: int, n_classes: list[int], model_name="bert-base-uncased"
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(vocab_size)

        self.n_classes = n_classes

        self.dropout = torch.nn.Dropout(0.1)
        self.hidden_dim = self.bert.config.hidden_size

        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_dim, n_class) for n_class in n_classes]
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label=None,
    ):
        X = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).last_hidden_state[:, 0, :]
        X = self.dropout(X)

        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(X))

        loss = None
        losses = []
        if label is not None:
            assert len(label) == len(self.n_classes)
            for i, logit in enumerate(logits):
                losses.append(torch.nn.functional.cross_entropy(logit, label[i]))
            loss = sum(losses)

        return ModelOutput(loss=loss, logits=logits)


class Bin_bert(torch.nn.Module):
    def __init__(
        self, vocab_size: int, n_label: int, model_name="bert-base-uncased"
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(vocab_size)

        self.n_label = n_label

        self.dropout = torch.nn.Dropout(0.1)
        self.hidden_dim = self.bert.config.hidden_size

        self.classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(self.hidden_dim, 1) for _ in range(self.n_label)]
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        label=None,
    ):
        X = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).last_hidden_state[:, 0, :]
        X = self.dropout(X)

        logits = []
        for classifier in self.classifiers:
            logits.append(classifier(X).squeeze(-1))

        loss = None
        losses = []
        if label is not None:
            assert len(label) == self.n_label
            for i, logit in enumerate(logits):
                losses.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logit, label[i].float()
                    )
                )
            loss = sum(losses)

        return ModelOutput(loss=loss, logits=logits)
