try:
    import torch
except ImportError:
    raise ImportError("Could not find torch >= 0.12, which is a required dependency for using the CRAFT model. "
                      "Run 'pip install convokit[neuralnetwork]' to fix this.")
import pandas as pd
from convokit.forecaster.CRAFT.craftUtil import loadPrecomputedVoc, batchIterator, CONSTANTS
from .CRAFT.craftNN import initialize_model
from .forecasterModel import ForecasterModel


class CRAFTModel(ForecasterModel):

    def __init__(self, device_type: str = 'cpu', batch_size: int = 64, max_length: int = 80,
                 forecast_feat_name: str = "prediction", forecast_prob_feat_name: str = "score"):
        """
        :param device_type: 'cpu' or 'cuda', default: 'cpu'
        :param batch_size: batch size for computation, default: 64
        :param max_length: no. of tokens to use, defau;t: 80
        :param forecast_feat_name: name of DataFrame column containing predictions, default: "prediction"
        :param forecast_prob_feat_name: name of DataFrame column containing prediction scores, default: "score"
        """

        super().__init__(forecast_feat_name=forecast_feat_name, forecast_prob_feat_name=forecast_prob_feat_name)
        assert device_type in ['cuda', 'cpu']
        self.device = torch.device(device_type)
        self.device_type = device_type
        self.voc = loadPrecomputedVoc("wikiconv", CONSTANTS['WORD2INDEX_URL'], CONSTANTS['INDEX2WORD_URL'])
        self.predictor = initialize_model(CONSTANTS['MODEL_URL'], self.voc, self.device, device_type)
        self.batch_size = batch_size
        self.max_length = max_length

    def evaluateBatch(self, predictor, input_batch, dialog_lengths,
                      dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, device, max_length):
        # Set device options
        input_batch = input_batch.to(device)
        dialog_lengths = dialog_lengths.to(device)
        utt_lengths = utt_lengths.to(device)
        # Predict future attack using predictor
        scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths,
                           batch_indices, dialog_indices, batch_size, max_length)
        predictions = (scores > 0.5).float()
        return predictions, scores

    def evaluateDataset(self, dataset, predictor, voc, batch_size, device):
        # create a batch iterator for the given data
        batch_iterator = batchIterator(voc, dataset, batch_size, shuffle=False)
        # find out how many iterations we will need to cover the whole dataset
        n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
        output_df = {
            "id": [],
            self.forecast_feat_name: [],
            self.forecast_prob_feat_name: []
        }
        for iteration in range(1, n_iters+1):
            # batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
            batch, batch_dialogs, true_batch_size = next(batch_iterator)
            # Extract fields from batch
            input_variable, dialog_lengths, utt_lengths, \
            batch_indices, dialog_indices, convo_ids, target_variable, mask, max_target_len = batch
            dialog_lengths_list = [len(x) for x in batch_dialogs]
            # run the model
            predictions, scores = self.evaluateBatch(predictor, input_variable,
                                                                dialog_lengths, dialog_lengths_list, utt_lengths,
                                                                batch_indices, dialog_indices,
                                                                true_batch_size, device, self.max_length)

            # format the output as a dataframe (which we can later re-join with the corpus)
            for i in range(true_batch_size):
                convo_id = convo_ids[i]
                pred = predictions[i].item()
                score = scores[i].item()
                output_df["id"].append(convo_id)
                output_df[self.forecast_feat_name].append(pred)
                output_df[self.forecast_prob_feat_name].append(score)

            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

        return pd.DataFrame(output_df).set_index("id")

    def train(self, id_to_context_reply_label):
        pass

    def forecast(self, id_to_context_reply_label):
        """
        Compute forecasts and forecast scores for the given dictionary of utterance id to (context, reply) pairs
        Return the values in a DataFrame
        :param id_to_context_reply_label: dict mapping utterance id to (context, reply, label)
        :return: a pandas DataFrame
        """
        dataset = [(context, reply, id_) for id_, (context, reply, label) in id_to_context_reply_label.items()]
        return self.evaluateDataset(dataset=dataset, predictor=self.predictor, voc=self.voc,
                                    batch_size=self.batch_size, device=self.device)
