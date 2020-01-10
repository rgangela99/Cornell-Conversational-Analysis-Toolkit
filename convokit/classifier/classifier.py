from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import svm
from convokit.classifier.util import *
from convokit import Transformer, CorpusObject


class Classifier(Transformer):
    def __init__(self, obj_type: str, pred_feats: List[str],
                 labeller: Callable[[CorpusObject], bool] = lambda x: True,
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "pred_score"):
        """

        :param obj_type: type of Corpus object to classify: 'conversation', 'user', or 'utterance'
        :param pred_feats: list of metadata keys containing the features to be used in prediction
        :param labeller: function to get the y label of the Corpus object, e.g. lambda utt: utt.meta['y']
        :param selector: function to select for Corpus objects to transform
        :param clf: optional classifier model, an SVM with linear kernel will be initialized by default
        :param clf_feat_name: metadata feature name to use in annotation for classification result, default: "prediction"
        :param clf_prob_feat_name: metadata feature name to use in annotation for classification probability, default: "score"
        """
        self.pred_feats = pred_feats
        self.labeller = labeller
        self.selector = selector
        self.obj_type = obj_type

        self.clf = svm.SVC(C=0.02, kernel='linear', probability=True) if clf is None else clf
        self.clf_feat_name = clf_feat_name
        self.clf_prob_feat_name = clf_prob_feat_name

    def fit(self, corpus: Corpus, y=None):
        """
        Trains the Transformer's classifier model
        :param corpus: target Corpus
        """
        X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Run classifier on given corpus's objects and annotate them with the predictions and prediction scores
        :param corpus: target Corpus
        :return: annotated Corpus
        """
        obj_id_to_feats = extract_feats_dict(corpus, self.obj_type, self.pred_feats, self.selector)
        feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index').reindex(index = list(obj_id_to_feats))
        X = csr_matrix(feats_df.values)
        idx_to_id = {idx: obj_id for idx, obj_id in enumerate(list(obj_id_to_feats))}
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            corpus_obj = corpus.get_object(self.obj_type, idx_to_id[idx])
            corpus_obj.add_meta(self.clf_feat_name, clf)
            corpus_obj.add_meta(self.clf_prob_feat_name, clf_prob)

        for obj in corpus.iter_objs(self.obj_type, self.selector):
            if self.clf_feat_name not in obj.meta:
                obj.meta[self.clf_feat_name] = None
                obj.meta[self.clf_prob_feat_name] = None

        return corpus

    def transform_objs(self, objs: List[CorpusObject]) -> List[CorpusObject]:
        """
        Run classifier on list of Corpus objects and annotate them with the predictions and prediction scores

        :param objs: list of Corpus objects
        :return: list of annotated Corpus objects
        """
        X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
        # obj_ids = [obj.id for obj in objs]
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)

        return objs

    def summarize(self, corpus: Corpus = None, objs: List[CorpusObject] = None, use_selector=True):
        """
        Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of classification results.
        Run either on a target Corpus or a list of Corpus objects
        :param corpus: target Corpus
        :param objs: list of Corpus objects
        :param use_selector: whether to use Classifier.selector for selecting Corpus objects
        :return: pandas DataFrame indexed by Corpus object id
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("summarize() takes in either a Corpus or a list of users / utterances / conversations")

        objId_clf_prob = []

        if objs is None:
            for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
                objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))
        else:
            for obj in objs:
                objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                            columns=['id', self.clf_feat_name, self.clf_prob_feat_name]).set_index('id').sort_values(self.clf_prob_feat_name)

    def evaluate_with_train_test_split(self, corpus: Corpus = None,
                 objs: List[CorpusObject] = None,
                 test_size: float = 0.2):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label, using a
        train-test split.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects
        :param corpus: target Corpus
        :param objs: target list of Corpus objects
        :param test_size: size of test set
        :return: accuracy and confusion matrix
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("This function takes in either a Corpus or a list of users / utterances / conversations")

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
            # obj_ids = [obj.id for obj in corpus.iter_objs(self.obj_type, self.selector)]
        elif objs:
            print("Using input list of corpus objects...")
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
            y = np.array([self.labeller(obj) for obj in objs])
            # obj_ids = [obj.id for obj in objs]

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print("Done.")
        return accuracy, confusion_matrix(y_true=y_test, y_pred=preds)

    def evaluate_with_cv(self, corpus: Corpus = None,
                         objs: List[CorpusObject] = None, cv=KFold(n_splits=5)):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label, using cross-validation
        for data splitting.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects

        :param corpus: target Corpus
        :param objs: target list of Corpus objects
        :param cv: cross-validation model to use: KFold(n_splits=5) by default.
        :return: cross-validated accuracy score
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("This function takes in either a Corpus or a list of users / utterances / conversations")

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
            # obj_ids = [obj.id for obj in corpus.iter_objs(self.obj_type, self.selector)]
        elif objs:
            print("Using input list of corpus objects...")
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
            y = np.array([self.labeller(obj) for obj in objs])
            # obj_ids = [obj.id for obj in objs]

        print("Running a cross-validated evaluation...")
        score = cross_val_score(self.clf, X, y, cv=cv)
        print("Done.")
        return score

    def confusion_matrix(self, corpus, use_selector=True):
        """
        Generate confusion matrix for transformed corpus using labeller for y_true and clf_feat_name as y_pred
        :param corpus: target Corpus
        :param use_selector: whether to use Classifier.selector for selecting Corpus objects
        :return: sklearn confusion matrix
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_feat_name])

        return confusion_matrix(y_true=y_true, y_pred=y_pred)

    def base_accuracy(self, corpus, use_selector=True):
        y_true, y_pred = self.get_y_true_pred(corpus, use_selector=use_selector)
        all_true_accuracy = np.array(y_true).mean()
        return max(all_true_accuracy, 1-all_true_accuracy)

    def accuracy(self, corpus, use_selector=True):
        y_true, y_pred = self.get_y_true_pred(corpus, use_selector=use_selector)
        return (np.array(y_true) == np.array(y_pred)).mean()

    def get_y_true_pred(self, corpus, use_selector=True):
        """
        Get lists of true and predicted labels
        :param corpus: target Corpus
        :param use_selector: whether to use Classifier.selector for selecting Corpus objects
        :return: list of true labels, and list of predicted labels
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_feat_name])

        return y_true, y_pred

    def classification_report(self, corpus, use_selector=True):
        """
        Generate classification report for transformed corpus using labeller for y_true and clf_feat_name as y_pred
        :param corpus: target Corpus
        :param use_selector: whether to use Classifier.selector for selecting Corpus objects
        :return: classification report
        """
        y_true = []
        y_pred = []
        for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
            y_true.append(self.labeller(obj))
            y_pred.append(obj.meta[self.clf_feat_name])

        return classification_report(y_true=y_true, y_pred=y_pred)

    def get_coefs(self, feature_names: List[str], coef_func=None):
        """
        Get dataframe of classifier coefficients
        :param feature_names: list of feature names to get coefficients for
        :param coef_func: function for accessing the list of coefficients from the classifier model; by default,
                            assumes it is a pipeline with a logistic regression component
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        return get_coefs_helper(self.clf, feature_names, coef_func)
