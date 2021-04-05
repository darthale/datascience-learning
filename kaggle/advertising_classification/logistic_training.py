import argparse
import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

S3 = "s3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()
    train_files = args.train
    model_dir = args.model_dir
    output_data_dir = args.output_data_dir

    train_data = None

    if train_files.startswith(S3):
        # files are stored in an S3 bukcet and we read it from there
        train_data = pd.read_csv(train_files, header=None)
    else:
        # case where files are in os.environ["SM_CHANNEL_TRAIN"]
        input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
                ).format(args.train, "train")
            )
        raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
        train_data = pd.concat(raw_data)

    # labels are in the first column
    y_train = train_data.iloc[:, 0]
    X_train = train_data.iloc[:, 1:]

    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train.values.ravel())

    # Store the trained model using the model dir specified in input args
    joblib.dump(logistic_regression_model, os.path.join(model_dir, "logreg.mdl"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "logreg.mdl"))
    return clf
