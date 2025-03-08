from utils import *
from data_loader import *
from tensorflow.keras.callbacks import CSVLogger
import argparse
import tensorflow as tf

def get_model(model_name):
    if model_name.lower() == "unet":
        from unet import build_model
    elif model_name.lower() == "segnet":
        from segnet import build_model
    elif model_name.lower() == "linknet":
        from linknet import build_model
    else:
        raise ValueError("Invalid model name. Choose from 'unet', 'segnet', or 'linknet'.")
    return build_model()

def main(model_name, lr=1e-4, epochs=250, batch=32):
    (train_x, train_y), (valid_x, valid_y) = load_data("/workspace/NishuPandey/WCEBleedGen/WCEBleedGen/", validation_size=0.2)
    train = tf_dataset(train_x, train_y)
    val = tf_dataset(valid_x, valid_y)
    model = get_model(model_name)
    
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou, dice_coefficient]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
    
    csv_logger = CSVLogger("model_history_log.csv", append=True)
    
    train_steps = len(train_x) // batch
    valid_steps = len(valid_x) // batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(train,
        validation_data=val,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=[csv_logger]
    )
    model.save_weights("unet_test.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: 'unet', 'segnet', or 'linknet'")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    main(model_name=args.model, lr=args.lr, epochs=args.epochs, batch=args.batch)
