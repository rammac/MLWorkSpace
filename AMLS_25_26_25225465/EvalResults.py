
from dataclasses import dataclass
from typing import Optional
import numpy as np
import plotille

@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: np.ndarray


# history is of type <keras.src.callbacks.history.History
def plot_history_loss(history, *, width=70, height=20, logger=None):
    """history: keras.callbacks.History from model.fit(...)"""
    hist = history.history
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])


    #test
    accuracy = hist.get("accuracy", [])
    val_accuracy = hist.get("val_accuracy", [])
    #test
    
    x = list(range(1, len(loss) + 1))
    ys = [loss] + ([val_loss] if val_loss else [])

    ya = [accuracy] + ([val_accuracy] if val_accuracy else [])
    ya_max = max(max(y) for y in ya)

    # y-limits
    y_min = min(min(y) for y in ys)
    y_max = max(max(y) for y in ys)

    y_max = max(y_max, ya_max)
    
    if y_min == y_max: 
        y_min -= 1e-6
        y_max += 1e-6

    fig = plotille.Figure()
    fig.width = width
    fig.height = height
    fig.x_label = "Epoch"
    fig.y_label = "Loss"
    fig.set_x_limits(min_=1, max_=len(x))
    fig.set_y_limits(min_=y_min, max_=y_max)

    fig.plot(x, loss, label="train loss")
    #test
    fig.plot(x, accuracy, label="accuracy")
    fig.plot(x, val_accuracy, label="val accuracy")
    #test
    if val_loss:
        fig.plot(x, val_loss, label="val loss")

    chart = fig.show(legend=True)
    if logger is not None:
        logger.info("\n%s", chart)
    else:
        print(chart)




