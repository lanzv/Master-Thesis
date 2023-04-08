from src.eval.pipelines import single_iteration_pipeline
from src.utils.plotters import plot_iteration_statistics

class IterationStatistics():
    
    def __init__(self, train_modes, dev_modes):
        """
        ToDo
        """
        self.train_statistics = {
            "accuracy": [],
            "f1": [],
            "mjww": [],
            "wtmf": [],
            "wufpc": [],
            "vocab_size": [],
            "avg_segment_len": [],
            "perplexity": [],
            "iterations": []
        }
        self.dev_statistics = {
            "accuracy": [],
            "f1": [],
            "mjww": [],
            "wtmf": [],
            "wufpc": [],
            "vocab_size": [],
            "avg_segment_len": [],
            "perplexity": [],
            "iterations": []
        }
        self.train_modes = train_modes
        self.dev_modes = dev_modes
        # Print head of statistics table
        print("Iteration (train / dev) |  bacor accuracy   |      bacor f1     |           perplexity          |   vocab size  | avg segment len |       mjww        |        wtmf       |              wufpc            |")

    def add_new_iteration(self, iteration: int, 
                          train_segments: list, dev_segments: list,
                          train_perplexity: float, dev_perplexity: float):
        """
        ToDo
        """
        
        # Get Train and Dev results
        train_accuracy, train_f1, train_mjww, train_wtmf, train_wufpc, train_vocab_size, train_avg_segment_len, \
        dev_accuracy, dev_f1, dev_mjww, dev_wtmf, dev_wufpc, dev_vocab_size, dev_avg_segment_len \
            = single_iteration_pipeline(train_segments, self.train_modes, dev_segments, self.dev_modes)
    
        
        # Store data
        self.train_statistics["accuracy"].append(train_accuracy*100)
        self.train_statistics["f1"].append(train_f1*100)
        self.train_statistics["mjww"].append(train_mjww*100)
        self.train_statistics["wtmf"].append(train_wtmf*100)
        self.train_statistics["wufpc"].append(train_wufpc)
        self.train_statistics["vocab_size"].append(train_vocab_size)
        self.train_statistics["avg_segment_len"].append(train_avg_segment_len)
        self.train_statistics["perplexity"].append(train_perplexity)
        self.train_statistics["iterations"].append(iteration)

        self.dev_statistics["accuracy"].append(dev_accuracy*100)
        self.dev_statistics["f1"].append(dev_f1*100)
        self.dev_statistics["mjww"].append(dev_mjww*100)
        self.dev_statistics["wtmf"].append(dev_wtmf*100)
        self.dev_statistics["wufpc"].append(dev_wufpc)
        self.dev_statistics["vocab_size"].append(dev_vocab_size)
        self.dev_statistics["avg_segment_len"].append(dev_avg_segment_len)
        self.dev_statistics["perplexity"].append(dev_perplexity)
        self.dev_statistics["iterations"].append(iteration)

        print("      {:>3d}.              | {:>6.2f}% / {:>6.2f}% | {:>6.2f}% / {:>6.2f}% | {:>13.2f} / {:>13.2f} | {:>5d} / {:>5d} |  {:>5.2f} / {:>5.2f}  | {:>6.2f}% / {:>6.2f}% | {:>6.2f}% / {:>6.2f}% | {:>5.2f} pitches / {:>5.2f} pitches |"
              .format(iteration, 
                    train_accuracy*100, dev_accuracy*100,
                    train_f1*100, dev_f1*100,
                    train_perplexity, dev_perplexity,
                    train_vocab_size, dev_vocab_size,
                    train_avg_segment_len, dev_avg_segment_len,
                    train_mjww*100, dev_mjww*100,
                    train_wtmf*100, dev_wtmf*100,
                    train_wufpc, dev_wufpc))
    
    def plot_all_statistics(self):
        """
        ToDo
        """
        # plot train statistics
        statistics_to_plot = {
            "Train Bacor - not tuned - Accuracy (%)": (self.train_statistics["iterations"], self.train_statistics["accuracy"]),
            "Train Bacor - not tuned - F1 (%)": (self.train_statistics["iterations"], self.train_statistics["f1"]),
            "Train Perplexity": (self.train_statistics["iterations"], self.train_statistics["perplexity"]),
            "Train Vocabulary Size": (self.train_statistics["iterations"], self.train_statistics["vocab_size"]),
            "Train Average Segment Length": (self.train_statistics["iterations"], self.train_statistics["avg_segment_len"]),
            "Train Melody Justified With Words (%)": (self.train_statistics["iterations"], self.train_statistics["mjww"]),
            "Train Weighted Top Mode Frequency (%)": (self.train_statistics["iterations"], self.train_statistics["wtmf"]),
            "Train Weighted Unique Final Pitch Count": (self.train_statistics["iterations"], self.train_statistics["wufpc"])
        }
        plot_iteration_statistics(statistics_to_plot)
        # plot dev statistics
        statistics_to_plot = {
            "Dev Bacor - not tuned - Accuracy (%)": (self.dev_statistics["iterations"], self.dev_statistics["accuracy"]),
            "Dev Bacor - not tuned - F1 (%)": (self.dev_statistics["iterations"], self.dev_statistics["f1"]),
            "Dev Perplexity": (self.dev_statistics["iterations"], self.dev_statistics["perplexity"]),
            "Dev Vocabulary Size": (self.dev_statistics["iterations"], self.dev_statistics["vocab_size"]),
            "Dev Average Segment Length": (self.dev_statistics["iterations"], self.dev_statistics["avg_segment_len"]),
            "Dev Melody Justified With Words (%)": (self.dev_statistics["iterations"], self.dev_statistics["mjww"]),
            "Dev Weighted Top Mode Frequency (%)": (self.dev_statistics["iterations"], self.dev_statistics["wtmf"]),
            "Dev Weighted Unique Final Pitch Count": (self.dev_statistics["iterations"], self.dev_statistics["wufpc"])
        }
        plot_iteration_statistics(statistics_to_plot)