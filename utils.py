import glob
import pickle
import random
import os.path
import pandas

class MetricsProcessor():
    def __init__(self, export_path):
        self.export_path = export_path
        self.final_dict = {}

    def add_metrics(self, model_name, metrics, total_labels):
        total_hits = 0
        total_discovered = 0

        hits = {}
        discovered = {}
        self.final_dict[model_name+"_precision"] = []
        self.final_dict[model_name+"_recall"] = []

        for graph_result in metrics:
            single_hits = 0.0
            single_discovered = 0.0
            single_total = 0.0
            precision = 0.0
            recall = 0.0

            if len(graph_result) > 0:
                single_total = float(graph_result[0][2])

            for edge in graph_result:
                if edge[1] != -1:
                    if edge[0] == edge[1]:
                        total_hits += 1.0
                        single_hits += 1.0

                        if edge[1] in hits.keys():
                            hits[edge[1]] += 1.0
                        else:
                            hits[edge[1]] = 1.0
                if edge[0] != -1:
                    total_discovered += 1.0
                    single_discovered += 1.0

                    if edge[0] in discovered.keys():
                        discovered[edge[0]] += 1.0
                    else:
                        discovered[edge[0]] = 1.0

            if (single_hits + single_discovered) > 0.0:
                precision = single_hits / single_discovered
            if single_total > 0.0:
                recall = single_hits / single_total

            self.final_dict[model_name+"_precision"].append(precision)
            self.final_dict[model_name+"_recall"].append(recall)

        print(model_name+"_precision", len(self.final_dict[model_name+"_precision"]))
        print(model_name+"_recall", len(self.final_dict[model_name+"_recall"]))
        print(hits)
        print(total_labels)

        total_true_edges = 0

        for true_edge in total_labels.keys():
            total_true_edges += total_labels[true_edge]

        precision = {
            'total': 0.0
        }
        recall = {
            'total': 0.0
        }

        if total_discovered > 0:
            precision = {
                'total': total_hits / total_discovered
            }

        if total_true_edges > 0:
            recall = {
                'total': total_hits / total_true_edges
            }

        for edge_index in hits.keys():
            discovered_total = 0
            true_total = 0

            if edge_index in discovered.keys():
                discovered_total = discovered[edge_index]
            if edge_index in total_labels.keys():
                true_total = total_labels[edge_index]

            if discovered_total > 0:
                precision[str(edge_index)] = hits[edge_index] / discovered_total
            else:
                precision[str(edge_index)] = 0.0

            if true_total > 0:
                recall[str(edge_index)] = hits[edge_index] / true_total
            else:
                recall[str(edge_index)] = 0

        print("Model Results:", model_name, flush=True)
        print("Precision", precision['total'], flush=True)
        print("Recall", recall['total'], flush=True)
        print("Hits", hits, flush=True)
        print("Discovered", discovered, flush=True)
        print("Totals", total_labels, flush=True)

    def export_metrics(self):
        pass
        #df = pandas.DataFrame.from_dict(self.final_dict, orient='columns')
        #df.to_csv("metrics.csv")

def dataset_load(dataset_name, dataset_path, train_ratio):
    train_name = dataset_name + '_train.pkl'
    test_name = dataset_name + '_test.pkl'
    train_set, test_set = None, None

    if os.path.isfile(train_name) and os.path.isfile(test_name):
        train = open(train_name, 'rb')
        test = open(test_name, 'rb')
        train_set = pickle.load(train)
        test_set = pickle.load(test)
        train.close()
        test.close()
    else:
        graph_list = glob.glob(os.path.join(dataset_path, '*.pkl'))
        random.shuffle(graph_list)
        test_size = int(len(graph_list) - (train_ratio * len(graph_list)))
        train_set = graph_list[test_size:]
        test_set = graph_list[:test_size]
        train = open(train_name, 'wb')
        pickle.dump(train_set, train)
        train.close()
        test = open(test_name, 'wb')
        pickle.dump(test_set, test)
        test.close()
        
    return train_set, test_set