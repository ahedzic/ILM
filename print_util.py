import pandas

class MetricTracker():
    def __init__(self):
        self.total_metrics = {}
        self.total_misses = 0.0
        self.total_metrics_attack = {}
        self.total_misses_attack = 0.0
        self.total_metrics_visible = {}
        self.total_misses_visible = 0.0
        self.total_count = {}
        self.distribution_total = []
        self.distribution_attack = []
        self.distribution_visible = []

    def print_metric_progression(self, edge_scores, edge_labels, id, edge_score_threshold):
        scores_dict = {}
        scores_dict_attack = {}
        scores_dict_visible = {}
        hits_total = 0
        hits_attack = 0
        hits_visible = 0
        fp_total = 0
        fp_attack = 0
        fp_visible = 0

        for e in range(len(edge_scores)):
            max_score = -1.0
            max_index = 0

            for s in range(len(edge_scores[e])):
                if edge_scores[e][s] > max_score:
                    max_score = edge_scores[e][s]
                    max_index = s

            max_score = float(max_score.cpu().detach())

            if (max_score > edge_score_threshold) and (edge_labels[e][max_index] > 0.0):
                scores_dict[max_score] = 'hit'

                if max_index == 1:
                    scores_dict_attack[max_score] = 'hit'
                if max_index == 6:
                    scores_dict_visible[max_score] = 'hit'
            else:
                if max_score > edge_score_threshold:
                    scores_dict[max_score] = 'fp'

                    if max_index == 1:
                        scores_dict_attack[max_score] = 'fp'
                    if max_index == 6:
                        scores_dict_visible[max_score] = 'fp'
                    
                for label_edge in range(len(edge_labels[e])):
                    if edge_labels[e][label_edge] > 0:
                        self.total_misses += 1.0
                        
                        if label_edge == 1:
                            self.total_misses_attack += 1.0
                        if label_edge == 6:
                            self.total_misses_visible += 1.0

        ordered_keys = sorted(scores_dict.keys(), reverse=True)[-int(len(scores_dict) * 0.2):]

        index = 0

        for i in range(len(ordered_keys)):
            if i in self.total_metrics.keys():
                self.total_metrics[i][scores_dict[ordered_keys[i]]] += 1
                self.total_count[i] += 1

                if i < 20:
                    if scores_dict[ordered_keys[i]] == 'hit':
                        hits_total += 1
                    elif scores_dict[ordered_keys[i]] == 'fp':
                        fp_total += 1
            else:
                new_dict = {'hit': 0, 'fp': 0}

                if scores_dict[ordered_keys[i]] == 'hit':
                    new_dict['hit'] = 1
                    if i < 20:
                        hits_total += 1
                elif scores_dict[ordered_keys[i]] == 'fp':
                    new_dict['fp'] = 1
                    if i < 20:
                        fp_total += 1

                self.total_metrics[i] = new_dict
                self.total_count[i] = 1

        ordered_keys_attack = sorted(scores_dict_attack.keys(), reverse=True)[-int(len(scores_dict_attack) * 0.2):]

        index = 0

        for i in range(len(ordered_keys_attack)):
            if i in self.total_metrics_attack.keys():
                self.total_metrics_attack[i][scores_dict_attack[ordered_keys_attack[i]]] += 1

                if i < 20:
                    if scores_dict_attack[ordered_keys_attack[i]] == 'hit':
                        hits_attack += 1
                    elif scores_dict_attack[ordered_keys_attack[i]] == 'fp':
                        fp_attack += 1
            else:
                new_dict = {'hit': 0, 'fp': 0}

                if scores_dict_attack[ordered_keys_attack[i]] == 'hit':
                    new_dict['hit'] = 1
                    if i < 20:
                        hits_attack += 1
                elif scores_dict_attack[ordered_keys_attack[i]] == 'fp':
                    new_dict['fp'] = 1
                    if i < 20:
                        fp_attack += 1

                self.total_metrics_attack[i] = new_dict

        ordered_keys_visible = sorted(scores_dict_visible.keys(), reverse=True)[-int(len(scores_dict_visible) * 0.2):]

        index = 0

        for i in range(len(ordered_keys_visible)):
            if i in self.total_metrics_visible.keys():
                self.total_metrics_visible[i][scores_dict_visible[ordered_keys_visible[i]]] += 1

                if i < 20:
                    if scores_dict_visible[ordered_keys_visible[i]] == 'hit':
                        hits_visible += 1
                    elif scores_dict_visible[ordered_keys_visible[i]] == 'fp':
                        fp_visible += 1
            else:
                new_dict = {'hit': 0, 'fp': 0}

                if scores_dict_visible[ordered_keys_visible[i]] == 'hit':
                    new_dict['hit'] = 1
                    if i < 20:
                        hits_visible += 1
                elif scores_dict_visible[ordered_keys_visible[i]] == 'fp':
                    new_dict['fp'] = 1
                    if i < 20:
                        fp_visible += 1

                self.total_metrics_visible[i] = new_dict

        if (hits_total+fp_total) > 0:
            self.distribution_total.append(hits_total / (hits_total+fp_total))
        else:
            self.distribution_total.append(0.0)
        if (hits_attack+fp_attack) > 0:
            self.distribution_attack.append(hits_attack / (hits_attack+fp_attack))
        else:
            self.distribution_attack.append(0.0)
        if (hits_visible+fp_visible) > 0:
            self.distribution_visible.append(hits_visible / (hits_visible+fp_visible))
        else:
            self.distribution_visible.append(0.0)

    def print_final(self, file_name):
        total_hits = 0.0
        total_fp = 0.0
        total_hits_attack = 0.0
        total_fp_attack = 0.0
        total_hits_visible = 0.0
        total_fp_visible = 0.0
        precision_values = []
        recall_values = []
        precision_values_attack = []
        recall_values_attack = []
        precision_values_visible = []
        recall_values_visible = []
        total_counts = []

        for i in range(len(self.total_metrics.keys()) - 20):
            total_hits = 0.0
            total_fp = 0.0

            for j in range(20):
                total_hits += self.total_metrics[i + j]['hit']
                total_fp += self.total_metrics[i + j]['fp']

            precision_values.append(total_hits/(total_hits+total_fp))
            recall_values.append(total_hits/(total_hits+self.total_misses))
            total_counts.append(self.total_count[i])

        if len(self.total_metrics_attack.keys()) > 20:
            for i in range(len(self.total_metrics_attack.keys()) - 20):
                total_hits_attack = 0.0
                total_fp_attack = 0.0

                for j in range(20):
                    total_hits_attack += self.total_metrics_attack[i + j]['hit']
                    total_fp_attack += self.total_metrics_attack[i + j]['fp']

                precision_values_attack.append(total_hits_attack/(total_hits_attack+total_fp_attack))
                recall_values_attack.append(total_hits_attack/(total_hits_attack+self.total_misses_attack))
        else:
            total_hits_attack = 0.0
            total_fp_attack = 0.0

            for j in range(len(self.total_metrics_attack.keys())):
                total_hits_attack += self.total_metrics_attack[j]['hit']
                total_fp_attack += self.total_metrics_attack[j]['fp']

            precision_values_attack.append(total_hits_attack/(total_hits_attack+total_fp_attack))
            recall_values_attack.append(total_hits_attack/(total_hits_attack+self.total_misses_attack))

        for i in range(len(self.total_metrics_visible.keys()) - 20):
            total_hits_visible = 0.0
            total_fp_visible = 0.0

            for j in range(20):
                total_hits_visible += self.total_metrics_visible[i + j]['hit']
                total_fp_visible += self.total_metrics_visible[i + j]['fp']
                
            precision_values_visible.append(total_hits_visible/(total_hits_visible+total_fp_visible))
            recall_values_visible.append(total_hits_visible/(total_hits_visible+self.total_misses_visible))

        print("Attack edges", total_hits_attack+self.total_misses_attack, "hits", total_hits_attack, "misses", self.total_misses_attack)
        print("Visible edges", total_hits_visible+self.total_misses_visible, "hits", total_hits_visible, "misses", self.total_misses_visible)
        final_dict = {'precision': precision_values, 'recall': recall_values, 'precision_attack': precision_values_attack, 'recall_attack': recall_values_attack, 'precision_visible': precision_values_visible, 'recall_visible': recall_values_visible, 'total_dist': self.distribution_total, 'attack_dist': self.distribution_attack, 'visible_dist': self.distribution_visible}
        df = pandas.DataFrame.from_dict(final_dict, orient='index')
        df.to_csv(file_name)
