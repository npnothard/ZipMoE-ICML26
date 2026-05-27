# Copyright (c) 2026 <ZipMoE / MINT, Nanjing University>.
# All rights reserved.
#
# This source code is licensed under the Academic Non-Commercial License.
# See the LICENSE file in the project root for details.


from collections import Counter, deque
import heapq

class ExpertPredictor:
    def __init__(self, n_sparse_layers, prefetcher_topk, window_size=26*6*10):
        self.freq_tables = [Counter() for _ in range(n_sparse_layers)]
        self.history = deque(maxlen=window_size)
        self.num_sparse_layers = n_sparse_layers
        self.prefetcher_topk = prefetcher_topk
        
    def update(self, layer_id, expert_ids):
        self.freq_tables[layer_id].update(expert_ids)
        self.history.append((layer_id, expert_ids))
        
    def predict_next_layer(self, current_layer):
        next_layer = ( current_layer + 1 ) % self.num_sparse_layers
        recent_activated = Counter()
        count = 0
        for layer, experts in reversed(self.history):
            if layer == next_layer:
                recent_activated.update(experts)
                count += 1
                if count == 5:
                    break
        scores = {}
        for expert_id in self.freq_tables[next_layer]:
            scores[expert_id] = (
                0.8 *  self.freq_tables[next_layer][expert_id] +
                0.2 * recent_activated.get(expert_id, 0) 
            )

        topk = heapq.nlargest(
            self.prefetcher_topk,
            scores.items(),
            key=lambda x: x[1]
        )

        return next_layer, [e for e, _ in topk]
        
        