from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel


class CudaLSTM(BaseModel):

    def __init__(self, cfg: dict):
        super(CudaLSTM, self).__init__(cfg=cfg)          # init parent, computes self.output_size

        self.hidden_size = cfg['hidden_size']             # LSTM memory capacity

        input_size = len(cfg['dynamic_inputs'])           # number of dynamic features (x_d)

        if cfg.get('static_attributes'):
            input_size += len(cfg['static_attributes'])  # append static feature count if present

        if cfg.get('use_basin_id_encoding', False):
            input_size += cfg['number_of_basins']        # append one-hot size if enabled

        self.lstm = nn.LSTM(
            input_size=input_size,                        # total features per time step
            hidden_size=self.hidden_size,                 # hidden state size
            num_layers=cfg.get('num_layers', 1),          # stacked LSTM layers, default 1
            bias=True,                                    # enable bias b
            batch_first=True,                             # input shape [batch, seq, features]
            dropout=cfg.get('output_dropout', 0.0),       # inter-layer dropout (needs num_layers>1)
        )

        self.dropout = nn.Dropout(p=cfg.get('output_dropout', 0.0))  # dropout before output head

        self.head = nn.Linear(self.hidden_size, self.output_size)     # hidden state → predictions


    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_d = data['x_d']                                # [batch, seq, n_dynamic] time series

        if 'x_s' in data:
            x_s = data['x_s'].unsqueeze(1).expand(-1, x_d.shape[1], -1)  # broadcast to all timesteps
            x_d = torch.cat([x_d, x_s], dim=-1)          # concat static features on feature dim

        if 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(1).expand(-1, x_d.shape[1], -1)  # same as x_s
            x_d = torch.cat([x_d, x_one_hot], dim=-1)    # concat basin id encoding

        lstm_out, (h_n, c_n) = self.lstm(x_d)            # run LSTM over full sequence

        lstm_out = self.dropout(lstm_out)                 # apply dropout

        y_hat = self.head(lstm_out)                       # linear layer → [batch, seq, output_size]

        return {'y_hat': y_hat, 'h_n': h_n, 'c_n': c_n} # predictions, hidden state, cell state