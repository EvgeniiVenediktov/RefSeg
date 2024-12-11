https://www.overleaf.com/project/66ef3c0c7a071842be27fa15

ETRIS source code: https://github.com/kkakkkka/ETRIS

Branches and extensions:
- **main** - ```Task Specific Decoder by Sanga Park```
  - Linformer for Efficient Attention - To optimize the computational efficiency of the attention mechanism, we replaced the existing \texttt{nn.MultiheadAttention} module in the HA module with Linformer. Dynamic Position-Aware Encoding - Instead of the fixed position encoding provided by CoordConv in GA  module, we implemented a learnable positional encoding mechanism based on Dynamic Positional Encoding.
- **evgenii** - ```LoRA by Evgenii Venediktov```
  - Special LoRA.MultiheadAttention replaces regular MultiheadAttention in model.clip.py, LoRA rank set to model_d//2.
- **selective-feature-adaptation** - ```Selective Feature Adaptation by Uma Sharma```
  - the SelectiveAdaptationLayer replaces InteractorT in layers.py. SelectiveAdaptationLayer is used instead of InteractorT in the HA l(Heirarchical Alignment) layers. Run details are the same as ETRIS main code.
- **hyunwoo** - ```Cross-Modal Fusion by Hyunwoo Yu```
  - Cross-Modal Fusion
